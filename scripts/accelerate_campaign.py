"""Fresh controls and parallel blind rollouts on an unchanged frozen split.

Private campaign data stays under --root. Old negative controls are summarized,
never reused. Keys are supplied through getpass and only enter rollout children.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quote_manifest(source):
    """Only repair shell encoding; retain every original selector and task."""
    result = deepcopy(source)
    changed = []
    for task in result["tasks"]:
        spec = task["eval_spec"]
        names = list(dict.fromkeys(spec["fail_to_pass"] + spec["pass_to_pass"]))
        old = "python -m pytest -q " + " ".join(names)
        if names and spec["test_command"] == old:
            new = shlex.join(["python", "-m", "pytest", "-q", *names])
            if new != old:
                spec["test_command"] = new
                changed.append(task["instance_id"])
    return result, changed


def rejection(report):
    if report.get("admitted"):
        return "admitted"
    controls = report.get("controls", [])
    if not controls:
        return report.get("reason") or "no_controls"
    if any(c.get("reason") == "isolated test unavailable" for c in controls):
        return "isolation_unavailable"
    gold = next((c for c in controls if c["name"] == "gold"), {})
    unfixed = next((c for c in controls if c["name"] == "unfixed"), {})
    if gold.get("reason"):
        return gold["reason"]
    if not gold.get("observed_pass"):
        return "gold_tests_failed"
    if unfixed.get("observed_pass"):
        return "unfixed_already_passes"
    return "control_mismatch"


def seed_ok(row, train):
    from codeagentbench.adapters.action import parse_action
    if row.get("task_id") not in train or row.get("evaluation_verdict") != "passed" or row.get("agent_status") != "completed":
        return False
    messages = row.get("messages", [])
    if not messages or messages[0].get("role") != "user":
        return False
    prompt = json.loads(messages[0]["content"])
    if prompt.get("instance_id") != row["task_id"]:
        return False
    if prompt.get("allowed_test_command") or prompt.get("test_patch") or prompt.get("gold_patch"):
        return False
    assistants = [m for m in messages if m.get("role") == "assistant"]
    return bool(assistants and parse_action(assistants[-1]["content"]).done)


def completed_controls(root, train):
    reports = {}
    for task in train:
        path = root / "quality" / (task + ".json")
        if not path.exists():
            continue
        try:
            report = read(path)
        except json.JSONDecodeError:
            continue
        if report.get("task_id") != task:
            raise ValueError("quality report task identity mismatch")
        reports[task] = report
    return reports


def execute(command, log, env, timeout):
    with log.open("x", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-m", "codeagentbench", *map(str, command)],
                                   env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            return 124


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--pool", type=Path, default=Path("data/expanded/moto-v2"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--quality-workers", type=int, default=6)
    p.add_argument("--api-workers", type=int, default=4)
    p.add_argument("--target", type=int, default=20)
    p.add_argument("--attempts", type=int, default=2)
    p.add_argument("--quality-only", action="store_true")
    p.add_argument("--collect-ready", action="store_true")
    args = p.parse_args()
    if os.name == "nt" or not shutil.which("bwrap"):
        raise RuntimeError("working Linux bubblewrap environment required")
    if min(args.quality_workers, args.api_workers, args.target, args.attempts) < 1:
        raise ValueError("worker counts, target and attempts must be positive")
    root = args.root.resolve()
    for d in (root, root / "pool", root / "logs", root / "quality", root / "exports", root / "outcomes"):
        d.mkdir(parents=True, exist_ok=True)
    split = read(args.pool / "split.json")
    train = split["train"]
    if len(train) != len(set(train)) or set(train).intersection(split["dev"] + split["eval"]):
        raise ValueError("invalid frozen partitions")
    manifest, changes = quote_manifest(read(args.pool / "manifest.json"))
    identity = {"source_manifest_sha256": digest(args.pool / "manifest.json"),
                "source_split_sha256": digest(args.pool / "split.json"), "script_sha256": digest(Path(__file__)),
                "source_root": str(args.source_root.resolve()), "target": args.target, "attempts": args.attempts,
                "quality_workers": args.quality_workers, "api_workers": args.api_workers,
                "model": "deepseek-flash", "max_steps": 32, "max_tokens": 240000, "max_seconds": 600,
                "balance_floor_cny": "0", "quoted_tasks": changes}
    if (root / "identity.json").exists() and read(root / "identity.json") != identity:
        previous = read(root / "identity.json")
        changed_fields = {k for k in set(previous) | set(identity) if previous.get(k) != identity.get(k)}
        if args.collect_ready and changed_fields == {"script_sha256"}:
            write(root / "script-update-receipt.json", {"previous_sha256": previous["script_sha256"],
                  "current_sha256": identity["script_sha256"], "reason": "enable collection alongside unchanged quality controls"})
        else:
            raise ValueError("campaign identity changed; use a new root")
    write(root / "identity.json", identity)
    write(root / "pool/manifest.json", manifest)
    if not (root / "pool/split.json").exists():
        shutil.copy2(args.pool / "split.json", root / "pool/split.json")
    cache = root / ".repo_cache"
    if not cache.exists():
        cache.symlink_to(Path("artifacts/.repo_cache").resolve(), target_is_directory=True)
    safe = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()),
            "CODEAGENTBENCH_EXECUTOR": "bwrap"}
    from codeagentbench.models import ToolIntent
    from codeagentbench.sandbox.executor import BashExecutor
    import tempfile
    with tempfile.TemporaryDirectory(prefix="cab-preflight-") as t:
        smoke = BashExecutor(t, backend="bwrap").execute(ToolIntent("preflight", "python -c 'import pytest,sure,responses,pytz,xmltodict,boto3'", t, 20))
        if smoke.exit_code != 0:
            raise RuntimeError("sandbox dependency preflight failed: " + smoke.stderr[-1000:])
    old = [read(q) for q in (args.source_root / "quality").glob("*.json")]
    seeds = [(s, read(s)) for s in (args.source_root / "exports").glob("*.jsonl")]
    write(root / "baseline-diagnosis.json", {"quality_reports": len(old),
          "reasons": dict(Counter(rejection(q) for q in old)),
          "clean_seed_tasks": [r["task_id"] for _, r in seeds if seed_ok(r, train)],
          "excluded_seeds": [r["task_id"] for _, r in seeds if not seed_ok(r, train)],
          "tasks": [{"task_id": q["task_id"], "reason": rejection(q)} for q in old]})
    def quality(task):
        path = root / "quality" / (task + ".json")
        if not path.exists():
            log = root / "logs" / (task + "-quality.log")
            if log.exists():
                return {"task_id": task, "admitted": False, "reason": "incomplete_quality_log", "controls": []}
            execute(["quality-check", root / "pool/manifest.json", task, "--timeout", 120, "--output", path], log, safe, 400)
        return read(path) if path.exists() else {"task_id": task, "admitted": False, "reason": "no_report", "controls": []}
    reports = completed_controls(root, train) if args.collect_ready else {}
    if not args.collect_ready:
        with ThreadPoolExecutor(max_workers=args.quality_workers) as pool:
            futures = [pool.submit(quality, task) for task in train]
            for future in as_completed(futures):
                q = future.result()
                reports[q["task_id"]] = q
                write(root / "quality-progress.json", {"checked": len(reports), "total": len(train),
                      "counts": dict(Counter(rejection(q) for q in reports.values()))})
                print(json.dumps({"phase": "quality", "checked": len(reports), "total": len(train), "task": q["task_id"], "result": rejection(q)}), flush=True)
    admitted = {task for task, q in reports.items() if q["admitted"]}
    summary_name = "ready-quality-summary.json" if args.collect_ready else "quality-summary.json"
    write(root / summary_name, {"checked": len(reports), "total": len(train), "admitted": sorted(admitted),
          "counts": dict(Counter(rejection(q) for q in reports.values()))})
    for source, row in seeds:
        target = root / "exports" / source.name
        if seed_ok(row, train) and row["task_id"] in admitted and not target.exists():
            shutil.copy2(source, target)
    passed = {read(s)["task_id"] for s in (root / "exports").glob("*.jsonl") if seed_ok(read(s), train)}
    print(json.dumps({"phase": "quality_complete", "admitted": len(admitted), "clean_seeds": len(passed), "target": args.target}), flush=True)
    if args.quality_only:
        return 0
    if len(admitted) < args.target:
        raise RuntimeError(f"only {len(admitted)} tasks admitted; inspect controls before paid collection")
    import fcntl
    collection_lock = (root / "collection.lock").open("a")
    fcntl.flock(collection_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    key = os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key (hidden): ")
    if not key:
        raise ValueError("API key required")
    api = {**safe, "DEEPSEEK_API_KEY": key, "DEEPSEEK_MIN_BALANCE_CNY": "0",
           "DEEPSEEK_MAX_OUTPUT_TOKENS": "1024", "DEEPSEEK_MODEL": "deepseek-flash"}
    import httpx
    balance = httpx.get("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + key}, timeout=20)
    balance.raise_for_status()
    del key
    if not balance.json().get("is_available"):
        raise RuntimeError("API account unavailable")
    halt = threading.Event()
    def rollout(task):
        result = {"task": task, "status": "rejected"}
        for attempt in range(args.attempts):
            if halt.is_set():
                return {"task": task, "status": "campaign_halted"}
            run_id = task + "-fast-" + str(attempt)
            outcome_path = root / "outcomes" / (run_id + ".json")
            if outcome_path.exists():
                previous = read(outcome_path)
                result = previous
                if previous.get("status") == "accepted":
                    return previous
                if previous.get("status") == "interrupted":
                    halt.set()
                    return previous
                continue
            log = root / "logs" / (run_id + "-rollout.log")
            if log.exists():
                halt.set()
                return {"task": task, "status": "uncertain_previous_call"}
            execute(["run", root / "pool/manifest.json", task, "--repo-root", root, "--run-id", run_id,
                     "--max-steps", 32, "--max-tool-calls", 32, "--max-tokens", 240000,
                     "--max-seconds", 600, "--skip-evaluation"], log, api, 750)
            summary_path = root / "runs" / run_id / "summary.json"
            summary = read(summary_path) if summary_path.exists() else {"status": "interrupted"}
            result = {"task": task, "run_id": run_id, "agent_status": summary["status"], "status": "rejected"}
            if summary["status"] in ("interrupted", "cancelled"):
                result["status"] = "interrupted"
                write(outcome_path, result)
                halt.set()
                return result
            code = execute(["evaluate-run", root / "pool/manifest.json", run_id, "--repo-root", root],
                           root / "logs" / (run_id + "-evaluation.log"), safe, 400)
            result["evaluation"] = "passed" if code == 0 else "failed_or_blocked"
            if code == 0 and summary["status"] == "completed":
                exported = root / "exports" / (task + ".jsonl")
                code = execute(["export-sft", root / "runs" / run_id / "events.jsonl", exported],
                               root / "logs" / (run_id + "-export.log"), safe, 60)
                if code == 0 and exported.exists() and seed_ok(read(exported), train):
                    result["status"] = "accepted"
            write(outcome_path, result)
            if result["status"] == "accepted":
                return result
        return result
    pending_tasks = iter(task for task in train if task in admitted and task not in passed)
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.api_workers) as pool:
        running = {}
        def fill():
            while not halt.is_set() and len(passed) < args.target and len(running) < min(args.api_workers, args.target - len(passed)):
                task = next(pending_tasks, None)
                if task is None:
                    break
                running[pool.submit(rollout, task)] = task
        fill()
        while running:
            future = next(as_completed(running))
            task = running.pop(future)
            result = future.result()
            outcomes.append(result)
            if result["status"] == "accepted":
                passed.add(task)
            write(root / "collection-progress.json", {"distinct_passed": len(passed), "target": args.target,
                  "halted": halt.is_set(), "outcomes": outcomes, "running": list(running.values())})
            print(json.dumps({"phase": "rollout", **result, "distinct_passed": len(passed), "target": args.target}), flush=True)
            fill()
    write(root / "collection-report.json", {"distinct_passed": len(passed), "target": args.target,
          "status": "completed" if len(passed) >= args.target else "halted" if halt.is_set() else "insufficient_successes",
          "passed_tasks": sorted(passed), "outcomes": outcomes})
    return 0 if len(passed) >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
