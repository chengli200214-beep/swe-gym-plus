"""A separate thinking-teacher cohort on exhausted TRAIN tasks, never heldout.

Existing collectors are not modified. The union is frozen before credential-free
SFT. A known HTTP receipt may be rejected; an unknown paid outcome is not replayed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from codeagentbench.adapters.model import ModelResponse
from scripts.accelerate_campaign import digest, execute, read, seed_ok, write


class ThinkingTeacher:
    model = "deepseek-v4-pro"
    max_output_tokens = 4096

    def __init__(self):
        self.key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.floor = Decimal(os.environ.get("DEEPSEEK_MIN_BALANCE_CNY", "0"))
        if not self.key or not self.floor.is_finite() or self.floor < 0:
            raise ValueError("valid API key and nonnegative balance floor required")

    def request_token_bound(self, messages):
        return len(json.dumps(messages, ensure_ascii=False).encode()) + 512 + self.max_output_tokens

    def complete(self, messages, *, temperature=0.0):
        import httpx
        request = {"model": self.model, "messages": messages, "thinking": {"type": "enabled"},
                   "reasoning_effort": "high", "response_format": {"type": "json_object"},
                   "max_tokens": self.max_output_tokens}
        if len(json.dumps(request, ensure_ascii=False).encode()) > 100000:
            raise RuntimeError("teacher request exceeds guarded 100 KB input limit")
        headers = {"Authorization": "Bearer " + self.key}
        for attempt in range(4):
            balance = httpx.get("https://api.deepseek.com/user/balance", headers=headers, timeout=20)
            balance.raise_for_status()
            b = balance.json()
            entries = [v for v in b.get("balance_infos", []) if v.get("currency") == "CNY"]
            if not b.get("is_available") or len(entries) != 1:
                raise RuntimeError("teacher balance unavailable; refusing paid request")
            amount = Decimal(str(entries[0]["total_balance"]))
            if not amount.is_finite() or amount <= self.floor + Decimal("1"):
                raise RuntimeError("teacher balance too close to configured floor")
            try:
                response = httpx.post("https://api.deepseek.com/chat/completions", headers=headers,
                                      json=request, timeout=120)
            except httpx.RequestError as exc:
                raise RuntimeError("teacher request outcome unknown; no automatic paid retry") from exc
            if response.status_code != 429 or attempt == 3:
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        # Reasoning is not exported or added to tool context. Its token usage is
        # still part of provider completion_tokens; no Flash price is reused.
        return ModelResponse(payload["choices"][0]["message"].get("content") or "",
                             int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                             estimated_cost_cny=None)


def qualified_paths(roots, train):
    result = {}
    for root in roots:
        for path in sorted((root / "exports").glob("*.jsonl")):
            try:
                row = read(path)
            except json.JSONDecodeError:
                continue  # Another writer may still be exporting this record.
            if seed_ok(row, train):
                result.setdefault(row["task_id"], path)
    return result


def exhausted_tasks(source, train):
    result = []
    for task in train:
        reports = [source / "outcomes" / (task + "-fast-" + str(i) + ".json") for i in range(2)]
        if all(p.exists() and read(p).get("status") == "rejected" for p in reports):
            quality = source / "quality" / (task + ".json")
            if quality.exists() and read(quality).get("admitted"):
                result.append(task)
    return result


def snapshot_exports(snapshot, paths, train):
    exports = snapshot / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    receipt = []
    for task in train:
        if task not in paths:
            continue
        source, target = paths[task], exports / (task + ".jsonl")
        if target.exists():
            if digest(target) != digest(source):
                raise ValueError("frozen training export changed")
        else:
            shutil.copy2(source, target)
        if not seed_ok(read(target), train):
            raise ValueError("copied export failed blind/completed/passed audit")
        receipt.append({"task_id": task, "source": str(source), "sha256": digest(target)})
    write(snapshot / "sources.json", {"distinct_tasks": len(receipt), "sources": receipt})
    return exports


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from codeagentbench import cli
        if len(sys.argv) < 3 or sys.argv[2] != "run":
            raise ValueError("thinking adapter is only allowed for blind agent runs")
        cli.DeepSeekModel = ThinkingTeacher
        return cli.main(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--formal-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--target", type=int, default=20)
    args = p.parse_args()
    if os.name == "nt" or not shutil.which("bwrap") or min(args.workers, args.target) < 1:
        raise ValueError("Linux bwrap and positive worker/target counts required")
    source, root, formal = args.source.resolve(), args.root.resolve(), args.formal_root.resolve()
    pool = source / "pool"
    train = read(pool / "split.json")["train"]
    for path in (root, root / "logs", root / "exports", root / "outcomes"):
        path.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "collector.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = {"source": str(source), "model": ThinkingTeacher.model, "thinking": "enabled/high",
                "max_output_tokens": 4096, "max_steps": 32, "max_tokens": 384000, "max_seconds": 600,
                "balance_floor_cny": "0", "workers": args.workers, "target": args.target,
                "manifest_sha256": digest(pool / "manifest.json"), "split_sha256": digest(pool / "split.json"),
                "script_sha256": digest(Path(__file__))}
    meta = root / "identity.json"
    if meta.exists() and read(meta) != identity:
        raise ValueError("thinking cohort identity changed")
    write(meta, identity)
    cache = root / ".repo_cache"
    if not cache.exists():
        cache.symlink_to((source / ".repo_cache").resolve(), target_is_directory=True)
    safe = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()),
            "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
    key = os.getenv("DEEPSEEK_API_KEY") or getpass.getpass("DeepSeek API key for thinking cohort (hidden): ")
    api = {**safe, "DEEPSEEK_API_KEY": key, "DEEPSEEK_MIN_BALANCE_CNY": "0"}
    del key
    def run(task):
        run_id = task + "-pro-0"
        outcome = root / "outcomes" / (task + ".json")
        if outcome.exists():
            return read(outcome)
        log = root / "logs" / (run_id + "-rollout.log")
        if log.exists():
            return {"task": task, "status": "uncertain"}
        command = [sys.executable, "-m", "scripts.fast_finish", "agent", "run", str(pool / "manifest.json"), task,
                   "--repo-root", str(root), "--run-id", run_id, "--max-steps", "32", "--max-tool-calls", "32",
                   "--max-tokens", "384000", "--max-seconds", "600", "--skip-evaluation"]
        with log.open("x") as stream:
            child = subprocess.Popen(command, env=api, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.wait(timeout=750)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        summary = root / "runs" / run_id / "summary.json"
        state = read(summary).get("status") if summary.exists() else "interrupted"
        result = {"task": task, "run_id": run_id, "agent_status": state, "status": "rejected"}
        if state in {"interrupted", "cancelled"}:
            result["status"] = "uncertain"
        else:
            code = execute(["evaluate-run", pool / "manifest.json", run_id, "--repo-root", root],
                           root / "logs" / (run_id + "-evaluation.log"), safe, 400)
            result["evaluation"] = "passed" if code == 0 else "failed_or_blocked"
            if code == 0 and state == "completed":
                target = root / "exports" / (task + ".jsonl")
                code = execute(["export-sft", root / "runs" / run_id / "events.jsonl", target],
                               root / "logs" / (run_id + "-export.log"), safe, 60)
                if code == 0 and target.exists() and seed_ok(read(target), train):
                    result["status"] = "accepted"
        write(outcome, result)
        return result
    tried = set(); outcomes = []; halted = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        running = {}
        while True:
            passed = qualified_paths([source, root], train)
            if len(passed) >= args.target or halted:
                break
            ready = [task for task in exhausted_tasks(source, train) if task not in passed and task not in tried]
            while ready and len(running) < min(args.workers, args.target - len(passed)):
                task = ready.pop(0); tried.add(task); running[executor.submit(run, task)] = task
            write(root / "progress.json", {"union_distinct_passed": len(passed), "target": args.target,
                  "halted": halted, "running": list(running.values()), "outcomes": outcomes})
            if not running:
                if (source / "collection-report.json").exists():
                    break
                time.sleep(15)
                continue
            future = next(as_completed(running)); running.pop(future)
            result = future.result(); outcomes.append(result); halted = result["status"] == "uncertain"
            print("THINKING_RESULT", json.dumps(result), "UNION", len(qualified_paths([source, root], train)), flush=True)
        for future in as_completed(running):
            result = future.result()
            outcomes.append(result)
            halted = halted or result["status"] == "uncertain"
    passed = qualified_paths([source, root], train)
    write(root / "report.json", {"union_distinct_passed": len(passed), "target": args.target,
          "status": "uncertain" if halted else "completed" if len(passed) >= args.target else "insufficient_successes",
          "outcomes": outcomes})
    if len(passed) < args.target or halted:
        raise RuntimeError("thinking cohort did not safely reach training minimum")
    exports = snapshot_exports(root / "snapshot", passed, train)
    # Exclusive pipeline.log also coordinates with the original waiting process.
    log = formal / "pipeline.log"
    if log.exists():
        print("EXISTING_TRAINING_PIPELINE", str(log), flush=True)
        return 0
    start = time.monotonic()
    with log.open("x") as stream:
        code = subprocess.run([sys.executable, "-m", "scripts.run_clean_training", "--exports", str(exports),
                               "--pool", str(pool), "--root", str(formal)], env=safe, stdout=stream, stderr=subprocess.STDOUT).returncode
    write(formal / "pipeline-receipt.json", {"returncode": code, "seconds": time.monotonic() - start,
          "credential_free": True, "cohort_sources": [str(source), str(root)]})
    print("FORMAL_PIPELINE_EXIT", code, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
