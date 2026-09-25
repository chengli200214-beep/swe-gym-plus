"""Bounded, resumable collection with credential-free admission/evaluation.

Runs are append-only. The key is obtained from the environment or getpass and
is passed only to rollout subprocesses, never saved in campaign artifacts.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from decimal import Decimal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=Path("data/splits/swegym-moto-v1.json"))
    parser.add_argument("--manifests", type=Path, nargs="+", default=[Path("data/manifests/swegym-smoke.json"), Path("data/manifests/swegym-moto-candidates.json")])
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--target", type=int, default=5)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--max-tokens", type=int, default=120000)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--prompt-key", action="store_true")
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("collection requires a Linux bubblewrap host")
    partitions = json.loads(args.split.read_text())
    if not set(args.tasks) <= set(partitions["train"]):
        raise ValueError("collection tasks must be assigned to train before running")
    manifests = {t["instance_id"]: p.resolve() for p in args.manifests for t in json.loads(p.read_text())["tasks"]}
    if not set(args.tasks) <= manifests.keys():
        raise ValueError("task missing from manifest")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "quality", "exports"):
        (root / name).mkdir(exist_ok=True)
    # Reuse already downloaded immutable base snapshots without exposing other
    # checkouts to tools (BashExecutor mounts only this task's object store).
    cache = root / ".repo_cache"
    source_cache = Path("artifacts/.repo_cache").resolve()
    if not cache.exists():
        cache.symlink_to(source_cache, target_is_directory=True)
    safe_env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()), "CODEAGENTBENCH_EXECUTOR": "bwrap"}
    key = getpass.getpass("DeepSeek API key (hidden): ") if args.prompt_key else os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DeepSeek key required")
    import httpx
    balance_response = httpx.get("https://api.deepseek.com/user/balance", headers={"Authorization": "Bearer " + key}, timeout=20)
    balance_response.raise_for_status()
    balances = [x for x in balance_response.json().get("balance_infos", []) if x.get("currency") == "CNY"]
    if len(balances) != 1:
        raise RuntimeError("CNY balance unavailable")
    initial_balance = Decimal(balances[0]["total_balance"])
    # Retain the user's 5.40 floor and keep this campaign within a two-yuan
    # balance window, with the adapter's extra 1-yuan request margin.
    floor = max(Decimal("5.40"), initial_balance - Decimal("2.00"))
    api_env = {**safe_env, "DEEPSEEK_API_KEY": key, "DEEPSEEK_MIN_BALANCE_CNY": str(floor), "DEEPSEEK_MAX_OUTPUT_TOKENS": "1024", "DEEPSEEK_MODEL": "deepseek-flash"}
    del key
    meta = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), "tasks": args.tasks, "target": args.target, "attempts": args.attempts, "max_steps": args.max_steps, "max_tokens": args.max_tokens, "max_seconds": args.max_seconds, "balance_floor_cny": str(floor), "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest()}
    meta_path = root / "campaign.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if any(previous[k] != meta[k] for k in ("tasks", "target", "max_steps", "max_tokens", "max_seconds", "split_sha256", "commit")):
            raise ValueError("campaign configuration changed; use a new root")
    else:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": "started", "tasks": len(args.tasks), "target": args.target, "balance_floor_cny": str(floor)}), flush=True)

    def run(command: list[str], log: Path, *, env=safe_env, timeout=600) -> int:
        with log.open("x") as output:
            try:
                return subprocess.run([sys.executable, "-m", "codeagentbench", *command], env=env, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False).returncode
            except subprocess.TimeoutExpired:
                output.write("\nCampaign process timeout\n")
                return 124

    passed = {p.stem for p in (root / "exports").glob("*.jsonl") if p.stat().st_size}
    results = []
    for task_id in args.tasks:
        if len(passed) >= args.target:
            break
        if task_id in passed:
            continue
        manifest = str(manifests[task_id])
        quality_path = root / "quality" / (task_id + ".json")
        if not quality_path.exists():
            quality_log = root / "logs" / (task_id + "-quality.log")
            if quality_log.exists():
                print(json.dumps({"task": task_id, "status": "blocked", "reason": "incomplete previous quality process"}), flush=True)
                continue
            run(["quality-check", manifest, task_id, "--timeout", "120", "--output", str(quality_path)], quality_log, timeout=400)
        if not quality_path.exists() or not json.loads(quality_path.read_text())["admitted"]:
            results.append({"task": task_id, "status": "not_admitted"})
            print(json.dumps(results[-1]), flush=True)
            continue
        for attempt in range(args.attempts):
            run_id = task_id + "-clean-" + str(attempt)
            rollout_log = root / "logs" / (run_id + "-rollout.log")
            if rollout_log.exists():
                # Do not repeat an uncertain paid call after restarting.
                continue
            run(["run", manifest, task_id, "--repo-root", str(root), "--run-id", run_id, "--max-steps", str(args.max_steps), "--max-tool-calls", str(args.max_steps), "--max-tokens", str(args.max_tokens), "--max-seconds", str(args.max_seconds), "--skip-evaluation"], rollout_log, env=api_env, timeout=args.max_seconds + 150)
            summary = root / "runs" / run_id / "summary.json"
            if not summary.exists():
                results.append({"task": task_id, "run_id": run_id, "status": "interrupted"})
                print(json.dumps(results[-1]), flush=True)
                # Authentication, balance, or provider failure: stop the whole
                # campaign instead of blindly issuing more paid requests.
                (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                return 2
            evaluation_log = root / "logs" / (run_id + "-evaluation.log")
            code = run(["evaluate-run", manifest, run_id, "--repo-root", str(root)], evaluation_log, timeout=400)
            result = {"task": task_id, "run_id": run_id, "status": json.loads(summary.read_text())["status"], "evaluation": "passed" if code == 0 else "failed_or_blocked"}
            results.append(result)
            print(json.dumps(result), flush=True)
            (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            if code == 0:
                exported = root / "exports" / (task_id + ".jsonl")
                run(["export-sft", str(root / "runs" / run_id / "events.jsonl"), str(exported)], root / "logs" / (run_id + "-export.log"), timeout=60)
                if exported.exists() and exported.stat().st_size:
                    passed.add(task_id)
                break
    print(json.dumps({"status": "completed" if len(passed) >= args.target else "insufficient_passed_tasks", "distinct_passed": len(passed), "target": args.target}), flush=True)
    return 0 if len(passed) >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
