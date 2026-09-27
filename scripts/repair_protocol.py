"""Retry known rejected training tasks with a different teacher protocol.

No original sources, controls, reports, or heldout partitions are modified.
The preceding Pro collector must finish before this four-worker batch starts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import getpass
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
from scripts.fast_finish import qualified_paths, snapshot_exports


class ProtocolTeacher:
    model = "deepseek-v4-pro"
    max_output_tokens = 16384

    def __init__(self):
        self.key = os.environ["DEEPSEEK_API_KEY"]

    def request_token_bound(self, messages):
        return len(json.dumps(messages, ensure_ascii=False).encode()) + 1024 + self.max_output_tokens

    def complete(self, messages, *, temperature=0.0):
        import httpx
        messages = [dict(m) for m in messages]
        if not messages or messages[0]["role"] != "system":
            raise ValueError("system action protocol required")
        messages[0]["content"] += (
            '\nThis harness has no native function-call or XML tools. '
            'Return one JSON object with command, done and message only. '
            'Do not emit tool_calls, invoke, parameter, shell_command tags or whitespace-only content. '
            'Execute only one concrete shell command per response and wait for its real result.'
        )
        request = {"model": self.model, "messages": messages, "thinking": {"type": "enabled"},
                   "reasoning_effort": "low", "response_format": {"type": "text"},
                   "max_tokens": self.max_output_tokens}
        if len(json.dumps(request, ensure_ascii=False).encode()) > 100000:
            raise RuntimeError("teacher request exceeds guarded 100 KB input limit")
        headers = {"Authorization": "Bearer " + self.key}
        for attempt in range(4):
            response = httpx.get("https://api.deepseek.com/user/balance", headers=headers, timeout=20)
            response.raise_for_status()
            balance = response.json()
            entries = [e for e in balance.get("balance_infos", []) if e.get("currency") == "CNY"]
            if not balance.get("is_available") or len(entries) != 1:
                raise RuntimeError("teacher CNY balance unavailable")
            amount = Decimal(str(entries[0]["total_balance"]))
            if not amount.is_finite() or amount <= 1:
                raise RuntimeError("teacher balance too close to zero floor")
            try:
                response = httpx.post("https://api.deepseek.com/chat/completions", headers=headers,
                                      json=request, timeout=180)
            except httpx.RequestError as exc:
                raise RuntimeError("teacher outcome unknown; refusing automatic paid retry") from exc
            if response.status_code != 429 or attempt == 3:
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        payload = response.json()
        choice, usage = payload["choices"][0], payload.get("usage", {})
        content = choice["message"].get("content") or ""
        print("TEACHER_RECEIPT", json.dumps({"id": payload.get("id"), "model": payload.get("model"),
              "finish_reason": choice.get("finish_reason"), "content_chars": len(content),
              "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
              "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens")}), flush=True)
        # No CoT or reconstructed command is exported. The unchanged runtime
        # validates the actual response; a malformed action remains a failure.
        return ModelResponse(content, int(usage.get("prompt_tokens", 0)),
                             int(usage.get("completion_tokens", 0)), estimated_cost_cny=None)


def retry_candidates(source, pro, train, passed):
    candidates = []
    for task in train:
        outcome = pro / "outcomes" / (task + ".json")
        quality = source / "quality" / (task + ".json")
        if task not in passed and outcome.exists() and read(outcome).get("status") == "rejected":
            if quality.exists() and read(quality).get("admitted"):
                candidates.append(task)
    return candidates


def teacher_environment(safe, key):
    # CLI requires this guard even when the cohort installs its own adapter.
    return {**safe, "DEEPSEEK_API_KEY": key, "DEEPSEEK_MIN_BALANCE_CNY": "0"}


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from codeagentbench import cli
        if len(sys.argv) < 3 or sys.argv[2] != "run":
            raise ValueError("adapter may only run blind agent tasks")
        cli.DeepSeekModel = ProtocolTeacher
        return cli.main(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--pro", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--formal-root", type=Path, required=True)
    args = p.parse_args()
    if os.name == "nt" or not shutil.which("bwrap"):
        raise RuntimeError("working Linux bubblewrap required")
    source, pro, root, formal = (v.resolve() for v in (args.source, args.pro, args.root, args.formal_root))
    pool = source / "pool"
    train = read(pool / "split.json")["train"]
    for path in (root, root / "logs", root / "outcomes", root / "exports"):
        path.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (root / "collector.lock").open("a")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Await the preceding batch without overlapping more than four Pro workers.
    started = time.monotonic()
    while not (pro / "report.json").exists():
        if time.monotonic() - started > 1800:
            raise RuntimeError("preceding Pro batch did not finish; inspect, do not overlap")
        print("WAIT_PRECEDING_PRO_BATCH", flush=True)
        time.sleep(30)
    if read(pro / "report.json").get("status") == "uncertain":
        raise RuntimeError("preceding Pro batch has unknown outcome")
    passed = qualified_paths([source, pro, root], train)
    candidates = retry_candidates(source, pro, train, passed)
    identity = {"model": ProtocolTeacher.model, "thinking": "enabled/low", "response_format": "text",
                "max_output_tokens": 16384, "max_steps": 32, "max_tool_calls": 32,
                "max_tokens": 512000, "max_seconds": 600, "workers": 4, "target": 20,
                "balance_floor_cny": "0", "source": str(source), "preceding_pro": str(pro),
                "manifest_sha256": digest(pool / "manifest.json"), "split_sha256": digest(pool / "split.json"),
                "script_sha256": digest(Path(__file__)), "candidates": candidates}
    if (root / "identity.json").exists():
        raise RuntimeError("batch already has an identity; inspect receipts before restarting")
    write(root / "identity.json", identity)
    (root / ".repo_cache").symlink_to((source / ".repo_cache").resolve(), target_is_directory=True)
    safe = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()),
            "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
    api = None
    if len(passed) < 20:
        key = getpass.getpass("DeepSeek key for protocol retry (hidden): ")
        if not key:
            raise ValueError("API key required")
        api = teacher_environment(safe, key)
        del key
    def run(task):
        run_id = task + "-protocol-0"
        command = [sys.executable, "-m", "scripts.repair_protocol", "agent", "run", str(pool / "manifest.json"), task,
                   "--repo-root", str(root), "--run-id", run_id, "--max-steps", "32", "--max-tool-calls", "32",
                   "--max-tokens", "512000", "--max-seconds", "600", "--skip-evaluation"]
        with (root / "logs" / (run_id + "-rollout.log")).open("x") as stream:
            child = subprocess.Popen(command, env=api, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.wait(timeout=850)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        summary = root / "runs" / run_id / "summary.json"
        state = read(summary).get("status") if summary.exists() else "interrupted"
        result = {"task": task, "run_id": run_id, "agent_status": state,
                  "status": "uncertain" if state in {"interrupted", "cancelled"} else "rejected"}
        if result["status"] != "uncertain":
            code = execute(["evaluate-run", pool / "manifest.json", run_id, "--repo-root", root],
                           root / "logs" / (run_id + "-evaluation.log"), safe, 400)
            result["evaluation"] = "passed" if code == 0 else "failed_or_blocked"
            if code == 0 and state == "completed":
                target = root / "exports" / (task + ".jsonl")
                code = execute(["export-sft", root / "runs" / run_id / "events.jsonl", target],
                               root / "logs" / (run_id + "-export.log"), safe, 60)
                if code == 0 and target.exists() and seed_ok(read(target), train):
                    result["status"] = "accepted"
        write(root / "outcomes" / (task + ".json"), result)
        return result
    pending = iter(candidates); outcomes = []; halted = False
    with ThreadPoolExecutor(max_workers=4) as executor:
        running = {}
        while True:
            passed = qualified_paths([source, pro, root], train)
            while not halted and len(passed) < 20 and len(running) < min(4, 20 - len(passed)):
                task = next(pending, None)
                if task is None:
                    break
                running[executor.submit(run, task)] = task
            write(root / "progress.json", {"union_distinct_passed": len(passed), "target": 20,
                  "running": list(running.values()), "halted": halted, "outcomes": outcomes})
            if not running:
                break
            future = next(as_completed(running)); running.pop(future)
            result = future.result(); outcomes.append(result); halted = halted or result["status"] == "uncertain"
            print("PROTOCOL_RESULT", json.dumps(result), flush=True)
    passed = qualified_paths([source, pro, root], train)
    write(root / "report.json", {"union_distinct_passed": len(passed), "target": 20,
          "status": "uncertain" if halted else "completed" if len(passed) >= 20 else "insufficient_successes",
          "outcomes": outcomes})
    if halted or len(passed) < 20:
        raise RuntimeError("batch did not safely reach twenty successful train tasks")
    exports = snapshot_exports(root / "snapshot", passed, train)
    formal.mkdir(parents=True, exist_ok=True)
    if (formal / "pipeline.log").exists():
        print("EXISTING_TRAINING_PIPELINE", flush=True); return 0
    started = time.monotonic()
    with (formal / "pipeline.log").open("x") as stream:
        code = subprocess.run([sys.executable, "-m", "scripts.run_clean_training", "--exports", str(exports),
                               "--pool", str(pool), "--root", str(formal)], env=safe,
                              stdout=stream, stderr=subprocess.STDOUT).returncode
    write(formal / "pipeline-receipt.json", {"returncode": code, "seconds": time.monotonic() - started,
          "credential_free": True, "cohort_sources": [str(source), str(pro), str(root)]})
    print("FORMAL_PIPELINE_EXIT", code, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
