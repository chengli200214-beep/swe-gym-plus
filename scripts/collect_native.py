"""Independent, receipt-grounded native-tool teacher cohort; no gold exposure."""
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
from scripts.fast_finish import qualified_paths, snapshot_exports
from scripts.repair_protocol_next import teacher_environment, verify_drained_receipts


TOOLS = [
    {"type": "function", "function": {"name": "execute_shell", "description":
     "Run exactly one shell command in the isolated task repository. Wait for the actual tool result.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"},
     "message": {"type": "string"}}, "required": ["command"], "additionalProperties": False}}},
    {"type": "function", "function": {"name": "finish_task", "description":
     "Finish after a concrete patch and a real post-edit test or smoke check. Independent tests still judge success.",
     "parameters": {"type": "object", "properties": {"message": {"type": "string"}},
     "required": ["message"], "additionalProperties": False}}},
]


class NativeTeacher:
    """Preserve native conversation privately; only real journal receipts are tools.

    reasoning_content is retained in process memory and returned to the same
    official API, as required for thinking-mode tool use. It is never exported
    to training data or printed. Native arguments become the existing action
    schema; no synthetic tool result or reconstructed command is executed.
    """
    model = "deepseek-v4-pro"
    max_output_tokens = 16384

    def __init__(self):
        self.key = os.environ["DEEPSEEK_API_KEY"]
        self.journal = Path(os.environ["CODEAGENTBENCH_NATIVE_JOURNAL"])
        self.history = []
        self.pending = None
        self.seen_receipts = set()

    def _sync(self, messages):
        if not self.history:
            if len(messages) < 2 or messages[0]["role"] != "system" or messages[1]["role"] != "user":
                raise ValueError("blind initial system/task prompt required")
            system = messages[0]["content"].replace(
                'Return exactly JSON: {"command":"...", "done":false, "message":"..."}. ',
                "Use the execute_shell and finish_task native tools, one call per response. ")
            system += (" Never write XML tool transcripts or invent tool outputs. "
                       "Previous native tool messages are actual execution receipts. "
                       "Keep reads short (about 60 lines). After locating the target, edit instead of rereading it.")
            self.history = [{"role": "system", "content": system}, dict(messages[1])]
        if self.pending:
            call_id, command = self.pending
            records = [json.loads(line) for line in self.journal.read_text().splitlines() if line.strip()]
            receipts = [r for r in records if r.get("type") == "receipt" and r.get("action_id") not in self.seen_receipts]
            if len(receipts) != 1 or receipts[0].get("command") != command:
                raise RuntimeError("native pending call lacks a unique matching real receipt")
            receipt = receipts[0]
            observed = {k: receipt[k] for k in ("exit_code", "stdout", "stderr", "timed_out")}
            self.history.append({"role": "tool", "tool_call_id": call_id, "content": json.dumps(observed, ensure_ascii=False)})
            self.seen_receipts.add(receipt["action_id"])
            self.pending = None
        return self.history

    def request_token_bound(self, messages):
        # Synchronize once, before the runtime checkpoints a paid request. This
        # includes the real receipt even if runtime context compression lost it.
        return len(json.dumps(self._sync(messages), ensure_ascii=False).encode()) + 2048 + self.max_output_tokens

    def complete(self, messages, *, temperature=0.0):
        import httpx
        request = {"model": self.model, "messages": self._sync(messages), "tools": TOOLS,
                   "thinking": {"type": "enabled"}, "reasoning_effort": "high",
                   "max_tokens": self.max_output_tokens}
        # No forced tool_choice: it is incompatible with thinking mode.
        if len(json.dumps(request, ensure_ascii=False).encode()) > 300000:
            raise RuntimeError("native conversation exceeds guarded 300 KB request limit")
        headers = {"Authorization": "Bearer " + self.key}
        for attempt in range(4):
            balance_response = httpx.get("https://api.deepseek.com/user/balance", headers=headers, timeout=20)
            balance_response.raise_for_status()
            balance = balance_response.json()
            entries = [e for e in balance.get("balance_infos", []) if e.get("currency") == "CNY"]
            if not balance.get("is_available") or len(entries) != 1:
                raise RuntimeError("teacher CNY balance unavailable")
            amount = Decimal(str(entries[0]["total_balance"]))
            if not amount.is_finite() or amount <= 1:
                raise RuntimeError("teacher balance too close to zero floor")
            try:
                response = httpx.post("https://api.deepseek.com/chat/completions", headers=headers, json=request, timeout=180)
            except httpx.RequestError as exc:
                raise RuntimeError("teacher outcome unknown; refusing automatic paid retry") from exc
            if response.status_code != 429 or attempt == 3:
                break
            time.sleep(2 ** attempt)
        response.raise_for_status()
        payload = response.json()
        choice, usage = payload["choices"][0], payload.get("usage", {})
        message = choice["message"]
        calls = message.get("tool_calls") or []
        receipt = {"id": payload.get("id"), "model": payload.get("model"),
                   "finish_reason": choice.get("finish_reason"), "native_calls": len(calls),
                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
                   "reasoning_tokens": usage.get("completion_tokens_details", {}).get("reasoning_tokens"),
                   "arguments_sha256": [hashlib.sha256(c.get("function", {}).get("arguments", "").encode()).hexdigest() for c in calls]}
        print("TEACHER_RECEIPT", json.dumps(receipt), flush=True)
        text = json.dumps({"invalid_native_response": "expected one complete validated native tool call"})
        if choice.get("finish_reason") != "length" and len(calls) == 1:
            call = calls[0]
            try:
                arguments = json.loads(call["function"]["arguments"])
                name = call["function"]["name"]
                if not isinstance(arguments, dict) or not isinstance(call.get("id"), str) or not call["id"]:
                    raise ValueError("invalid native call")
                if name == "execute_shell":
                    if set(arguments) - {"command", "message"} or not isinstance(arguments.get("command"), str) or not arguments["command"].strip():
                        raise ValueError("invalid shell arguments")
                    if not isinstance(arguments.get("message", ""), str):
                        raise ValueError("invalid shell message")
                    command = arguments["command"].strip()
                    self.pending = (call["id"], command)
                    text = json.dumps({"command": command, "done": False, "message": arguments.get("message", "")})
                elif name == "finish_task":
                    if set(arguments) != {"message"} or not isinstance(arguments["message"], str):
                        raise ValueError("invalid finish arguments")
                    text = json.dumps({"command": "", "done": True, "message": arguments["message"]})
                else:
                    raise ValueError("unknown native tool")
                self.history.append({"role": "assistant", "content": message.get("content") or "",
                                     "reasoning_content": message.get("reasoning_content") or "", "tool_calls": calls})
            except (ValueError, KeyError, TypeError):
                pass
        return ModelResponse(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)), estimated_cost_cny=None)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "agent":
        from codeagentbench import cli
        if len(sys.argv) < 3 or sys.argv[2] != "run":
            raise ValueError("native adapter only runs blind training tasks")
        cli.DeepSeekModel = NativeTeacher
        return cli.main(sys.argv[2:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pro", type=Path, required=True)
    parser.add_argument("--preceding", type=Path, action="append", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()
    if os.name == "nt" or not shutil.which("bwrap"):
        raise RuntimeError("working Linux bubblewrap required")
    source, pro, root, formal = [p.resolve() for p in (args.source, args.pro, args.root, args.formal_root)]
    pool = source / "pool"
    train = read(pool / "split.json")["train"]
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    locks = []
    for path in [root, *[p.resolve() for p in args.preceding]]:
        handle = (path / "collector.lock").open("a" if path == root else "r")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(handle)
    preceding_receipts = [verify_drained_receipts(p) for p in args.preceding]
    if read(pro / "report.json").get("status") == "uncertain":
        raise RuntimeError("preceding Pro result is unknown")
    cohort_sources = [source, pro, *args.preceding, root]
    passed = qualified_paths(cohort_sources, train)
    candidates = [t for t in train if t not in passed and (source / "quality" / (t + ".json")).exists()
                  and read(source / "quality" / (t + ".json")).get("admitted")
                  and (pro / "outcomes" / (t + ".json")).exists()
                  and read(pro / "outcomes" / (t + ".json")).get("status") == "rejected"]
    # Training-only ordering based on public issue length; heldout tasks remain frozen.
    tasks = read(pool / "manifest.json")
    if isinstance(tasks, dict):
        tasks = tasks.get("tasks", [])
    lengths = {t["instance_id"]: len(t.get("issue", "")) for t in tasks}
    candidates.sort(key=lambda t: (lengths.get(t, 0), t))
    if (root / "identity.json").exists():
        raise RuntimeError("native cohort already exists; inspect, do not restart")
    identity = {"model": NativeTeacher.model, "thinking": "enabled/high", "protocol": "native-tools-with-real-receipts",
                "reasoning_storage": "process-memory-only; same-official-API replay", "workers": 4, "target": 20,
                "max_steps": 32, "max_tool_calls": 32, "max_tokens": 512000, "max_seconds": 600,
                "max_output_tokens": 16384, "candidates": candidates, "preceding_receipts": preceding_receipts,
                "script_sha256": digest(Path(__file__)), "manifest_sha256": digest(pool / "manifest.json"),
                "split_sha256": digest(pool / "split.json")}
    write(root / "identity.json", identity)
    for folder in ("logs", "outcomes", "exports"):
        (root / folder).mkdir(exist_ok=True)
    (root / ".repo_cache").symlink_to((source / ".repo_cache").resolve(), target_is_directory=True)
    safe = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()),
            "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
    api = teacher_environment(safe, getpass.getpass("DeepSeek native tool key (hidden): "), "high") if len(passed) < 20 else None
    def run(task):
        run_id = task + "-native-0"
        env = {**api, "CODEAGENTBENCH_NATIVE_JOURNAL": str(root / "runs" / run_id / "actions.jsonl")}
        command = [sys.executable, "-m", "scripts.collect_native", "agent", "run", str(pool / "manifest.json"), task,
                   "--repo-root", str(root), "--run-id", run_id, "--max-steps", "32", "--max-tool-calls", "32",
                   "--max-tokens", "512000", "--max-seconds", "600", "--skip-evaluation"]
        with (root / "logs" / (run_id + "-rollout.log")).open("x") as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                child.wait(timeout=850)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        summary = root / "runs" / run_id / "summary.json"
        state = read(summary).get("status") if summary.exists() else "interrupted"
        result = {"task": task, "run_id": run_id, "agent_status": state,
                  "status": "uncertain" if state in {"interrupted", "cancelled"} else "rejected"}
        if result["status"] != "uncertain":
            code = execute(["evaluate-run", pool / "manifest.json", run_id, "--repo-root", root], root / "logs" / (run_id + "-evaluation.log"), safe, 400)
            result["evaluation"] = "passed" if code == 0 else "failed_or_blocked"
            if code == 0 and state == "completed":
                target = root / "exports" / (task + ".jsonl")
                code = execute(["export-sft", root / "runs" / run_id / "events.jsonl", target], root / "logs" / (run_id + "-export.log"), safe, 60)
                if code == 0 and target.exists() and seed_ok(read(target), train):
                    result["status"] = "accepted"
        write(root / "outcomes" / (task + ".json"), result)
        return result
    pending = iter(candidates); outcomes = []; halted = False
    with ThreadPoolExecutor(max_workers=4) as executor:
        running = {}
        while True:
            passed = qualified_paths(cohort_sources, train)
            while not halted and len(passed) < 20 and len(running) < min(4, 20 - len(passed)):
                task = next(pending, None)
                if task is None: break
                running[executor.submit(run, task)] = task
            write(root / "progress.json", {"union_distinct_passed": len(passed), "target": 20,
                  "running": list(running.values()), "halted": halted, "outcomes": outcomes})
            if not running: break
            future = next(as_completed(running)); running.pop(future)
            result = future.result(); outcomes.append(result); halted |= result["status"] == "uncertain"
            print("NATIVE_RESULT", json.dumps(result), flush=True)
    passed = qualified_paths(cohort_sources, train)
    write(root / "report.json", {"union_distinct_passed": len(passed), "target": 20,
          "status": "uncertain" if halted else "completed" if len(passed) >= 20 else "insufficient_successes", "outcomes": outcomes})
    if halted or len(passed) < 20:
        raise RuntimeError("native cohort did not safely reach twenty successful training tasks")
    exports = snapshot_exports(root / "snapshot", passed, train)
    formal.mkdir(parents=True, exist_ok=True)
    if (formal / "pipeline.log").exists():
        raise RuntimeError("training pipeline already exists; inspect, do not duplicate")
    started = time.monotonic()
    with (formal / "pipeline.log").open("x") as stream:
        code = subprocess.run([sys.executable, "-m", "scripts.run_clean_training", "--exports", str(exports),
                               "--pool", str(pool), "--root", str(formal)], env=safe, stdout=stream, stderr=subprocess.STDOUT).returncode
    write(formal / "pipeline-receipt.json", {"returncode": code, "seconds": time.monotonic() - started,
          "credential_free": True, "cohort_sources": [str(p) for p in cohort_sources]})
    print("FORMAL_PIPELINE_EXIT", code, flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
