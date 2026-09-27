"""Six credential-free SFT rollouts comparing context policies; no training."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_clean_training import EVALUATION_LIMITS, source_receipt, write_json

POLICIES = ("last-action-v2", "recent-history-v3")
EXPECTED_TASKS = ["getmoto__moto-6082", "getmoto__moto-6212", "getmoto__moto-6028"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append(path, record):
    with path.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def child(trace, argv):
    from transformers import set_seed
    from codeagentbench.adapters.action import parse_action
    from codeagentbench.adapters.model import LocalHFModel
    from codeagentbench.cli import main as cli

    set_seed(0)
    original = LocalHFModel.complete
    calls = 0

    def observed(model, messages, **kwargs):
        nonlocal calls
        calls += 1
        view = model._prepare_messages(messages)
        latest_receipt = max((i for i, m in enumerate(view) if m["content"].startswith("Tool result:\n")), default=1)
        warnings = [m["content"] for m in view[latest_receipt + 1:]
                    if m["role"] == "user" and m["content"].startswith(("Harness warning:", "Harness checkpoint:"))]
        append(trace, {"type": "input", "call": calls, "policy": model.prompt_policy,
                       "temperature": kwargs.get("temperature"), "messages": view,
                       "latest_interventions": warnings,
                       "context_sha256": hashlib.sha256(json.dumps(view, ensure_ascii=False).encode()).hexdigest()})
        response = original(model, messages, **kwargs)
        try:
            action = asdict(parse_action(response.text))
        except ValueError:
            action = None
        append(trace, {"type": "output", "call": calls, "action": action,
                       "prompt_tokens": response.prompt_tokens, "completion_tokens": response.completion_tokens})
        return response

    LocalHFModel.complete = observed
    return cli(argv)


def summarize(run_dir, trace, evaluation_code):
    summary = read(run_dir / "summary.json")
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    verdicts = [e for e in events if e["type"] == "evaluation"]
    assert len(verdicts) == 1
    verdict = verdicts[0]["verdict"]
    assert evaluation_code == (0 if verdict == "passed" else 1)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    outputs = {r["call"]: r for r in rows if r["type"] == "output"}
    reactions = []
    for row in (r for r in rows if r["type"] == "input"):
        if not any(w.startswith("Harness warning: the same command") for w in row["latest_interventions"]):
            continue
        actions = [m for m in row["messages"] if m["role"] == "assistant"]
        previous = json.loads(actions[-1]["content"])["command"]
        following = outputs.get(row["call"], {}).get("action")
        reactions.append({"call": row["call"], "previous_command": previous,
                          "following_action": following,
                          "changed_action": bool(following and (following["done"] or following["command"] != previous))})
    return {"agent_status": summary["status"], "reason": summary["failure_reason"],
            "repeat_exit": "identical command repeated" in summary["failure_reason"],
            "patch_present": bool(summary["diff"]), "verdict": verdict,
            "steps": summary["steps"], "budget": summary["budget"],
            "repeat_warning_reactions": reactions,
            "max_history_turns": max((sum(m["role"] == "assistant" for m in r["messages"])
                                      for r in rows if r["type"] == "input"), default=0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    args = parser.parse_args()
    root, prior, pool = args.root.resolve(), args.prior.resolve(), args.pool.resolve()
    repo = Path(__file__).resolve().parents[1]
    prior_identity = read(prior / "experiment.json")
    assert read(prior / "verification.json")["status"] == "verified_completed"
    for name in ("manifest", "split"):
        assert sha(pool / (name + ".json")) == prior_identity[name + "_sha256"]
    selected = [p["task_id"] for p in read(prior / "pairs.json")
                if p["partition"] == "eval" and p["status"] == "evaluated"][:3]
    assert selected == EXPECTED_TASKS
    model = prior / "checkpoint"
    old_inventory = {x["path"]: x["sha256"] for x in read(prior / "final-assets.verified.json")["files"]}
    for name in ("checkpoint/adapter_model.safetensors", "checkpoint/adapter_config.json"):
        assert sha(prior / name) == old_inventory[name]
    import torch
    import importlib.metadata as md
    current_env = {"python": sys.version, "torch": torch.__version__, "rocm": torch.version.hip,
                   "gpu": torch.cuda.get_device_name(0),
                   "packages": {n: md.version(n) for n in ("transformers", "peft", "accelerate")}}
    old_env = read(prior / "environment.json")
    assert all(current_env[k] == old_env[k] for k in current_env)
    # Earlier controls are reused only with matching data, packages and evaluator.
    unchanged = ["src/codeagentbench/tasks/quality.py", "src/codeagentbench/adapters/swe_gym.py",
                 "src/codeagentbench/sandbox/executor.py", "src/codeagentbench/verification/evaluator.py"]
    assert all(sha(repo / p) == prior_identity["source_sha256"][p] for p in unchanged)
    controls = {}
    for task in selected:
        path = prior / "quality" / (task + ".json")
        control = read(path)
        assert control["admitted"] and sha(path) == old_inventory["quality/" + task + ".json"]
        controls[task] = sha(path)
    root.mkdir(exist_ok=False)
    for folder in ("logs", "paired", "traces"):
        (root / folder).mkdir()
    (root / "paired/.repo_cache").symlink_to((repo / "artifacts/.repo_cache").resolve(), target_is_directory=True)
    env = {"PATH": os.environ["PATH"], "HOME": "/tmp", "PYTHONPATH": str(repo / "src"),
           "PYTHONHASHSEED": "0", "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
    identity = {"tasks": selected, "policies": POLICIES, "seed": 0, "temperature": 0.2,
                "limits": EVALUATION_LIMITS, "model_path": str(model), "workers": 2,
                "adapter_sha256": sha(model / "adapter_model.safetensors"),
                "manifest_sha256": sha(pool / "manifest.json"), "split_sha256": sha(pool / "split.json"),
                "source_sha256": source_receipt(repo), "reused_quality_sha256": controls,
                "environment": current_env}
    write_json(root / "experiment.json", identity)

    def execute(label, command, child_env, timeout):
        start = time.monotonic()
        with (root / "logs" / (label + ".log")).open("x") as log:
            proc = subprocess.Popen(command, cwd=repo, env=child_env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        write_json(root / "logs" / (label + ".receipt.json"), {"returncode": code, "seconds": time.monotonic() - start})
        print(json.dumps({"stage": label, "returncode": code}), flush=True)
        return code

    def pair(index, task):
        record = {"task_id": task}
        order = POLICIES if index % 2 == 0 else POLICIES[::-1]
        for policy in order:
            run_id = task + "-" + policy
            trace = root / "traces" / (run_id + ".jsonl")
            command = [sys.executable, "-m", "scripts.probe_context_history", "child", str(trace),
                       "run", str(pool / "manifest.json"), task, "--repo-root", str(root / "paired"),
                       "--run-id", run_id, "--model-backend", "local", "--model-path", str(model), "--skip-evaluation"]
            for key, value in EVALUATION_LIMITS.items():
                command.extend(["--" + key.replace("_", "-"), str(value)])
            child_env = dict(env, CODEAGENTBENCH_LOCAL_PROMPT_POLICY=policy)
            code = execute(run_id, command, child_env, 450)
            run_dir = root / "paired/runs" / run_id
            if code or not (run_dir / "summary.json").exists():
                record[policy] = {"verdict": "blocked", "returncode": code, "reason": "rollout incomplete"}
                continue
            evaluation_code = execute(run_id + "-evaluate", [sys.executable, "-m", "codeagentbench", "evaluate-run",
                                      str(pool / "manifest.json"), run_id, "--repo-root", str(root / "paired")], env, 400)
            record[policy] = summarize(run_dir, trace, evaluation_code)
        return record

    completed = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(pair, i, task) for i, task in enumerate(selected)]
        for future in as_completed(futures):
            record = future.result()
            completed[record["task_id"]] = record
            write_json(root / "pairs.json", [completed[t] for t in selected if t in completed])
            print(json.dumps({"finished_task": record["task_id"]}), flush=True)
    assert source_receipt(repo) == identity["source_sha256"]
    assert sha(model / "adapter_model.safetensors") == identity["adapter_sha256"]
    report = {"status": "completed", "tasks": selected, "groups": {}, "limitations": [
        "Three previously inspected Moto tasks; diagnostic only", "Single fixed seed with sampling",
        "Same round-two adapter; no retraining", "More history consumes more of the same token budget",
        "Previous environment controls reused after identity checks", "No external private backup"]}
    for policy in POLICIES:
        records = [completed[t][policy] for t in selected]
        report["groups"][policy] = {"tasks": len(records), "repeat_exits": sum(r.get("repeat_exit", False) for r in records),
                                    "patches": sum(r.get("patch_present", False) for r in records),
                                    "passed": sum(r["verdict"] == "passed" for r in records),
                                    "blocked": sum(r["verdict"] == "blocked" for r in records)}
    write_json(root / "report.json", report)
    print("CONTEXT_PROBE_COMPLETE_JSON " + json.dumps(report), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        raise SystemExit(child(Path(sys.argv[2]), sys.argv[3:]))
    main()
