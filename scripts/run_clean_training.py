"""Credential-free, append-only data audit, BF16 LoRA and matched evaluation.

Run as `python -m scripts.run_clean_training`. This never calls DeepSeek.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.prepare_action_sft import prepare
from scripts.prepare_passed_sft import combine
from scripts.accelerate_campaign import seed_ok

EVALUATION_LIMITS = {"max_steps": 12, "max_tokens": 60000, "max_seconds": 300, "max_tool_calls": 12, "max_new_tokens": 768}


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def source_receipt(repo):
    """Identify actual source bytes, including uncommitted experiment fixes."""
    files = sorted((repo / "src/codeagentbench").rglob("*.py")) + sorted((repo / "scripts").rglob("*.py"))
    return {file.relative_to(repo).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}


def model_source_receipt(model):
    hashes = {}
    for file in sorted(model.glob("*")):
        if file.is_file() and not file.name.startswith("."):
            digest = hashlib.sha256()
            with file.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[file.name] = {"size": file.stat().st_size, "sha256": digest.hexdigest()}
    return hashes


def verify_preflight(root, pool):
    path = root / "quality-preflight.json"
    if path.exists():
        receipt = json.loads(path.read_text())
        for name in ("manifest", "split"):
            actual = hashlib.sha256((pool / (name + ".json")).read_bytes()).hexdigest()
            if receipt.get(name + "_sha256") != actual:
                raise ValueError("heldout preflight belongs to a different frozen " + name)


def paired_order(task_index):
    return ["base", "sft"] if task_index % 2 == 0 else ["sft", "base"]


def ordered_pairs(tasks, by_task):
    return [by_task[task] for task in tasks if task in by_task]


def wait_for_baseline_prefetch(root, pool, model, timeout=1200):
    """Join append-only Base-first work before the paired stage touches its logs."""
    identity = root / "base-prefetch.json"
    if not identity.exists():
        return
    import fcntl
    deadline = time.monotonic() + timeout
    with (root / "base-prefetch.lock").open("r") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Base prefetch has not drained; refusing duplicate inference")
                time.sleep(2)
        prefetch = json.loads(identity.read_text())
        for name in ("manifest", "split"):
            if prefetch[name + "_sha256"] != hashlib.sha256((pool / (name + ".json")).read_bytes()).hexdigest():
                raise ValueError("Base prefetch belongs to another frozen " + name)
        if prefetch["model_path"] != str(model.resolve()) or prefetch["source_sha256"] != source_receipt(Path(__file__).resolve().parents[1]):
            raise ValueError("Base prefetch model/source changed")
        if prefetch["model_sha256"] != model_source_receipt(model) or prefetch["limits"] != EVALUATION_LIMITS:
            raise ValueError("Base prefetch model bytes/budget changed")
        report = root / "base-prefetch-report.json"
        if not report.exists() or json.loads(report.read_text()).get("status") != "completed":
            raise RuntimeError("Base prefetch outcome incomplete; inspect before paired evaluation")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exports", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--minimum-tasks", type=int, default=20)
    p.add_argument("--model", type=Path, default=Path("/mnt/workspace/models/Qwen2.5-Coder-3B-Instruct"))
    p.add_argument("--pool", type=Path, default=Path("data/expanded/moto-v2"))
    p.add_argument("--config", type=Path, default=Path("configs/sft-clean-3b-amd.yaml"))
    p.add_argument("--eval-workers", type=int, default=2)
    p.add_argument("--action-data-mode", choices=["legacy", "last-action-v2"], default="legacy")
    args = p.parse_args()
    if args.eval_workers < 1:
        raise ValueError("evaluation workers must be positive")
    if os.name == "nt":
        raise ValueError("real experiment requires a Linux GPU host")
    root = args.root.resolve()
    split = json.loads((args.pool / "split.json").read_text())
    sources = sorted(args.exports.glob("*.jsonl"))
    if len(sources) < args.minimum_tasks:
        raise ValueError(f"need {args.minimum_tasks} distinct successful tasks; found {len(sources)}")
    for source in sources:
        row = json.loads(source.read_text())
        if not seed_ok(row, split["train"]):
            raise ValueError("training source must be blind, completed, passed and in the frozen train split")
    verify_preflight(root, args.pool)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("data", "logs", "quality", "paired"):
        (root / name).mkdir(exist_ok=True)
    cache = root / "paired" / ".repo_cache"
    if not cache.exists():
        cache.symlink_to(Path("artifacts/.repo_cache").resolve(), target_is_directory=True)
    env = {"PATH": os.environ["PATH"], "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()), "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
    if args.action_data_mode == "last-action-v2":
        env["CODEAGENTBENCH_LOCAL_PROMPT_POLICY"] = "last-action-v2"
    # Credential names and values are never copied to training/evaluation.
    def execute(label, module, arguments, timeout):
        receipt = root / "logs" / (label + ".receipt.json")
        log = root / "logs" / (label + ".log")
        if receipt.exists():
            return json.loads(receipt.read_text())["returncode"]
        if log.exists():
            raise RuntimeError(f"uncertain interrupted process: {label}; inspect before retry")
        started = time.monotonic()
        with log.open("x") as stream:
            process = subprocess.Popen([sys.executable, "-m", module, *map(str, arguments)], env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        write_json(receipt, {"returncode": code, "seconds": time.monotonic() - started})
        print(json.dumps({"stage": label, "returncode": code}), flush=True)
        return code

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    identity = {"commit": commit, "model_path": str(args.model.resolve()), "minimum_tasks": args.minimum_tasks,
                "source_sha256": source_receipt(Path(__file__).resolve().parents[1]),
                "manifest_sha256": hashlib.sha256((args.pool / "manifest.json").read_bytes()).hexdigest(),
                "evaluation_workers": args.eval_workers,
                "split_sha256": hashlib.sha256((args.pool / "split.json").read_bytes()).hexdigest(),
                "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "exports": {s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                "evaluation_limits": dict(EVALUATION_LIMITS)}
    if args.action_data_mode != "legacy":
        identity["action_data_mode"] = args.action_data_mode
    meta = root / "experiment.json"
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError("experiment identity changed; do not mix runs")
    write_json(meta, identity)
    import torch
    import importlib.metadata as md
    write_json(root / "environment.json", {"python": sys.version, "torch": torch.__version__, "rocm": torch.version.hip,
               "gpu": torch.cuda.get_device_name(0), "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
               "packages": {n: md.version(n) for n in ("transformers", "peft", "accelerate")}})
    model_receipt = root / "model-files.json"
    if not model_receipt.exists():
        hashes = model_source_receipt(args.model)
        write_json(model_receipt, {"source": "ModelScope Qwen/Qwen2.5-Coder-3B-Instruct", "requested_revision": "master", "note": "Downloaded snapshot identified by immutable file digests", "files": hashes})
    combined, actions = root / "data/trajectories.jsonl", root / "data/actions-raw.jsonl"
    if not combined.exists():
        combine(combined, sources)
    if args.action_data_mode == "legacy" and not actions.exists():
        prepare(combined, actions, context_chars=10000, include_done=True, history=True)
    audited = root / "data/actions.jsonl"
    audit_module = "scripts.prepare_aligned_sft" if args.action_data_mode == "last-action-v2" else "scripts.audit_training_data"
    audit_source = combined if args.action_data_mode == "last-action-v2" else actions
    if execute("audit", audit_module, [audit_source, audited, "--model", args.model, "--split", args.pool / "split.json"], 300):
        raise RuntimeError("data audit failed")
    audit = json.loads(audited.with_suffix(".audit.json").read_text())
    if audit["distinct_tasks"] < args.minimum_tasks:
        raise ValueError("too few distinct tasks survive token audit")
    checkpoint = root / "checkpoint"
    training_module = "scripts.train_aligned_sft" if args.action_data_mode == "last-action-v2" else "codeagentbench.train_sft"
    if execute("training", training_module, ["--config", args.config, "--train-file", audited, "--output-dir", checkpoint], 7200):
        raise RuntimeError("training failed; inspect logs")

    wait_for_baseline_prefetch(root, args.pool, args.model)
    manifest = args.pool / "manifest.json"
    tasks = split["dev"] + split["eval"]
    def evaluate_pair(task_index, task):
        quality = root / "quality" / (task + ".json")
        execute(task + "-quality", "codeagentbench", ["quality-check", manifest, task, "--timeout", 120, "--output", quality], 400)
        if not quality.exists() or not json.loads(quality.read_text())["admitted"]:
            return {"task_id": task, "partition": "dev" if task in split["dev"] else "eval", "status": "not_admitted"}
        pair = {"task_id": task, "partition": "dev" if task in split["dev"] else "eval", "status": "evaluated"}
        # Alternate order to reduce systematic warm-cache order bias.
        for label in paired_order(task_index):
            run_id = task + "-" + label
            limits = identity["evaluation_limits"]
            command = ["run", manifest, task, "--repo-root", root / "paired", "--run-id", run_id, "--model-backend", "local", "--model-path", args.model if label == "base" else checkpoint, "--skip-evaluation"]
            for name, value in limits.items():
                command += ["--" + name.replace("_", "-"), value]
            code = execute(run_id, "codeagentbench", command, 450)
            run_dir = root / "paired/runs" / run_id
            if (run_dir / "summary.json").exists():
                execute(run_id + "-evaluate", "codeagentbench", ["evaluate-run", manifest, run_id, "--repo-root", root / "paired"], 400)
                summary = json.loads((run_dir / "summary.json").read_text())
                events = [json.loads(s) for s in (run_dir / "events.jsonl").read_text().splitlines() if s]
                evaluation = next((e for e in reversed(events) if e.get("type") == "evaluation"), {})
                pair[label] = {"agent_status": summary["status"], "verdict": evaluation.get("verdict", "blocked"), "budget": summary.get("budget"), "patch_present": bool(summary.get("diff")), "reason": summary.get("failure_reason", "")}
            else:
                pair[label] = {"verdict": "blocked", "reason": "runner did not produce a summary", "returncode": code}
        return pair
    by_task = {}
    # Each pair remains sequential with alternating model order. Independent
    # tasks share the same bounded concurrency for both models; no labels select
    # tasks, model order or budgets. Only this coordinator writes pairs.json.
    with ThreadPoolExecutor(max_workers=args.eval_workers) as pool:
        futures = [pool.submit(evaluate_pair, i, task) for i, task in enumerate(tasks)]
        for future in as_completed(futures):
            pair = future.result()
            by_task[pair["task_id"]] = pair
            write_json(root / "pairs.json", ordered_pairs(tasks, by_task))
            print(json.dumps(pair), flush=True)
    pairs = ordered_pairs(tasks, by_task)
    report = {"training_audit": audit, "training_metrics": json.loads((checkpoint / "train_metrics.json").read_text()), "partitions": {}}
    for partition in ("dev", "eval"):
        items = [r for r in pairs if r["partition"] == partition]
        admitted = [r for r in items if r["status"] == "evaluated"]
        wins = sum(r["sft"]["verdict"] == "passed" and r["base"]["verdict"] != "passed" for r in admitted)
        losses = sum(r["base"]["verdict"] == "passed" and r["sft"]["verdict"] != "passed" for r in admitted)
        discordant = wins + losses
        exact_p = min(1.0, 2 * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
        report["partitions"][partition] = {"frozen_tasks": len(items), "admitted": len(admitted), "not_admitted": len(items) - len(admitted),
                    "base_passed": sum(r["base"]["verdict"] == "passed" for r in admitted), "sft_passed": sum(r["sft"]["verdict"] == "passed" for r in admitted),
                    "sft_only_passed": wins, "base_only_passed": losses, "paired_exact_p": exact_p}
        for model in ("base", "sft"):
            successes = report["partitions"][partition][model + "_passed"]
            report["partitions"][partition][model + "_wilson95_admitted"] = wilson(successes, len(admitted))
    report["limitations"] = ["Moto-only pilot; not full SWE-Gym performance", "Single training seed", "No external private backup", "Infrastructure exclusions are reported and not replaced"]
    write_json(root / "report.json", report)
    print(json.dumps({"status": "completed", "report": str(root / "report.json"), "partitions": report["partitions"]}), flush=True)


if __name__ == "__main__":
    main()
