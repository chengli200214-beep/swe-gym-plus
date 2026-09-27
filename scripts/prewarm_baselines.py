"""Credential-free Base-first prefetch on the frozen, already-admitted heldout set."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.run_clean_training import EVALUATION_LIMITS, model_source_receipt, paired_order, source_receipt, verify_preflight, write_json


def eligible_tasks(split, root):
    tasks = split["dev"] + split["eval"]
    for i, task in enumerate(tasks):
        quality = root / "quality" / (task + ".json")
        if paired_order(i)[0] == "base" and quality.exists() and json.loads(quality.read_text()).get("admitted") is True:
            yield task


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/mnt/workspace/models/Qwen2.5-Coder-3B-Instruct"))
    args = parser.parse_args()
    if os.name == "nt":
        raise RuntimeError("Linux GPU host required")
    import fcntl
    root, pool = args.root.resolve(), args.pool.resolve()
    with (root / "base-prefetch.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "pipeline.log").exists() or (root / "base-prefetch.json").exists():
            raise RuntimeError("pipeline/prefetch already exists; inspect, never duplicate")
        verify_preflight(root, pool)
        split = json.loads((pool / "split.json").read_text())
        tasks = list(eligible_tasks(split, root))
        identity = {"model_path": str(args.model.resolve()), "workers": 2, "tasks": tasks,
                    "order": "only original Base-first pairs; SFT follows training",
                    "manifest_sha256": hashlib.sha256((pool / "manifest.json").read_bytes()).hexdigest(),
                    "split_sha256": hashlib.sha256((pool / "split.json").read_bytes()).hexdigest(),
                    "source_sha256": source_receipt(Path(__file__).resolve().parents[1]),
                    "model_sha256": model_source_receipt(args.model), "limits": dict(EVALUATION_LIMITS)}
        for name in ("paired", "logs"):
            (root / name).mkdir(exist_ok=True)
        cache = root / "paired/.repo_cache"
        if not cache.exists():
            cache.symlink_to(Path("artifacts/.repo_cache").resolve(), target_is_directory=True)
        write_json(root / "base-prefetch.json", identity)
        env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()),
               "CODEAGENTBENCH_EXECUTOR": "bwrap", "TOKENIZERS_PARALLELISM": "false"}
        def execute(label, arguments, timeout):
            receipt, log = root / "logs" / (label + ".receipt.json"), root / "logs" / (label + ".log")
            if receipt.exists() or log.exists():
                raise RuntimeError("pre-existing baseline outcome; inspect: " + label)
            started = time.monotonic()
            with log.open("x") as stream:
                child = subprocess.Popen([sys.executable, "-m", "codeagentbench", *map(str, arguments)],
                    env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait(); code = 124
            write_json(receipt, {"returncode": code, "seconds": time.monotonic() - started, "prefetched": True})
            print(json.dumps({"prefetch_stage": label, "returncode": code}), flush=True)
        def run(task):
            run_id = task + "-base"
            command = ["run", pool / "manifest.json", task, "--repo-root", root / "paired", "--run-id", run_id,
                       "--model-backend", "local", "--model-path", args.model, "--skip-evaluation"]
            for name, value in identity["limits"].items():
                command += ["--" + name.replace("_", "-"), value]
            execute(run_id, command, 450)
            if (root / "paired/runs" / run_id / "summary.json").exists():
                execute(run_id + "-evaluate", ["evaluate-run", pool / "manifest.json", run_id, "--repo-root", root / "paired"], 400)
            return task
        with ThreadPoolExecutor(max_workers=2) as executor:
            finished = list(executor.map(run, tasks))
        write_json(root / "base-prefetch-report.json", {"status": "completed", "tasks": finished,
                   "credential_free": True, "same_budget": True, "kept_pair_order": True})
        print(json.dumps({"prefetch_finished": len(finished), "model_backend": "local", "api_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
