"""Precompute fixed heldout environment controls, never model outcomes.

Writes the same quality receipts consumed by run_clean_training. Run before
training starts; source manifest/split/runtime commit are recorded separately.
No provider credentials are copied into any subprocess.
"""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pool", type=Path, default=Path("data/expanded/moto-v2"))
    p.add_argument("--workers", type=int, default=2, choices=(1, 2))
    args = p.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("quality", "logs"):
        (root / directory).mkdir(exist_ok=True)
    if (root / "logs/training.log").exists():
        raise ValueError("training has started; do not race the main pipeline")
    split_path, manifest = args.pool / "split.json", args.pool / "manifest.json"
    split = json.loads(split_path.read_text())
    identity = {"commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "workers": args.workers, "tasks": split["dev"] + split["eval"]}
    meta = root / "preflight-controls.json"
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError("preflight identity changed")
    if not meta.exists():
        with meta.open("x") as stream:
            json.dump(identity, stream, indent=2)
    env = {"PATH": "/usr/bin:/bin", "HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve()), "CODEAGENTBENCH_EXECUTOR": "bwrap"}

    def run(task):
        output = root / "quality" / (task + ".json")
        log = root / "logs" / (task + "-quality.log")
        receipt = root / "logs" / (task + "-quality.receipt.json")
        if receipt.exists():
            return task, "existing"
        if log.exists():
            raise RuntimeError("uncertain preflight: " + task)
        command = [sys.executable, "-m", "codeagentbench", "quality-check", str(manifest), task, "--timeout", "120", "--output", str(output)]
        started = time.monotonic()
        with log.open("x") as stream:
            child = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=400)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
                code = 124
        payload = {"returncode": code, "seconds": time.monotonic() - started, "preflight": identity}
        temporary = receipt.with_suffix(".tmp")
        with temporary.open("x") as stream:
            json.dump(payload, stream)
        temporary.replace(receipt)
        admitted = json.loads(output.read_text())["admitted"] if output.exists() else False
        print(json.dumps({"task": task, "admitted": admitted, "returncode": code}), flush=True)
        return task, admitted

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run, identity["tasks"]))
    print(json.dumps({"controls_finished": len(results), "model_calls": 0}), flush=True)


if __name__ == "__main__":
    main()
