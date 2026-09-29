"""Collect one admitted train task with a blind paid rollout and keyless judging.

This is intentionally a one-attempt collector: an interrupted API request has
an unknown outcome and must never be silently retried. Only the independently
passed, normally completed trajectory is exported for aligned training.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import getpass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from codeagentbench.models import EvalSpec
from codeagentbench.tasks.manifest import Manifest, load_manifest, save_manifest


def prepare(
    source: Path, split: Path, quality: Path, cache_root: Path,
    root: Path, task_id: str, public_test: str,
) -> tuple[Path, Path]:
    """Freeze a public-file command without exposing evaluation labels to rollout."""
    partitions = json.loads(split.read_text(encoding="utf-8"))
    if task_id not in partitions["train"]:
        raise ValueError("paid collection is restricted to the frozen train split")
    admission = json.loads(quality.read_text(encoding="utf-8"))
    if admission.get("task_id") != task_id or admission.get("admitted") is not True:
        raise ValueError("the task lacks a matching passed quality control")
    manifest = load_manifest(source)
    task = next((row for row in manifest.tasks if row.instance_id == task_id), None)
    if task is None or task.split != "train":
        raise ValueError("the source manifest does not assign this task to train")
    relative = Path(public_test)
    if (relative.is_absolute() or relative.parts[:1] != ("tests",)
            or any(part in {"..", "."} for part in relative.parts)
            or relative.suffix != ".py"):
        raise ValueError("visible test must be a public Python file under tests/")
    cache_key = hashlib.sha256(f"{task.repo}\0{task.base_commit}".encode()).hexdigest()[:24]
    snapshot = (cache_root / cache_key).resolve(strict=True)
    test_file = (snapshot / relative).resolve(strict=True)
    if not test_file.is_relative_to(snapshot) or not test_file.is_file():
        raise ValueError("visible test is not a regular file in the immutable base")
    if not (snapshot / ".git").is_dir():
        raise ValueError("base snapshot is not a Git checkout")
    public_task = replace(task, metadata={**task.metadata,
        "agent_test_command": f"python -m pytest -q -x {relative.as_posix()}"})
    root.mkdir(parents=True, exist_ok=False)
    cache_link = root / ".repo_cache"
    cache_link.symlink_to(cache_root.resolve(strict=True), target_is_directory=True)
    blind = root / "blind-train-manifest.json"
    formal = root / "formal-train-manifest.json"
    save_manifest(Manifest(manifest.dataset, manifest.revision,
                           (replace(public_task, eval_spec=EvalSpec()),)), blind)
    save_manifest(Manifest(manifest.dataset, manifest.revision, (public_task,)), formal)
    (root / "identity.json").write_text(json.dumps({
        "task_id": task_id,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
        "quality_sha256": hashlib.sha256(quality.read_bytes()).hexdigest(),
        "visible_test_file": relative.as_posix(),
        "visible_test_sha256": hashlib.sha256(test_file.read_bytes()).hexdigest(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }, indent=2) + "\n", encoding="utf-8")
    return blind, formal


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int) -> int:
    with log.open("x", encoding="utf-8") as output:
        process = subprocess.Popen(command, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            output.write("\nCollector timeout; paid request outcome may be unknown.\n")
            return 124


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--quality", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--public-test", required=True)
    parser.add_argument("--minimum-balance-cny", type=Decimal, default=Decimal("5.40"))
    args = parser.parse_args()
    if os.name == "nt" or os.environ.get("CODEAGENTBENCH_EXECUTOR") != "nsjail":
        raise RuntimeError("this collector requires the verified Linux NsJail executor")
    if not os.environ.get("CODEAGENTBENCH_ROOTFS"):
        raise RuntimeError("the verified NsJail rootfs must be specified")
    blind, formal = prepare(args.source, args.split, args.quality, args.cache_root,
                            args.root, args.task_id, args.public_test)
    safe_env = {name: os.environ[name] for name in (
        "PATH", "CODEAGENTBENCH_EXECUTOR", "CODEAGENTBENCH_ROOTFS",
        "CODEAGENTBENCH_LOCAL_PROMPT_POLICY") if name in os.environ}
    safe_env.update({"HOME": "/tmp", "PYTHONPATH": str(Path("src").resolve())})
    key = getpass.getpass("DeepSeek train-only key (hidden): ")
    if not key:
        raise ValueError("API key missing")
    import httpx
    response = httpx.get("https://api.deepseek.com/user/balance",
                         headers={"Authorization": "Bearer " + key}, timeout=20)
    response.raise_for_status()
    payload = response.json()
    balances = [row for row in payload.get("balance_infos", []) if row.get("currency") == "CNY"]
    if not payload.get("is_available") or len(balances) != 1:
        raise RuntimeError("CNY balance unavailable; no paid rollout started")
    amount = Decimal(str(balances[0]["total_balance"]))
    if not amount.is_finite() or amount <= args.minimum_balance_cny + Decimal("1"):
        raise RuntimeError("balance too close to the configured floor")
    floor = max(args.minimum_balance_cny, amount - Decimal("2"))
    paid_env = {**safe_env, "DEEPSEEK_API_KEY": key,
                "DEEPSEEK_MIN_BALANCE_CNY": str(floor),
                "DEEPSEEK_MAX_OUTPUT_TOKENS": "1024", "DEEPSEEK_MODEL": "deepseek-flash"}
    del key
    run_id = args.task_id + "-train-aligned-1"
    base = [sys.executable, "-m", "codeagentbench"]
    code = _run(base + ["run", str(blind), args.task_id, "--repo-root", str(args.root),
                        "--run-id", run_id, "--require-visible-test-before-done",
                        "--max-steps", "24", "--max-tool-calls", "24",
                        "--max-tokens", "180000", "--max-seconds", "600",
                        "--skip-evaluation"], args.root / "rollout.log", paid_env, 750)
    paid_env.clear()
    summary_path = args.root / "runs" / run_id / "summary.json"
    if not summary_path.is_file():
        print(json.dumps({"task": args.task_id, "rollout_exit": code,
                          "result": "uncertain_or_pre_run_error; do_not_auto_retry"}))
        return 2
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") in {"interrupted", "cancelled"}:
        print(json.dumps({"task": args.task_id, "status": summary["status"],
                          "result": "do_not_auto_retry"}))
        return 2
    eval_verdict = "skipped_no_patch"
    if summary.get("diff"):
        evaluated = _run(base + ["evaluate-run", str(formal), run_id,
                                 "--repo-root", str(args.root), "--timeout", "600"],
                         args.root / "evaluation.log", safe_env, 750)
        eval_verdict = "passed" if evaluated == 0 else "failed_or_blocked"
    qualified = (summary.get("status") == "completed"
                 and summary.get("visible_test_passed") is True
                 and bool(summary.get("diff")) and eval_verdict == "passed")
    if qualified:
        exported = _run(base + ["export-sft", str(args.root / "runs" / run_id / "events.jsonl"),
                                str(args.root / "passed-trajectory.jsonl")],
                        args.root / "export.log", safe_env, 60)
        qualified = exported == 0 and (args.root / "passed-trajectory.jsonl").is_file()
    result = {"task": args.task_id, "run_id": run_id, "status": summary.get("status"),
              "visible_test_passed": summary.get("visible_test_passed"),
              "nonempty_patch": bool(summary.get("diff")),
              "independent_evaluation": eval_verdict, "qualified_export": qualified}
    (args.root / "collector-result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))
    return 0 if qualified else 1


if __name__ == "__main__":
    raise SystemExit(main())
