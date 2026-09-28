"""A frozen three-task, credential-free autonomous repair gate.

Admission never supplies gold/test patches to the model. Admission failures are
retained rather than replaced with tasks selected by model success.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from codeagentbench.adapters.model import LocalHFModel
from codeagentbench.models import Candidate, RunConfig
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.tasks.quality import run_controls
from codeagentbench.verification.evaluator import Evaluator


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome_flags(result, evaluation):
    """Do not conflate a passing candidate patch with a clean agent finish."""
    patch_verified = bool(result.diff) and evaluation.passed
    run_completed = result.status == "completed"
    return {"patch_verified": patch_verified, "run_completed": run_completed,
            "autonomous_success": patch_verified and run_completed,
            "final_patch_sha256": hashlib.sha256(result.diff.encode("utf-8")).hexdigest()
            if result.diff else None}


def visible_test_command(workspace: Path, directory: str) -> str:
    """Only name an already-visible test directory; never inspect EvalSpec."""
    if not isinstance(directory, str) or not re.fullmatch(r"tests(?:/[A-Za-z0-9_-]+)+", directory):
        raise ValueError("visible test directory must be a simple relative path under tests/")
    root = workspace.resolve()
    target = root
    for part in directory.split("/"):
        target = target / part
        if target.is_symlink():
            raise ValueError("visible test directory contains a symlink")
    if not target.is_dir() or not target.resolve().is_relative_to(root):
        raise ValueError("visible test directory is not in the unmodified task checkout")
    return f"python -m pytest -q -x {directory}"


def freeze(source: Path, root: Path, task_ids: list[str]):
    if len(task_ids) != 3 or len(set(task_ids)) != 3:
        raise ValueError("exactly three distinct development tasks required")
    manifest = load_manifest(source)
    tasks = [next(t for t in manifest.tasks if t.instance_id == task_id) for task_id in task_ids]
    if any(t.split != "dev" for t in tasks):
        raise ValueError("do not silently move training or sealed eval tasks into development")
    root.mkdir(parents=True, exist_ok=False)
    path = root / "manifest.json"
    path.write_text(json.dumps({"dataset": manifest.dataset, "revision": manifest.revision,
                               "tasks": [t.to_dict() for t in tasks]}, indent=2))
    identity = {"kind": "autonomous-three-task-dev-gate", "source_sha256": digest(source),
                "manifest_sha256": digest(path), "tasks": task_ids,
                "rule": "at least one non-empty autonomous patch independently passed",
                "no_outcome_replacement": True}
    (root / "freeze.json").write_text(json.dumps(identity, indent=2))
    return identity


def load_frozen(root):
    identity = json.loads((root / "freeze.json").read_text())
    path = root / "manifest.json"
    if digest(path) != identity["manifest_sha256"]:
        raise ValueError("frozen manifest changed")
    tasks = load_manifest(path).tasks
    if [t.instance_id for t in tasks] != identity["tasks"]:
        raise ValueError("frozen task identities changed")
    return tasks


def admit(root: Path, cache: Path):
    tasks = load_frozen(root)
    output = root / "quality"
    output.mkdir(exist_ok=False)

    def check(task):
        report = run_controls(task, timeout=180, cache_root=cache).to_dict()
        (output / (task.instance_id + ".json")).write_text(json.dumps(report, indent=2))
        print(json.dumps({"task_id": task.instance_id, "admitted": report["admitted"],
                          "reason": report["reason"]}), flush=True)
        return report

    with ThreadPoolExecutor(max_workers=3) as pool:
        reports = list(pool.map(check, tasks))
    summary = {"admitted": sum(r["admitted"] for r in reports), "frozen": len(tasks),
               "manifest_sha256": digest(root / "manifest.json")}
    (root / "admission-summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run(root: Path, model_path: Path, *, cache: Path | None = None, output_root: Path | None = None,
        visible_test_directories: dict[str, str] | None = None):
    tasks = load_frozen(root)
    visible_test_directories = visible_test_directories or {}
    if visible_test_directories and (output_root is None or set(visible_test_directories) != {t.instance_id for t in tasks}):
        raise ValueError("visible-test retest requires a new output root and one directory for every frozen task")
    destination = root if output_root is None else output_root
    output = destination / "run-report.json"
    if output.exists():
        raise ValueError("preserve the existing experiment report")
    reports = [json.loads((root / "quality" / (t.instance_id + ".json")).read_text()) for t in tasks]
    if any(r["task_id"] != t.instance_id for t, r in zip(tasks, reports)):
        raise ValueError("admission identity mismatch")
    if not any(r["admitted"] for r in reports):
        raise ValueError("no admitted tasks; fix environment, do not run the model")
    if output_root is not None:
        destination.mkdir(parents=True, exist_ok=False)
    os.environ["CODEAGENTBENCH_LOCAL_PROMPT_POLICY"] = "recent-history-v3"
    model = LocalHFModel(model_path, max_new_tokens=2048)
    config = RunConfig(model=str(model_path), temperature=0, max_steps=16,
        max_tool_calls=16, max_tokens=180000, max_seconds=600, max_cost_usd=0,
        repository_inventory=True, require_visible_test_before_done=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    source_files = [Path("src/codeagentbench") / p for p in (
        "runtime.py", "models.py", "adapters/action.py", "adapters/text_edit.py", "adapters/file_tools.py",
        "adapters/source_read.py", "harness/source_evidence.py", "harness/context_history.py",
        "harness/tool_observation.py",
        "adapters/source_search.py",
        "adapters/repository_inventory.py", "harness/repository_inventory.py",
        "sandbox/nsjail.py", "sandbox/executor.py", "sandbox/bounded_process.py",
        "verification/evaluator.py")]
    source_files.append(Path("scripts/autodl_dev_gate.py"))
    report = {"autonomous": True, "manifest_sha256": digest(root / "manifest.json"),
              "development_retest": output_root is not None,
              "visible_test_directories": visible_test_directories,
              "visible_test_rule": "existing base-checkout directories only; no evaluator selectors or test patches",
              "git_commit": commit, "source_sha256": {str(p): digest(p) for p in source_files},
              "environment": {"python": sys.version, "backend": selected_backend(),
                  "rootfs": os.environ.get("CODEAGENTBENCH_ROOTFS", "default"),
                  "nsjail_sha256": digest(os.environ.get("CODEAGENTBENCH_NSJAIL", "/root/autodl-tmp/bin/nsjail")),
                  "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}},
              "model": str(model_path), "config": config.to_dict(), "tasks": [],
              "trained": False, "gate_passed": False}
    store = ArtifactStore(destination / "artifacts")
    cache = cache or root / ".repo_cache"
    manager = WorkspaceManager(store.root, cache_root=cache)
    for task, quality in zip(tasks, reports):
        record = {"task_id": task.instance_id, "admitted": quality["admitted"],
                  "quality_sha256": digest(root / "quality" / (task.instance_id + ".json"))}
        if not quality["admitted"]:
            record["status"] = "blocked_admission"
        else:
            run_id = "base7b-edit-" + task.instance_id
            workspace = manager.create(task, run_id)
            agent_task = task
            if visible_test_directories:
                command = visible_test_command(workspace.path, visible_test_directories[task.instance_id])
                agent_task = replace(task, metadata={**task.metadata, "agent_test_command": command})
                record["visible_test_command"] = command
            result = AgentRuntime(store).run(agent_task, workspace, model, config, run_id=run_id)
            evaluation = Evaluator(destination / "evaluations", cache_root=cache).evaluate(task,
                Candidate("candidate-0", run_id, result.diff, result.status), timeout_seconds=180)
            store.append_event(run_id, {"type": "evaluation", **evaluation.to_dict()})
            record.update({"run_id": run_id, "status": result.status, "failure_reason": result.failure_reason,
                           "steps": result.steps, "diff_present": bool(result.diff),
                           "independent_evaluation": evaluation.to_dict(),
                           **outcome_flags(result, evaluation)})
        report["tasks"].append(record)
        # The frozen gate asks for a passing autonomous patch, even if the run
        # then fails during its final protocol step. Report both facts.
        report["gate_passed"] = any(r.get("patch_verified") for r in report["tasks"])
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: record[k] for k in record if k != "independent_evaluation"}), flush=True)
    return {"gate_passed": report["gate_passed"], "frozen": len(tasks),
            "admitted": sum(r["admitted"] for r in reports),
            "passed": sum(bool(r.get("patch_verified")) for r in report["tasks"]),
            "normally_completed_passed": sum(bool(r.get("autonomous_success")) for r in report["tasks"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("freeze", "admit", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--task-ids", nargs=3)
    p.add_argument("--cache", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output-root", type=Path, help="new directory for an explicitly labeled development retest")
    p.add_argument("--visible-test-directories",
                   help="JSON mapping of each frozen task ID to an existing checkout tests/ directory; development retest only")
    args = p.parse_args()
    if args.output_root is not None and args.phase != "run":
        p.error("output-root is only valid for run; it never changes the frozen task set")
    if args.visible_test_directories is not None and args.phase != "run":
        p.error("visible-test-directories is only valid for run")
    if selected_backend() != "nsjail" or os.getenv("DEEPSEEK_API_KEY"):
        raise ValueError("this experiment requires credential-free same-machine NsJail")
    if args.phase == "freeze":
        if args.source is None or args.task_ids is None:
            p.error("freeze requires source and three task IDs")
        result = freeze(args.source, args.root, args.task_ids)
    elif args.phase == "admit":
        if args.cache is None:
            p.error("admit requires cache")
        result = admit(args.root, args.cache)
    else:
        if args.model is None:
            p.error("run requires model")
        visible_dirs = (json.loads(args.visible_test_directories)
                        if args.visible_test_directories is not None else None)
        if visible_dirs is not None and not isinstance(visible_dirs, dict):
            raise ValueError("visible-test-directories must be a JSON object")
        result = run(args.root, args.model, cache=args.cache, output_root=args.output_root,
                     visible_test_directories=visible_dirs)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
