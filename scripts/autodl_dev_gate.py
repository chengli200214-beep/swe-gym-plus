"""A frozen three-task, credential-free autonomous repair gate.

Admission never supplies gold/test patches to the model. Admission failures are
retained rather than replaced with tasks selected by model success.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
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


def run(root: Path, model_path: Path, *, cache: Path | None = None):
    tasks = load_frozen(root)
    output = root / "run-report.json"
    if output.exists():
        raise ValueError("preserve the existing experiment report")
    reports = [json.loads((root / "quality" / (t.instance_id + ".json")).read_text()) for t in tasks]
    if any(r["task_id"] != t.instance_id for t, r in zip(tasks, reports)):
        raise ValueError("admission identity mismatch")
    if not any(r["admitted"] for r in reports):
        raise ValueError("no admitted tasks; fix environment, do not run the model")
    os.environ["CODEAGENTBENCH_LOCAL_PROMPT_POLICY"] = "recent-history-v3"
    model = LocalHFModel(model_path, max_new_tokens=2048)
    config = RunConfig(model=str(model_path), temperature=0, max_steps=16,
        max_tool_calls=16, max_tokens=180000, max_seconds=600, max_cost_usd=0)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    source_files = [Path("src/codeagentbench") / p for p in (
        "runtime.py", "models.py", "adapters/action.py", "adapters/text_edit.py", "adapters/file_tools.py",
        "adapters/source_read.py", "harness/source_evidence.py", "harness/context_history.py",
        "sandbox/nsjail.py", "sandbox/executor.py", "sandbox/bounded_process.py",
        "verification/evaluator.py")]
    source_files.append(Path("scripts/autodl_dev_gate.py"))
    report = {"autonomous": True, "manifest_sha256": digest(root / "manifest.json"),
              "git_commit": commit, "source_sha256": {str(p): digest(p) for p in source_files},
              "environment": {"python": sys.version, "backend": selected_backend(),
                  "rootfs": os.environ.get("CODEAGENTBENCH_ROOTFS", "default"),
                  "nsjail_sha256": digest(os.environ.get("CODEAGENTBENCH_NSJAIL", "/root/autodl-tmp/bin/nsjail")),
                  "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft")}},
              "model": str(model_path), "config": config.to_dict(), "tasks": [],
              "trained": False, "gate_passed": False}
    store = ArtifactStore(root / "artifacts")
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
            result = AgentRuntime(store).run(task, workspace, model, config, run_id=run_id)
            evaluation = Evaluator(root / "evaluations", cache_root=cache).evaluate(task,
                Candidate("candidate-0", run_id, result.diff, result.status), timeout_seconds=180)
            store.append_event(run_id, {"type": "evaluation", **evaluation.to_dict()})
            record.update({"run_id": run_id, "status": result.status, "failure_reason": result.failure_reason,
                           "steps": result.steps, "diff_present": bool(result.diff),
                           "independent_evaluation": evaluation.to_dict(),
                           "autonomous_success": bool(result.diff) and evaluation.passed})
        report["tasks"].append(record)
        report["gate_passed"] = any(r.get("autonomous_success") for r in report["tasks"])
        output.write_text(json.dumps(report, indent=2))
        print(json.dumps({k: record[k] for k in record if k != "independent_evaluation"}), flush=True)
    return {"gate_passed": report["gate_passed"], "frozen": len(tasks),
            "admitted": sum(r["admitted"] for r in reports),
            "passed": sum(bool(r.get("autonomous_success")) for r in report["tasks"])}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=("freeze", "admit", "run"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--task-ids", nargs=3)
    p.add_argument("--cache", type=Path)
    p.add_argument("--model", type=Path)
    args = p.parse_args()
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
        result = run(args.root, args.model, cache=args.cache)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
