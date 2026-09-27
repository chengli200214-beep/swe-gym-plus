"""One-task edit diagnostic with public-source grounding, not a benchmark.

The controller supplies a real sandbox read receipt, never the gold patch,
sealed test names, a written solution or invented tool history. This reduces
search difficulty deliberately; report it separately from autonomous runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

from codeagentbench.adapters.model import LocalHFModel
from codeagentbench.models import Candidate, RunConfig, ToolIntent
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.verification.evaluator import Evaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    if selected_backend() != "nsjail" or os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("probe requires credential-free NsJail execution")
    admission = json.loads(args.admission.read_text())
    if not admission.get("admitted") or admission.get("task_id") != "getmoto__moto-6212":
        raise RuntimeError("target task has not passed current admission")
    task = next(t for t in load_manifest(args.manifest).tasks if t.instance_id == "getmoto__moto-6212")
    workspace = WorkspaceManager(args.artifact_root).create(task, args.run_id)
    receipt = BashExecutor(workspace.path, backend="nsjail").execute(ToolIntent(
        "controller-source-read", "sed -n '1,190p' moto/athena/models.py", str(workspace.path), 30))
    if receipt.exit_code != 0 or not receipt.stdout:
        raise RuntimeError("public source read failed")
    store = ArtifactStore(args.artifact_root)
    evidence = {"kind": "public-source-grounded-edit-diagnostic", "receipt": asdict(receipt),
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "autonomous_benchmark": False, "sealed_eval_reused_for_debug": True}
    context = (str(task.metadata.get("context") or task.metadata.get("hints_text") or "")
               + "\nController supplied public-source observation (not a solution):\n"
               + receipt.stdout
               + "\nThe file exists at moto/athena/models.py, not moto/athena/models/athena.py. "
                 "Use the actual text above, not guessed symbols. Produce a small complete edit "
                 "and a real smoke/test check. Do not repeat already observed reads. "
                 "Python's pathlib is available for edits. Keep shell commands short.")
    grounded = replace(task, metadata={**task.metadata, "context": context})
    model = LocalHFModel(args.model, base_model_path=args.base, max_new_tokens=2048)
    config = RunConfig(model=str(args.model), temperature=0, max_steps=6, max_tool_calls=6,
                       max_tokens=100000, max_seconds=300, max_cost_usd=0, seed=0)
    result = AgentRuntime(store).run(grounded, workspace, model, config, run_id=args.run_id)
    (store.run_dir(args.run_id) / "controller-observation.json").write_text(json.dumps(evidence, indent=2))
    # The evaluator still receives the original task and creates a fresh checkout.
    evaluation = Evaluator(args.artifact_root / "evaluations").evaluate(
        task, Candidate("candidate-0", args.run_id, result.diff, result.status), timeout_seconds=120)
    store.append_event(args.run_id, {"type": "evaluation", **evaluation.to_dict()})
    report = {"run_id": args.run_id, "kind": evidence["kind"], "status": result.status,
              "failure_reason": result.failure_reason, "diff_present": bool(result.diff),
              "steps": result.steps, "evaluation": evaluation.to_dict()}
    (store.run_dir(args.run_id) / "diagnostic.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
