"""Bounded structured-edit diagnostic, separate from autonomous Agent metrics.

Models propose data (exact before/after text), not Python or shell programs.
The trusted adapter validates the proposal, then applies it inside NsJail.
No gold patch or sealed test content is supplied to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
from dataclasses import asdict
from pathlib import Path

from codeagentbench.adapters.model import LocalHFModel
from codeagentbench.models import Candidate, ToolIntent
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.verification.evaluator import Evaluator

TARGET = "moto/athena/models.py"
APPLY_EDIT = '''
import json, sys
from pathlib import Path
proposal = json.loads(sys.argv[1])
assert proposal["path"] == "moto/athena/models.py"
path = Path(proposal["path"])
assert not path.is_symlink()
text = path.read_text()
assert text.count(proposal["before"]) == 1, "before text must match exactly once"
updated = text.replace(proposal["before"], proposal["after"], 1)
assert updated != text
compile(updated, str(path), "exec")
path.write_text(updated)
print("STRUCTURED_EDIT_APPLIED")
'''


def validate_proposal(text: str) -> dict[str, str]:
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text.strip(), re.DOTALL)
    proposal = json.loads(fenced.group(1) if fenced else text)
    if not isinstance(proposal, dict) or set(proposal) != {"path", "before", "after"}:
        raise ValueError("exactly path/before/after fields required")
    if any(not isinstance(value, str) or not value or len(value) > 4000 for value in proposal.values()):
        raise ValueError("proposal fields must be bounded non-empty strings")
    if proposal["path"] != TARGET or proposal["before"] == proposal["after"]:
        raise ValueError("invalid path or no-op proposal")
    return proposal


def correction_feedback(reason: str) -> str:
    return (
        "Actual edit result: no successful edit was applied.\n"
        + reason[:2000]
        + "\nReturn one corrected JSON proposal. Copy before exactly from Current public source, "
          "not from the issue traceback or a dependency. The only writable target is "
          "moto/athena/models.py. Keep the original issue and source in view. "
          "Do not claim success or invent test results."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=1)
    parser.add_argument("--issue-view", choices=("full", "before-traceback", "repro-and-exception"), default="full")
    args = parser.parse_args()
    if selected_backend() != "nsjail" or os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("proposal diagnostic requires credential-free NsJail")
    admission = json.loads(args.admission.read_text())
    if not admission.get("admitted") or admission.get("task_id") != "getmoto__moto-6212":
        raise RuntimeError("target task admission required")
    task = next(t for t in load_manifest(args.manifest).tasks if t.instance_id == "getmoto__moto-6212")
    workspace = WorkspaceManager(args.artifact_root).create(task, args.run_id)
    output = args.artifact_root / "structured-probes" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    executor = BashExecutor(workspace.path, backend="nsjail")
    read = executor.execute(ToolIntent("public-source", f"sed -n '1,190p' {TARGET}", str(workspace.path), 30))
    if read.exit_code != 0:
        raise RuntimeError("source read failed")
    # Explicit diagnostic control: retain the original issue in the artifact,
    # but optionally omit the dependency stack after the public reproduction.
    # No solution, gold patch or hidden test is added to either view.
    issue = task.issue if args.issue_view == "full" else task.issue.split("Traceback:", 1)[0]
    if args.issue_view == "repro-and-exception":
        exceptions = re.findall(r"\b\w+(?:Error|Exception):[^\r\n]*", task.issue)
        issue += "\nReported exception from original issue:\n" + "\n".join(dict.fromkeys(exceptions))[:1000]
    messages = [
        {"role": "system", "content": "You are a code repair model. Return ONE complete JSON object "
         "with exactly path, before, after string fields. Propose one small exact text replacement "
         "in moto/athena/models.py. before must be copied exactly from current source, after is "
         "your corrected source. Use valid JSON escapes. No shell commands, explanations or imagined "
         "test results. Do not output the entire file. Example shape (unrelated example): "
         '{"path":"moto/athena/models.py","before":"value = old","after":"value = new"}.'},
        {"role": "user", "content": "Issue:\n" + issue + "\nCurrent public source:\n" + read.stdout},
    ]
    (output / "input.json").write_text(json.dumps({"messages": messages, "read": asdict(read),
        "issue_original": task.issue, "issue_view": args.issue_view}, indent=2))
    model = LocalHFModel(args.model, base_model_path=args.base, max_new_tokens=1024)
    # This diagnostic uses proposal/result messages, not the bash action
    # protocol. Its bounded native history must retain the actual failure;
    # recent-history-v3 intentionally recognizes only real bash receipts.
    model.prompt_policy = "native"
    report = {"kind": "structured-edit-diagnostic", "model": str(args.model),
              "autonomous_benchmark": False, "training_eligible": False,
              "prompt_policy": "native", "max_attempts": args.max_attempts, "issue_view": args.issue_view,
              "status": "rejected", "diff_present": False, "evaluation": None, "attempts": []}
    for attempt in range(args.max_attempts):
        response = model.complete(messages, temperature=0)
        (output / f"model-{attempt + 1:02d}.json").write_text(json.dumps(asdict(response), indent=2))
        record = {"attempt": attempt + 1}
        try:
            proposal = validate_proposal(response.text)
            command = "python -c " + shlex.quote(APPLY_EDIT) + " " + shlex.quote(json.dumps(proposal))
            receipt = executor.execute(ToolIntent(f"structured-edit-{attempt + 1}", command, str(workspace.path), 30))
            record["receipt"] = asdict(receipt)
            diff = workspace.diff()
            report["diff_present"] = bool(diff)
            if receipt.exit_code == 0 and diff:
                record["status"] = report["status"] = "proposal_executed"
                (output / "candidate.patch").write_text(diff)
                evaluation = Evaluator(args.artifact_root / "evaluations").evaluate(
                    task, Candidate("structured-0", args.run_id, diff, "proposal_executed"), timeout_seconds=120)
                report["evaluation"] = evaluation.to_dict()
            else:
                record["reason"] = receipt.stderr or "edit command did not create a patch"
        except (ValueError, TypeError) as exc:
            record["reason"] = str(exc)
        report["attempts"].append(record)
        # Persist before another model request: prior failures survive interruption.
        (output / "report.json").write_text(json.dumps(report, indent=2))
        if report["diff_present"]:
            break
        if attempt + 1 < args.max_attempts:
            messages.extend([
                {"role": "assistant", "content": response.text},
                {"role": "user", "content": correction_feedback(str(record.get("reason", "edit rejected")))},
            ])
            (output / f"correction-{attempt + 1:02d}.json").write_text(json.dumps(messages[-2:], indent=2))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
