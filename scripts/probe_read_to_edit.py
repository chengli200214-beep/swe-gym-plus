"""Paired, assisted read-to-edit diagnostic on an already exposed task.

Two read actions are supplied by this diagnostic, not by the model. They run
through the normal NsJail/runtime tool path and produce real source receipts.
All later actions use the unchanged deployment protocol. This is not an
autonomous benchmark run or training data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from codeagentbench.adapters.action import parse_action
from codeagentbench.adapters.model import LocalHFModel, ModelResponse
from codeagentbench.models import Candidate, RunConfig
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.backends import selected_backend
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.verification.evaluator import Evaluator


TASK_ID = "getmoto__moto-4895"
SCRIPTED_READS = (
    {"path": "moto/settings.py", "start_line": 45, "end_line": 60},
    {"path": "moto/core/utils.py", "start_line": 405, "end_line": 430},
)


class ScriptedPreReadModel:
    """Supply two declared helper reads, then delegate verbatim to one model."""

    def __init__(self, model: LocalHFModel) -> None:
        self.model = model
        self.prompt_policy = model.prompt_policy
        self.scripted = [
            json.dumps({"read": read, "done": False})
            for read in SCRIPTED_READS
        ]
        self.first_real_prompt_sha256: str | None = None
        self.first_real_token_ids_sha256: str | None = None
        self.first_real_input_tokens: int | None = None

    def request_token_bound(self, messages: list[dict[str, str]]) -> int:
        return self.model.request_token_bound(messages)

    def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0) -> ModelResponse:
        if self.scripted:
            return ModelResponse(self.scripted.pop(0))
        if self.first_real_prompt_sha256 is None:
            serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            encoded = self.model._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)
            token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if token_ids and isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            self.first_real_prompt_sha256 = hashlib.sha256(serialized.encode()).hexdigest()
            self.first_real_token_ids_sha256 = hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()
            self.first_real_input_tokens = len(token_ids)
        return self.model.complete(messages, temperature=temperature)


def action_kind(text: str) -> str:
    try:
        action = parse_action(text)
    except ValueError:
        return "invalid"
    if action.edit is not None:
        return "edit"
    if action.read is not None:
        return "read"
    if action.search is not None:
        return "search"
    return "done" if action.done else "shell"


def summarize_events(path: Path) -> dict:
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    model_events = [event for event in events if event.get("type") == "model"]
    tool_events = [event for event in events if event.get("type") == "tool"]
    scripted_tools = tool_events[:len(SCRIPTED_READS)]
    if len(model_events) < len(SCRIPTED_READS) or len(scripted_tools) != len(SCRIPTED_READS):
        raise RuntimeError("scripted source observations did not complete")
    for expected, event in zip(SCRIPTED_READS, scripted_tools):
        intent, receipt = event["intent"], event["receipt"]
        action = parse_action(intent["action_json"])
        if (action.read is None or action.read.to_dict() != expected
                or receipt["exit_code"] != 0 or receipt["timed_out"]):
            raise RuntimeError("a scripted source read failed or changed identity")
    actual = model_events[len(SCRIPTED_READS):]
    model_tool_events = tool_events[len(SCRIPTED_READS):]
    return {
        "scripted_read_receipts": len(scripted_tools),
        "model_action_kinds": [action_kind(event["content"]) for event in actual],
        "valid_edit_proposed": any(action_kind(event["content"]) == "edit" for event in actual),
        "edit_executed": any(
            action_kind(event["intent"]["action_json"]) == "edit"
            and event["receipt"]["exit_code"] == 0
            and not event["receipt"]["timed_out"]
            and event["intent"]["pre_digest"] != event["receipt"]["post_digest"]
            for event in model_tool_events
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if selected_backend() != "nsjail" or os.getenv("DEEPSEEK_API_KEY"):
        raise RuntimeError("this diagnostic requires credential-free NsJail")
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    if admission.get("task_id") != TASK_ID or admission.get("admitted") is not True:
        raise RuntimeError("the previously exposed task requires its passing admission controls")
    task = next(task for task in load_manifest(args.manifest).tasks if task.instance_id == TASK_ID)
    if task.base_commit != "9ab0ac4fb30de9eddda0de4401265943a2d13828":
        raise RuntimeError("task base commit differs from the admitted diagnostic")
    args.artifact_root.mkdir(parents=True, exist_ok=False)
    store = ArtifactStore(args.artifact_root / "artifacts")
    workspace = WorkspaceManager(store.root, cache_root=args.cache).create(task, args.run_id)
    model = ScriptedPreReadModel(LocalHFModel(
        args.model, base_model_path=args.base, max_new_tokens=2048))
    config = RunConfig(model=str(args.model), temperature=0, max_steps=6,
        max_tool_calls=6, max_tokens=100_000, max_seconds=600, max_cost_usd=0,
        repository_inventory=False, require_visible_test_before_done=True)
    result = AgentRuntime(store).run(task, workspace, model, config, run_id=args.run_id)
    store.append_event(args.run_id, {
        "type": "diagnostic_provenance", "autonomous_benchmark": False,
        "scripted_model_steps": list(range(len(SCRIPTED_READS))),
        "reason": "helper-selected source reads; all receipts are real NsJail results",
    })
    metrics = summarize_events(store.run_dir(args.run_id) / "events.jsonl")
    report = {
        "kind": "assisted-read-to-edit-diagnostic", "task_id": TASK_ID,
        "exposed_development_task": True, "autonomous_benchmark": False,
        "training_eligible": False, "scripted_actions_not_model_generated": list(SCRIPTED_READS),
        "model": str(args.model), "base": str(args.base) if args.base else None,
        "prompt_policy": model.prompt_policy, "config": config.to_dict(),
        "first_real_prompt_sha256": model.first_real_prompt_sha256,
        "first_real_token_ids_sha256": model.first_real_token_ids_sha256,
        "first_real_input_tokens": model.first_real_input_tokens,
        "status": result.status, "failure_reason": result.failure_reason,
        "steps": result.steps, "diff_present": bool(result.diff),
        "evaluation": None, **metrics,
    }
    (args.artifact_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if result.diff:
        evaluation = Evaluator(args.artifact_root / "evaluations", cache_root=args.cache).evaluate(
            task, Candidate("assisted-candidate-0", args.run_id, result.diff, result.status), timeout_seconds=180)
        report["evaluation"] = evaluation.to_dict()
        (args.artifact_root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
