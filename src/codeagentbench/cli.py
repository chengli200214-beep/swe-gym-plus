"""Command-line entry points for manifest validation and local runs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from codeagentbench.adapters.model import DeepSeekModel, LocalHFModel, ScriptedModel
from codeagentbench.models import Candidate, RunConfig
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import import_swe_gym, load_manifest, save_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="codeagentbench")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-manifest")
    validate.add_argument("path", type=Path)
    imported = sub.add_parser("import")
    imported.add_argument("source")
    imported.add_argument("output", type=Path)
    imported.add_argument("--revision", required=True)
    imported.add_argument("--limit", type=int)
    run = sub.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("task_id")
    run.add_argument("--repo-root", type=Path, default=Path("artifacts"))
    run.add_argument("--script", type=Path)
    run.add_argument("--model-backend", choices=("deepseek", "local"), default="deepseek")
    run.add_argument("--model-path", type=Path)
    run.add_argument("--base-model-path", type=Path)
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--run-id")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-tool-calls", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=16000)
    run.add_argument("--max-seconds", type=float, default=600.0)
    run.add_argument("--max-cost-usd", type=float, default=5.0)
    quality = sub.add_parser("quality-check")
    quality.add_argument("manifest", type=Path)
    quality.add_argument("task_id")
    quality.add_argument("--timeout", type=float, default=600.0)
    export_sft = sub.add_parser("export-sft")
    export_sft.add_argument("events", type=Path)
    export_sft.add_argument("output", type=Path)
    export_sft.add_argument("--include-unsuccessful", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "validate-manifest":
        manifest = load_manifest(args.path)
        print(json.dumps({"dataset": manifest.dataset, "revision": manifest.revision, "tasks": len(manifest.tasks)}))
        return 0
    if args.command == "import":
        save_manifest(import_swe_gym(args.source, revision=args.revision, limit=args.limit), args.output)
        print(args.output)
        return 0
    if args.command == "quality-check":
        from codeagentbench.tasks.quality import run_controls

        manifest = load_manifest(args.manifest)
        task = next((item for item in manifest.tasks if item.instance_id == args.task_id), None)
        if task is None:
            print(f"unknown task: {args.task_id}", file=sys.stderr)
            return 2
        report = run_controls(task, timeout=args.timeout, cache_root=Path("artifacts/.repo_cache"))
        print(json.dumps(report.to_dict(), ensure_ascii=False))
        return 0 if report.admitted else 1
    if args.command == "export-sft":
        from codeagentbench.training.export import export_run_to_sft

        exported = export_run_to_sft(
            args.events,
            args.output,
            successful_only=not args.include_unsuccessful,
        )
        print(json.dumps({"output": str(args.output), "exported": exported}))
        return 0 if exported else 1
    manifest = load_manifest(args.manifest)
    task = next((item for item in manifest.tasks if item.instance_id == args.task_id), None)
    if task is None:
        print(f"unknown task: {args.task_id}", file=sys.stderr)
        return 2
    if args.script:
        model = ScriptedModel(json.loads(args.script.read_text(encoding="utf-8")))
    elif args.model_backend == "local":
        if not args.model_path:
            print("--model-path is required with --model-backend local", file=sys.stderr)
            return 2
        model = LocalHFModel(
            args.model_path,
            base_model_path=args.base_model_path,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        model = DeepSeekModel()
    store = ArtifactStore(args.repo_root)
    run_id = args.run_id or f"{task.instance_id}-{int(time.time())}"
    workspace = WorkspaceManager(args.repo_root).create(task, run_id)
    from codeagentbench.runtime import AgentRuntime
    config = RunConfig(
        model=getattr(model, "model", "scripted"),
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        max_tokens=args.max_tokens,
        max_seconds=args.max_seconds,
        max_cost_usd=args.max_cost_usd,
    )
    result = AgentRuntime(store).run(task, workspace, model, config, run_id=workspace.run_id)
    candidate = Candidate("candidate-0", result.run_id, result.diff, result.status, visible_test_passed=result.visible_test_passed)
    from codeagentbench.verification.evaluator import Evaluator

    evaluation = Evaluator(args.repo_root / "evaluations").evaluate(task, candidate)
    store.append_event(result.run_id, {"type": "evaluation", **evaluation.to_dict()})
    print(json.dumps({"run_id": result.run_id, "status": result.status, "diff": result.diff, "evaluation": evaluation.to_dict()}, ensure_ascii=False))
    # The independent evaluator is the acceptance authority. A run may exhaust
    # its conversational budget immediately after producing a valid patch, so
    # preserve that budget status in the JSON result but do not turn a formally
    # passing candidate into a failed CLI command.
    return 0 if evaluation.passed else 1
