"""Command-line entry points for manifest validation and local runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import signal
import time
from pathlib import Path

from codeagentbench.adapters.model import DeepSeekModel, LocalHFModel, ScriptedModel
from codeagentbench.models import Candidate, RunConfig
from codeagentbench.sandbox.workspace import Workspace, WorkspaceManager
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
    run.add_argument("--resume", action="store_true", help="resume an acknowledged checkpoint with its original budget")
    run.add_argument("--max-steps", type=int, default=8)
    run.add_argument("--max-tool-calls", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=16000)
    run.add_argument("--max-seconds", type=float, default=600.0)
    run.add_argument("--max-cost-usd", type=float, default=5.0)
    run.add_argument("--skip-evaluation", action="store_true", help="generate a candidate only; evaluate later in a credential-free process")
    evaluate_run = sub.add_parser("evaluate-run")
    evaluate_run.add_argument("manifest", type=Path)
    evaluate_run.add_argument("run_id")
    evaluate_run.add_argument("--repo-root", type=Path, default=Path("artifacts"))
    quality = sub.add_parser("quality-check")
    quality.add_argument("manifest", type=Path)
    quality.add_argument("task_id")
    quality.add_argument("--timeout", type=float, default=600.0)
    quality.add_argument("--output", type=Path, help="save the complete control report as JSON")
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
        if manifest.dataset == "SWE-Gym" and os.getenv("CODEAGENTBENCH_EXECUTOR") != "bwrap":
            print("SWE-Gym quality controls require CODEAGENTBENCH_EXECUTOR=bwrap", file=sys.stderr)
            return 2
        task = next((item for item in manifest.tasks if item.instance_id == args.task_id), None)
        if task is None:
            print(f"unknown task: {args.task_id}", file=sys.stderr)
            return 2
        report = run_controls(task, timeout=args.timeout, cache_root=Path("artifacts/.repo_cache"))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
    if args.command == "evaluate-run":
        if os.getenv("DEEPSEEK_API_KEY"):
            print("evaluation must run without DEEPSEEK_API_KEY in its process environment", file=sys.stderr)
            return 2
        if os.getenv("CODEAGENTBENCH_EXECUTOR") != "bwrap":
            print("evaluate-run requires CODEAGENTBENCH_EXECUTOR=bwrap for candidate test isolation", file=sys.stderr)
            return 2
        summary_path = args.repo_root / "runs" / args.run_id / "summary.json"
        if not summary_path.is_file():
            print(f"missing run summary: {summary_path}", file=sys.stderr)
            return 2
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest = load_manifest(args.manifest)
        task = next((item for item in manifest.tasks if item.instance_id == summary.get("task_id")), None)
        if task is None:
            print("run task is absent from the manifest", file=sys.stderr)
            return 2
        from codeagentbench.verification.evaluator import Evaluator

        candidate = Candidate("candidate-0", args.run_id, str(summary.get("diff") or ""), str(summary.get("status") or "unknown"))
        evaluation = Evaluator(args.repo_root / "evaluations").evaluate(task, candidate)
        ArtifactStore(args.repo_root).append_event(args.run_id, {"type": "evaluation", **evaluation.to_dict()})
        print(json.dumps({"run_id": args.run_id, "evaluation": evaluation.to_dict()}, ensure_ascii=False))
        return 0 if evaluation.passed else 1
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
        if not args.skip_evaluation:
            print("DeepSeek runs require --skip-evaluation; run evaluate-run later without the API key", file=sys.stderr)
            return 2
        if os.getenv("CODEAGENTBENCH_EXECUTOR") != "bwrap":
            print("DeepSeek runs require CODEAGENTBENCH_EXECUTOR=bwrap to isolate model-generated commands", file=sys.stderr)
            return 2
        if "DEEPSEEK_MIN_BALANCE_CNY" not in os.environ:
            print("DeepSeek runs require DEEPSEEK_MIN_BALANCE_CNY to stop before the spending floor", file=sys.stderr)
            return 2
        model = DeepSeekModel()
    store = ArtifactStore(args.repo_root)
    run_id = args.run_id or f"{task.instance_id}-{int(time.time())}"
    store.run_dir(run_id)  # Validate before constructing any workspace path.
    if args.resume:
        if not args.run_id:
            parser.error("--resume requires --run-id")
        workspace_path = args.repo_root / run_id / "workspace"
        if not workspace_path.is_dir():
            parser.error("resume workspace is missing")
        workspace = Workspace(run_id, workspace_path, task.base_commit)
    else:
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
    if args.resume:
        metadata = json.loads((store.run_dir(run_id) / "run.json").read_text(encoding="utf-8"))
        config = RunConfig(**metadata["config"])
    cancelled = False
    def request_cancel(signum, frame):
        nonlocal cancelled
        cancelled = True
    previous_handler = signal.signal(signal.SIGTERM, request_cancel)
    try:
        result = AgentRuntime(store).run(task, workspace, model, config, run_id=workspace.run_id, resume=args.resume, cancellation_requested=lambda: cancelled)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    if args.skip_evaluation:
        print(json.dumps({"run_id": result.run_id, "status": result.status, "diff_present": bool(result.diff), "evaluation": None}, ensure_ascii=False))
        return 0
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
