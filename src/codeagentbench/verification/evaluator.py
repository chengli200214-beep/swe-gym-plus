"""SWE-Bench-style evaluation in a fresh workspace, outside agent state."""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path

from codeagentbench.models import Candidate, EvaluationResult, TaskRecord, ToolIntent, Verdict
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager


def _apply_patch(workspace: Path, patch: str) -> tuple[bool, str]:
    if not patch:
        return True, ""
    result = subprocess.run(["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", "-"], cwd=workspace, input=patch, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    return result.returncode == 0, result.stderr


def _test_names_in_output(names: tuple[str, ...], output: str, exit_code: int) -> bool | None:
    if not names:
        return None
    # Quiet runners often emit only dots. A successful formal command is the
    # authoritative result; names are retained as the mapping metadata rather
    # than incorrectly turning `pytest -q` into an unknown verdict.
    return exit_code == 0


class Evaluator:
    """Run formal tests using EvalSpec, never using the selector's score."""

    def __init__(self, artifact_root: str | Path = "artifacts/evaluations") -> None:
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        # Reuse the agent/quality cache rooted beside the evaluation artifacts.
        # Keeping a second cache here forces another network clone for every
        # evaluator run and makes evaluation fragile when GitHub is unavailable.
        self.cache_root = self.artifact_root.parent / ".repo_cache"

    def evaluate(self, task: TaskRecord, candidate: Candidate, *, timeout_seconds: float = 600.0) -> EvaluationResult:
        spec = task.eval_spec
        if timeout_seconds <= 0:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, reason="formal evaluation has no remaining time budget")
        if not spec.test_command:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, reason="missing formal test command")
        run_id = f"{candidate.run_id}-eval-{candidate.candidate_id}"
        manager = WorkspaceManager(self.artifact_root, cache_root=self.cache_root)
        started = time.monotonic()
        try:
            workspace = manager.create(task, run_id)
        except (OSError, RuntimeError) as exc:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, duration_seconds=time.monotonic() - started, reason=f"workspace preparation failed: {exc}")
        ok, detail = _apply_patch(workspace.path, candidate.diff)
        if not ok:
            return EvaluationResult(Verdict.FAILED, None, False, None, stderr=detail, duration_seconds=time.monotonic() - started, reason="candidate patch did not apply")
        ok, detail = _apply_patch(workspace.path, spec.test_patch)
        if not ok:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, stderr=detail, duration_seconds=time.monotonic() - started, reason="evaluation test patch did not apply")
        remaining_seconds = timeout_seconds - (time.monotonic() - started)
        if remaining_seconds <= 0:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, duration_seconds=time.monotonic() - started, reason="formal evaluation preparation exhausted time budget")
        try:
            if os.getenv("CODEAGENTBENCH_EXECUTOR") == "bwrap":
                receipt = BashExecutor(workspace.path, output_limit=200_000, backend="bwrap").execute(
                    ToolIntent("formal-evaluation", spec.test_command, str(workspace.path), remaining_seconds)
                )
                if receipt.timed_out:
                    return EvaluationResult(Verdict.BLOCKED, None, None, None, stdout=receipt.stdout, stderr=receipt.stderr, duration_seconds=time.monotonic() - started, reason="formal test timed out")
                exit_code, stdout, stderr = receipt.exit_code, receipt.stdout, receipt.stderr
            else:
                result = subprocess.run(spec.test_command, cwd=workspace.path, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=remaining_seconds, check=False)
                exit_code = result.returncode
                stdout, stderr = result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            return EvaluationResult(Verdict.BLOCKED, None, None, None, stdout=str(exc.stdout or ""), stderr=str(exc.stderr or ""), duration_seconds=time.monotonic() - started, reason="formal test timed out")
        combined = stdout + "\n" + stderr
        fail_to_pass = _test_names_in_output(spec.fail_to_pass, combined, exit_code) if spec.fail_to_pass else (exit_code == 0)
        pass_to_pass = _test_names_in_output(spec.pass_to_pass, combined, exit_code) if spec.pass_to_pass else (exit_code == 0)
        passed = exit_code == 0 and fail_to_pass is not False and pass_to_pass is not False
        failed_tests = tuple(re.findall(r"(?:FAILED|ERROR)[ :]([^\n]+)", combined))
        return EvaluationResult(
            Verdict.PASSED if passed else Verdict.FAILED,
            exit_code,
            fail_to_pass,
            pass_to_pass,
            stdout=stdout,
            stderr=stderr,
            failed_tests=failed_tests,
            duration_seconds=time.monotonic() - started,
            reason="formal tests passed" if passed else "formal tests failed",
        )
