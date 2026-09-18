"""Model-free task quality controls for SWE-Bench-style tasks."""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from codeagentbench.models import EvalSpec, TaskRecord
from codeagentbench.sandbox.workspace import WorkspaceManager


@dataclass(frozen=True)
class ControlResult:
    """One control run and the expectation it was compared against."""

    name: str
    exit_code: int | None
    expected_pass: bool
    observed_pass: bool
    stdout: str
    stderr: str
    duration_seconds: float
    reason: str = ""


@dataclass(frozen=True)
class QualityReport:
    """Task admission decision based on both negative and positive controls."""

    task_id: str
    admitted: bool
    controls: tuple[ControlResult, ...]
    reason: str = ""

    def to_dict(self) -> dict:
        return {"task_id": self.task_id, "admitted": self.admitted, "reason": self.reason, "controls": [asdict(item) for item in self.controls]}


def _apply_patch(workspace: Path, patch: str) -> tuple[bool, str]:
    if not patch:
        return True, ""
    result = subprocess.run(
        ["git", "apply", "--ignore-space-change", "--ignore-whitespace", "--whitespace=nowarn", "-"],
        input=patch,
        text=True,
        cwd=workspace,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0, result.stderr


def _run_tests(workspace: Path, command: str, timeout: float = 600.0) -> tuple[int | None, str, str, float]:
    started = time.monotonic()
    try:
        result = subprocess.run(command, cwd=workspace, shell=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False)
        return result.returncode, result.stdout, result.stderr, time.monotonic() - started
    except subprocess.TimeoutExpired as exc:
        return None, str(exc.stdout or ""), str(exc.stderr or ""), time.monotonic() - started


def run_controls(task: TaskRecord, *, timeout: float = 600.0, cache_root: str | Path | None = None) -> QualityReport:
    """Run base+tests (expected fail) and base+gold+tests (expected pass)."""

    spec: EvalSpec = task.eval_spec
    if not spec.test_command:
        return QualityReport(task.instance_id, False, (), "missing test_command")
    controls: list[ControlResult] = []
    with tempfile.TemporaryDirectory(prefix="cab-quality-") as temp:
        root = Path(temp)
        manager = WorkspaceManager(root, cache_root=cache_root)
        for name, gold_patch, expected_pass in (
            ("unfixed", "", False),
            ("gold", spec.gold_patch, True),
        ):
            try:
                workspace = manager.create(task, name)
            except (OSError, RuntimeError) as exc:
                controls.append(ControlResult(name, None, expected_pass, False, "", str(exc), 0.0, "prepare failed"))
                continue
            test_ok, patch_error = _apply_patch(workspace.path, spec.test_patch)
            if test_ok and gold_patch:
                test_ok, patch_error = _apply_patch(workspace.path, gold_patch)
            if not test_ok:
                controls.append(ControlResult(name, None, expected_pass, False, "", patch_error, 0.0, "patch failed"))
                continue
            exit_code, stdout, stderr, duration = _run_tests(workspace.path, spec.test_command, timeout)
            observed = exit_code == 0
            controls.append(ControlResult(name, exit_code, expected_pass, observed, stdout, stderr, duration))
    admitted = len(controls) == 2 and all(item.observed_pass == item.expected_pass for item in controls)
    reason = "" if admitted else "control expectation mismatch"
    return QualityReport(task.instance_id, admitted, tuple(controls), reason)
