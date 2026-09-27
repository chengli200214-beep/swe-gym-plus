import json
from pathlib import Path
import signal
import sys
import time

import pytest

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.harness.cancellation import termination_requested
from codeagentbench.models import Candidate, RunConfig, TaskRecord
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.bounded_process import ProcessResult, run_bounded
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.verification.evaluator import Evaluator

ROOT = Path(__file__).parents[1]


def test_termination_scope_is_local_and_restores_previous_handler():
    previous = signal.getsignal(signal.SIGTERM)
    with termination_requested() as requested:
        assert not requested()
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        assert requested()
    assert signal.getsignal(signal.SIGTERM) is previous
    with termination_requested() as requested:
        assert not requested()


def test_runtime_forwards_cancellation_and_records_the_actual_receipt(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    task = load_manifest(ROOT / "data/manifests/demo.json").tasks[0]
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "cancelled")
    flag = False
    def requested():
        return flag
    class Sandbox:
        def __init__(self, path):
            assert path == workspace.path
        def execute(self, *args, cancellation_requested=None):
            nonlocal flag
            assert cancellation_requested is requested
            flag = True
            return ProcessResult(None, "partial output", "Harness cancellation requested", "cancelled")
    monkeypatch.setattr("codeagentbench.sandbox.nsjail.NsjailSandbox", Sandbox)
    store = ArtifactStore(tmp_path / "artifacts")
    result = AgentRuntime(store).run(task, workspace, ScriptedModel([{"command": "python --version"}]),
        RunConfig(max_steps=4, max_seconds=60), run_id="cancelled", cancellation_requested=requested)
    assert result.status == "cancelled"
    assert result.state.next_step == 1
    assert result.state.pending_action_id is None
    events = [json.loads(s) for s in (store.run_dir("cancelled") / "events.jsonl").read_text().splitlines()]
    tool = next(e for e in events if e["type"] == "tool")
    assert tool["receipt"]["status"] == "cancelled"
    assert tool["receipt"]["stdout"] == "partial output"
    assert len([e for e in events if e["type"] == "model"]) == 1


def test_evaluator_never_labels_cancelled_tests_as_model_failure(tmp_path, monkeypatch):
    task = load_manifest(ROOT / "data/manifests/demo.json").tasks[0]
    predicate = lambda: False
    class Sandbox:
        def __init__(self, path):
            pass
        def execute(self, *args, cancellation_requested=None):
            assert cancellation_requested is predicate
            return ProcessResult(None, "", "Harness cancellation requested", "cancelled")
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    monkeypatch.setattr("codeagentbench.sandbox.nsjail.NsjailSandbox", Sandbox)
    result = Evaluator(tmp_path / "eval").evaluate(task, Candidate("0", "x", "", "cancelled"),
        cancellation_requested=predicate)
    assert result.verdict.value == "blocked"
    assert result.reason == "formal evaluation cancelled"
    assert result.fail_to_pass is None and result.pass_to_pass is None


def test_evaluator_does_not_prepare_workspace_after_cancellation(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("cancelled evaluation must not create a workspace")
    monkeypatch.setattr(WorkspaceManager, "create", forbidden)
    task = TaskRecord("test", "unused", "local", "unused")
    result = Evaluator(tmp_path).evaluate(task, Candidate("0", "x", "", "cancelled"),
        cancellation_requested=lambda: True)
    assert result.reason == "formal evaluation cancelled"


def active(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group supervision")
@pytest.mark.parametrize("closed_pipes", [False, True])
def test_bounded_cancellation_kills_owned_group_even_with_closed_pipes(tmp_path, closed_pipes):
    marker = tmp_path / "processes.json"
    program = (
        "import json,os,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"open({str(marker)!r},'w').write(json.dumps([os.getpid(),p.pid])); "
        + ("os.close(1); os.close(2); " if closed_pipes else "") + "time.sleep(30)"
    )
    started = time.monotonic()
    result = run_bounded([sys.executable, "-c", program], timeout=20, output_limit=2000,
        writable_paths=(tmp_path,), cancellation_requested=lambda: marker.exists() and time.monotonic() - started > .4)
    assert result.status == "cancelled" and result.exit_code is None
    assert time.monotonic() - started < 3
    assert not any(active(pid) for pid in json.loads(marker.read_text()))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group supervision")
def test_bounded_cancellation_before_launch_has_no_effect(tmp_path):
    marker = tmp_path / "must-not-exist"
    result = run_bounded([sys.executable, "-c", f"open({str(marker)!r},'w').close()"],
        timeout=20, output_limit=2000, writable_paths=(tmp_path,), cancellation_requested=lambda: True)
    assert result.status == "cancelled"
    assert not marker.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group supervision")
def test_bounded_callback_failure_still_cleans_up(tmp_path):
    marker = tmp_path / "pid"
    program = f"import os,time; open({str(marker)!r},'w').write(str(os.getpid())); time.sleep(30)"
    def broken():
        if marker.exists():
            raise RuntimeError("cancellation observer failed")
        return False
    with pytest.raises(RuntimeError, match="observer failed"):
        run_bounded([sys.executable, "-c", program], timeout=20, output_limit=2000,
            writable_paths=(tmp_path,), cancellation_requested=broken)
    assert not active(int(marker.read_text()))
