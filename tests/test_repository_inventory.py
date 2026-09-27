from dataclasses import asdict
import json
import os
from pathlib import Path

import pytest

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.adapters.repository_inventory import inventory_command, inventory_packet
from codeagentbench.models import RunConfig, TaskRecord, ToolIntent
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.bounded_process import ProcessResult
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.training.export import export_run_to_sft
from scripts.prepare_aligned_sft import examples

ROOT = Path(__file__).parents[1]


def fixture_run(tmp_path):
    source = tmp_path / "source"
    (source / "package" / "actual_module").mkdir(parents=True)
    (source / "package" / "actual_module" / "__init__.py").write_text("")
    (source / "private.txt").write_text("CONTENTS_MUST_NOT_ENTER_INVENTORY")
    task = TaskRecord("inventory", str(source), "local", "inspect actual paths")
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "inventory")
    store = ArtifactStore(tmp_path / "artifacts")
    return task, workspace, store, AgentRuntime(store)


def events(store, run_id="inventory"):
    return [json.loads(s) for s in (store.run_dir(run_id) / "events.jsonl").read_text().splitlines()]


def test_actual_inventory_is_a_budgeted_observation_not_a_model_action(tmp_path):
    task, workspace, store, runtime = fixture_run(tmp_path)
    result = runtime.run(task, workspace, ScriptedModel([{"done": True}]),
        RunConfig(repository_inventory=True, max_seconds=60), run_id="inventory")
    recorded = events(store)
    observation = next(e for e in recorded if e["type"] == "harness_tool")
    packet = inventory_packet(observation["receipt"])
    assert packet["child_directories"]["package"] == ["actual_module"]
    assert "private.txt" in packet["root_entries"]
    assert ".git" not in packet["root_entries"]
    assert "CONTENTS_MUST_NOT_ENTER_INVENTORY" not in json.dumps(packet)
    model = next(e for e in recorded if e["type"] == "model")
    assert json.loads(model["context_messages"][1]["content"])["repository_inventory"] == packet
    assert result.state.source_observations == []
    assert result.state.inventory_consumed
    assert json.loads((store.run_dir("inventory") / "summary.json").read_text())["budget"]["tool_calls"] == 1
    assert len([e for e in recorded if e["type"] == "model"]) == 1
    assert len([e for e in recorded if e["type"] == "tool"]) == 0


def test_inventory_cannot_authorize_a_source_edit(tmp_path):
    task, workspace, store, runtime = fixture_run(tmp_path)
    result = runtime.run(task, workspace, ScriptedModel([
        {"edit": {"path": "private.txt", "before": "CONTENTS_MUST_NOT_ENTER_INVENTORY", "after": "wrong"}},
        {"done": True}]), RunConfig(repository_inventory=True, max_seconds=60), run_id="inventory")
    assert result.diff == ""
    assert any("successful read action" in e.get("reason", "") for e in events(store))


def test_inventory_does_not_create_free_tools_or_model_calls(tmp_path):
    task, workspace, store, runtime = fixture_run(tmp_path)
    result = runtime.run(task, workspace, ScriptedModel([{"done": True}]),
        RunConfig(repository_inventory=True, max_tool_calls=0, max_seconds=60), run_id="inventory")
    assert result.status == "blocked"
    assert not any(e["type"] in {"model", "harness_tool"} for e in events(store))


def test_inventory_cancellation_records_the_receipt_without_calling_model(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEAGENTBENCH_EXECUTOR", "nsjail")
    task, workspace, store, runtime = fixture_run(tmp_path)
    class Sandbox:
        def __init__(self, path):
            pass
        def execute(self, *args, **kwargs):
            return ProcessResult(None, "partial", "Harness cancellation requested", "cancelled")
    monkeypatch.setattr("codeagentbench.sandbox.nsjail.NsjailSandbox", Sandbox)
    result = runtime.run(task, workspace, ScriptedModel([]),
        RunConfig(repository_inventory=True, max_steps=0, max_seconds=60), run_id="inventory")
    assert result.status == "cancelled"
    assert not any(e["type"] == "model" for e in events(store))
    assert next(e for e in events(store) if e["type"] == "harness_tool")["receipt"]["status"] == "cancelled"


@pytest.mark.parametrize("phase", ["after_inventory_receipt", "after_checkpoint"])
def test_confirmed_inventory_is_not_reexecuted_or_recounted_on_resume(tmp_path, phase):
    task, workspace, store, runtime = fixture_run(tmp_path)
    config = RunConfig(repository_inventory=True, max_seconds=60)
    def crash(actual):
        if actual == phase:
            raise RuntimeError("controlled crash")
    with pytest.raises(RuntimeError, match="controlled crash"):
        runtime.run(task, workspace, ScriptedModel([{"command": "python --version"}]),
            config, run_id="inventory", failure_injector=crash)
    result = runtime.run(task, workspace, ScriptedModel([{"done": True}]), config, run_id="inventory", resume=True)
    journal = [json.loads(s) for s in (store.run_dir("inventory") / "actions.jsonl").read_text().splitlines()]
    assert sum(e["type"] == "intent" and e["action_id"] == "inventory-inventory" for e in journal) == 1
    assert sum(e["type"] == "receipt" and e["action_id"] == "inventory-inventory" for e in journal) == 1
    assert sum(e["type"] == "harness_tool" for e in events(store)) == 1
    assert result.state.inventory_consumed
    budget = json.loads((store.run_dir("inventory") / "summary.json").read_text())["budget"]
    assert budget["tool_calls"] == (1 if phase == "after_inventory_receipt" else 2)


def test_unknown_inventory_outcome_is_not_replayed(tmp_path):
    task, workspace, store, runtime = fixture_run(tmp_path)
    config = RunConfig(repository_inventory=True, max_seconds=60)
    def crash(phase):
        if phase == "after_inventory_intent":
            raise RuntimeError("controlled crash")
    with pytest.raises(RuntimeError, match="controlled crash"):
        runtime.run(task, workspace, ScriptedModel([]), config, run_id="inventory", failure_injector=crash)
    with pytest.raises(RuntimeError, match="outcome is unknown"):
        runtime.run(task, workspace, ScriptedModel([]), config, run_id="inventory", resume=True)


def test_stored_old_config_remains_resumable_without_inventing_an_inventory(tmp_path):
    task, workspace, store, runtime = fixture_run(tmp_path)
    config = RunConfig(max_seconds=60)
    def crash(phase):
        if phase == "after_checkpoint":
            raise RuntimeError("controlled crash")
    with pytest.raises(RuntimeError, match="controlled crash"):
        runtime.run(task, workspace, ScriptedModel([{"command": "python --version"}]), config,
            run_id="inventory", failure_injector=crash)
    path = store.run_dir("inventory") / "run.json"
    metadata = json.loads(path.read_text())
    metadata["config"].pop("repository_inventory")
    path.write_text(json.dumps(metadata))
    runtime.run(task, workspace, ScriptedModel([{"done": True}]), config, run_id="inventory", resume=True)
    assert not any(e["type"] == "harness_tool" for e in events(store))


def test_large_inventory_returns_complete_names_and_an_explicit_limit(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    for i in range(600):
        (package / ("directory_with_a_long_real_name_" + str(i))).mkdir()
    receipt = BashExecutor(tmp_path).execute(ToolIntent("inventory", inventory_command(), str(tmp_path)))
    packet = inventory_packet(asdict(receipt))
    assert packet["truncated"]
    assert len(receipt.stdout.encode("utf-8")) <= 3800
    assert all((package / name).is_dir() for name in packet["child_directories"]["package"])


@pytest.mark.skipif(os.name != "posix", reason="directory symlink on Linux")
def test_inventory_never_follows_a_directory_symlink(tmp_path):
    workspace, outside = tmp_path / "workspace", tmp_path / "outside"
    workspace.mkdir()
    (outside / "not_a_repository_directory").mkdir(parents=True)
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    receipt = BashExecutor(workspace).execute(ToolIntent("inventory", inventory_command(), str(workspace)))
    packet = inventory_packet(asdict(receipt))
    assert "link" in packet["root_entries"]
    assert "link" not in packet["child_directories"]
    assert "not_a_repository_directory" not in json.dumps(packet)


def test_initial_inventory_is_preserved_and_verified_in_sft_not_a_label(tmp_path):
    task = load_manifest(ROOT / "data/manifests/demo.json").tasks[0]
    store = ArtifactStore(tmp_path / "artifacts")
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "inventory")
    runtime = AgentRuntime(store)
    result = runtime.run(task, workspace, ScriptedModel([
        {"read": {"path": "src/calculator.py", "start_line": 1, "end_line": 80}},
        {"edit": {"path": "src/calculator.py", "before": "left - right", "after": "left + right"}},
        {"done": True}]), RunConfig(repository_inventory=True, max_seconds=60), run_id="inventory")
    assert result.diff
    # This test exercises a structural export contract, not model accuracy.
    store.append_event("inventory", {"type": "evaluation", "verdict": "passed"})
    event_path = store.run_dir("inventory") / "events.jsonl"
    output = tmp_path / "export.jsonl"
    assert export_run_to_sft(event_path, output)
    row = json.loads(output.read_text())
    assert row["framework_tool_calls"] == 1 and row["tool_calls"] == 3
    samples = examples(row, events(store), policy="recent-history-v3")
    assert len(samples) == 3
    assert all("repository_inventory" in json.loads(s["messages"][1]["content"]) for s in samples)
    assert all(inventory_command() not in s["messages"][-1]["content"] for s in samples)
    corrupted = [dict(e) for e in events(store) if e["type"] != "harness_tool"]
    with pytest.raises(ValueError, match="real harness receipt"):
        examples(row, corrupted, policy="recent-history-v3")
