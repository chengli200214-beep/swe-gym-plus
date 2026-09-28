"""Bounded source navigation and its evidence boundary."""
from __future__ import annotations

import json

import pytest

from codeagentbench.adapters.action import parse_action
from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.adapters.source_read import SourceRead
from codeagentbench.adapters.source_search import SourceSearch, search_command, validate_search
from codeagentbench.harness.source_evidence import grounded_command, observe_source
from codeagentbench.models import RunConfig, TaskRecord, ToolIntent
from codeagentbench.runtime import AgentRuntime, _repeat_recovery_guidance
from codeagentbench.sandbox.executor import BashExecutor
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore


def search(path="app.py", query="target"):
    return {"search": {"path": path, "query": query}, "done": False}


def read(start, end):
    return {"read": {"path": "app.py", "start_line": start, "end_line": end}, "done": False}


@pytest.mark.parametrize("payload", [
    {"path": "../app.py", "query": "target"},
    {"path": "app.py", "query": ""},
    {"path": "app.py", "query": "x" * 201},
    {"path": "app.py", "query": "bad\nquery"},
    {"path": "app.py", "query": "target", "extra": True},
])
def test_search_rejects_invalid_path_query_or_shape(payload):
    with pytest.raises(ValueError):
        validate_search(payload)


def test_directory_path_error_gives_a_safe_file_discovery_action():
    with pytest.raises(ValueError, match="one source file, not a directory"):
        validate_search({"path": "moto/rds/tests/", "query": "test_rds.py"})


def test_search_action_roundtrips_as_location_only_tool():
    action = parse_action(json.dumps(search(query="TargetName")))
    assert action.executable and action.search == SourceSearch("app.py", "TargetName")
    assert parse_action(json.dumps(action.to_dict())) == action
    assert action.tool_command() == search_command(action.search)


def test_search_returns_literal_casefold_line_matches_but_not_edit_evidence(tmp_path):
    (tmp_path / "app.py").write_text("header\nTargetName = 1\nTARGETNAME = 2\n")
    action = parse_action(json.dumps(search(query="targetname")))
    observations = []
    command = grounded_command(action, observations)
    receipt = BashExecutor(tmp_path).execute(ToolIntent("s", command, str(tmp_path), side_effect=False))
    assert receipt.exit_code == 0, receipt.stderr
    packet = json.loads(receipt.stdout)
    assert packet["match_count"] == 2
    assert [item["line"] for item in packet["matches"]] == [2, 3]
    assert len(receipt.stdout.encode("utf-8")) <= 3000
    observe_source(observations, action, vars(receipt))
    assert observations == []
    with pytest.raises(ValueError, match="successful read"):
        grounded_command(parse_action(json.dumps({
            "edit": {"path": "app.py", "before": "TargetName = 1", "after": "TargetName = 3"},
        })), observations)


def test_zero_match_recovery_guidance_prefers_shorter_literal_or_real_read():
    hint = _repeat_recovery_guidance(parse_action(json.dumps(search(query="def guessed_signature(self, **kwargs):"))))
    assert "shorten the literal query" in hint
    assert "bounded read" in hint
    assert "guess a longer signature" in hint
    prompt = AgentRuntime._system_prompt()
    assert "Search is a literal substring lookup" in prompt


def _make_navigation_run(tmp_path, responses):
    source = tmp_path / "repo"
    source.mkdir()
    lines = [f"# filler {number}" for number in range(1, 12)]
    lines[4] = "value = 1"
    (source / "app.py").write_text("\n".join(lines) + "\n")
    (source / "test_app.py").write_text(
        "import unittest\nimport app\n"
        "class TestApp(unittest.TestCase):\n"
        "    def test_value(self):\n"
        "        self.assertEqual(app.value, 2)\n"
    )
    task = TaskRecord("navigation-test", str(source), "local", "Change value in app.py from 1 to 2")
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "navigation-test-run")
    store = ArtifactStore(tmp_path / "artifacts")
    result = AgentRuntime(store).run(
        task,
        workspace,
        ScriptedModel(responses),
        RunConfig(max_steps=16, max_seconds=90, max_tool_calls=16),
        run_id="navigation-test-run",
    )
    events = [json.loads(line) for line in (store.run_dir(result.run_id) / "events.jsonl").read_text().splitlines()]
    return result, workspace, events


def test_runtime_refuses_fourth_read_then_search_allows_focused_fix(tmp_path):
    result, workspace, events = _make_navigation_run(tmp_path, [
        read(1, 2), read(3, 4), read(5, 6), read(7, 8),
        search(query="value = 1"), read(5, 5),
        {"edit": {"path": "app.py", "before": "value = 1", "after": "value = 2"}, "done": False},
        {"command": "python -m unittest -q", "done": False}, {"done": True},
    ])
    assert result.status == "completed" and result.visible_test_passed, result.failure_reason
    assert "value = 2" in result.diff
    blocked = [event for event in events if event["type"] == "source_navigation_blocked"]
    assert len(blocked) == 1 and blocked[0]["executed"] is False
    tools = [event for event in events if event["type"] == "tool"]
    actions = [json.loads(event["intent"]["action_json"]) for event in tools]
    assert sum("read" in action for action in actions) == 4
    search_event = next(event for event, action in zip(tools, actions) if "search" in action)
    assert search_event["intent"]["side_effect"] is False
    model_contexts = [event["context_messages"] for event in events if event["type"] == "model"]
    assert any("no tool call or budget was spent" in str(context) for context in model_contexts)


def test_runtime_stops_after_second_read_limit_without_running_it(tmp_path):
    result, _, events = _make_navigation_run(tmp_path, [
        read(1, 2), read(3, 4), read(5, 6), read(7, 8), read(9, 10),
    ])
    assert result.status == "failed"
    assert "navigation limit repeated" in result.failure_reason
    assert len([event for event in events if event["type"] == "tool"]) == 3
    blocked = [event for event in events if event["type"] == "source_navigation_blocked"]
    assert len(blocked) == 2 and [event["warning_repeated"] for event in blocked] == [False, True]
