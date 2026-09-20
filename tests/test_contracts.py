from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.harness.budget import BudgetExceeded, BudgetLedger
from codeagentbench.harness.context import ContextManager
from codeagentbench.harness.recovery import ActionJournal
from codeagentbench.models import Candidate, EvalSpec, RunConfig, TaskRecord, ToolIntent, Verdict
from codeagentbench.rollout.selector import RuleCandidateSelector
from codeagentbench.rollout.coordinator import RolloutCoordinator
from codeagentbench.runtime import AgentRuntime, parse_action
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.tasks.manifest import load_manifest
from codeagentbench.verification.evaluator import Evaluator


ROOT = Path(__file__).parents[1]


def demo_task() -> TaskRecord:
    manifest = load_manifest(ROOT / "data/manifests/demo.json")
    return manifest.tasks[0]


def test_agent_view_does_not_leak_gold_patch() -> None:
    task = TaskRecord("x", "repo", "abc", "fix", eval_spec=EvalSpec(gold_patch="SECRET-GOLD"))
    view = task.agent_view().to_dict()
    assert "gold_patch" not in view
    assert "SECRET-GOLD" not in json.dumps(view)


def test_parse_action_repairs_unescaped_quotes_inside_command() -> None:
    action = parse_action(
        '{"command": "findstr /n /c:"launch_template" moto\\\\ec2\\\\models.py", '
        '"done": false, "message": "locate code"}'
    )
    assert action.command == 'findstr /n /c:"launch_template" moto\\ec2\\models.py'
    assert action.done is False


def test_parse_action_repairs_explicit_truncated_fenced_command() -> None:
    action = parse_action('```json\n{"command": "printf \'hello\\n\'')
    assert action.command == "printf 'hello\\n'"
    assert action.done is False
    assert "truncated" in action.message


def test_parse_action_repairs_prose_and_shell_single_quote_escape() -> None:
    action = parse_action(
        "The previous edit failed; I will retry.\n"
        "```json\n"
        "{\"command\": \"sed -i 's/return \\\'old\\\'/return \\\'new\\\'/g' file.py\", \"done\": false}\n"
        "```"
    )
    assert action.command == "sed -i 's/return 'old'/return 'new'/g' file.py"
    assert action.done is False


def test_parse_action_extracts_labeled_edit_command() -> None:
    action = parse_action(
        "The JSON action was malformed.\n"
        "**Edit Command:**\n"
        "```bash\npython -c \"print('repair')\"\n```\n"
        "**Test Command:**\n```bash\npython -m pytest -q\n```"
    )
    assert action.command == "python -c \"print('repair')\""


def test_parse_deepseek_dsml_shell_call() -> None:
    action = parse_action(
        '<｜｜DSML｜｜ calls>\n'
        '<｜｜DSML｜｜ invoke name="shell">\n'
        '<｜｜DSML｜｜ parameter name="command" string="true">'
        'findstr /n /i &quot;AddExecutor&quot; moto\\dynamodb\\parsing\\executors.py'
        '</｜｜DSML｜｜ parameter>\n'
        '</｜｜DSML｜｜ invoke>'
    )
    assert action.command == 'findstr /n /i "AddExecutor" moto\\dynamodb\\parsing\\executors.py'


def test_parse_deepseek_dsml_command_tag() -> None:
    action = parse_action(
        '<｜｜DSML｜｜ invoke name="shell"><command>type tests\\test_api.py</command></｜｜DSML｜｜ invoke>'
    )
    assert action.command == "type tests\\test_api.py"


def test_parse_deepseek_dsml_invoke_command() -> None:
    action = parse_action(
        '<｜｜DSML｜｜ invoke name="command" string="true">python -m pytest -q</｜｜DSML｜｜ parameter>'
    )
    assert action.command == "python -m pytest -q"


def test_parse_deepseek_dsml_inline_command() -> None:
    action = parse_action(
        r'<｜｜DSML｜｜ invoke name="command": "python -c \"print(1)\"",'
        r'"done": false>'
    )
    assert action.command == 'python -c "print(1)"'


def test_parse_last_action_from_multiple_json_candidates() -> None:
    action = parse_action(
        'Planning... {"command":"dir","done":false} '
        '{"command":"python -m pytest -q","done":false}'
    )
    assert action.command == "python -m pytest -q"


def test_parse_action_does_not_accept_synthetic_multi_step_done() -> None:
    action = parse_action(
        '{"command":"dir","done":false} '
        '{"command":"type moto.py","done":false} '
        '{"command":"git diff","done":true}'
    )
    assert action.command == "dir"
    assert action.done is False


def test_parse_deepseek_dsml_plain_parameter() -> None:
    action = parse_action(
        '<｜｜DSML｜｜ invoke name="shell"><parameter name="command">dir</parameter></｜｜DSML｜｜ invoke>'
    )
    assert action.command == "dir"


def test_budget_is_cumulative_and_rejects_overrun() -> None:
    ledger = BudgetLedger(10, 10, 1, 2)
    ledger.consume(tokens=8, seconds=1, cost_usd=0.2, tool_calls=1)
    try:
        ledger.consume(tokens=3)
    except BudgetExceeded:
        pass
    else:
        raise AssertionError("expected budget overrun")
    assert ledger.snapshot.tokens == 8


def test_context_compression_keeps_issue_and_failure() -> None:
    context = ContextManager("the issue must remain")
    context.add("search", "x" * 500, priority=1)
    context.add("test_failure", "assertion failed at tests/test_x.py", priority=90)
    record = context.compress(180)
    assert record.dropped_ids
    rendered = context.render()
    assert "the issue must remain" in rendered
    assert "assertion failed" in rendered


def test_recovery_stops_unknown_side_effect(tmp_path: Path) -> None:
    journal = ActionJournal(tmp_path / "actions.jsonl")
    journal.record_intent(ToolIntent("a1", "touch file", str(tmp_path), pre_digest="before"))
    decisions = journal.decide_recovery("after")
    assert decisions[0].decision == "stop"
    assert "unknown" in decisions[0].reason


def test_runtime_produces_real_diff_and_checkpoint(tmp_path: Path) -> None:
    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "run-1")
    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(
        task,
        workspace,
        ScriptedModel([
            {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text().replace('left - right', 'left + right'))\""},
            {"command": "python -m pytest -q"},
            {"done": True},
        ]),
        RunConfig(max_steps=4, max_seconds=60, max_tool_calls=4),
        run_id="run-1",
    )
    assert result.status == "completed"
    assert "left + right" in result.diff
    assert (tmp_path / "artifacts/runs/run-1/checkpoint.json").exists()
    assert (tmp_path / "artifacts/runs/run-1/actions.jsonl").exists()


def test_runtime_recovery_does_not_replay_unacknowledged_action(tmp_path: Path) -> None:
    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "crashed")
    runtime = AgentRuntime(ArtifactStore(tmp_path / "artifacts"))
    with pytest.raises(RuntimeError, match="simulated worker crash"):
        runtime.run(
            task,
            workspace,
            ScriptedModel([{"command": "python -c \"print('side effect')\""}]),
            RunConfig(max_steps=2, max_seconds=60, max_tool_calls=2),
            run_id="crashed",
            failure_injector=lambda phase: (_ for _ in ()).throw(RuntimeError("simulated worker crash")) if phase == "after_intent" else None,
        )
    decisions = runtime.recover("crashed", workspace)
    assert decisions[0]["decision"] == "stop"


def test_evaluator_uses_fresh_workspace_and_formal_test(tmp_path: Path) -> None:
    task = demo_task()
    candidate_workspace = WorkspaceManager(tmp_path / "candidate").create(task, "candidate-1")
    (candidate_workspace.path / "src/calculator.py").write_text(
        'def add(left, right):\n    """Return the sum of two numbers."""\n    return left + right\n',
        encoding="utf-8",
    )
    diff = candidate_workspace.diff()
    candidate = Candidate("c1", "rollout-1", diff, "completed", visible_test_passed=True)
    result = Evaluator(tmp_path / "evaluations").evaluate(task, candidate)
    assert result.verdict is Verdict.PASSED
    assert result.fail_to_pass is True


def test_selector_does_not_use_formal_label() -> None:
    selector = RuleCandidateSelector()
    weak = Candidate("weak", "r", "diff", "completed", visible_test_passed=False)
    strong = Candidate("strong", "r", "diff2", "completed", visible_test_passed=True)
    decision = selector.select([weak, strong])
    assert decision.selected_candidate_id == "strong"


def test_multi_rollout_uses_independent_workspaces_and_reports_selection(tmp_path: Path) -> None:
    task = demo_task()
    group = RolloutCoordinator(str(tmp_path / "artifacts")).run(
        task,
        lambda _: ScriptedModel([
            {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text().replace('left - right', 'left + right'))\""},
            {"command": "python -m pytest -q"},
            {"done": True},
        ]),
        RunConfig(candidate_count=2, max_steps=4, max_seconds=120, max_tool_calls=8),
        group_id="group-1",
    )
    assert len(group.candidates) == 2
    assert len({item.run_id for item in group.candidates}) == 2
    assert group.selection.selected_candidate_id in {"0", "1"}
    assert group.metrics["oracle_coverage_at_k"] is True
