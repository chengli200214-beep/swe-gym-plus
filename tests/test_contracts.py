from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from codeagentbench.adapters.model import ScriptedModel
from codeagentbench.harness.budget import BudgetExceeded, BudgetLedger
from codeagentbench.harness.context import ContextManager
from codeagentbench.harness.recovery import ActionJournal
from codeagentbench.models import Candidate, EvalSpec, EvaluationResult, RunConfig, TaskRecord, ToolIntent, Verdict
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


def test_snapshot_cache_reuses_concurrent_winner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager = WorkspaceManager(tmp_path / "workspaces")
    original_replace = Path.replace

    def competing_replace(source: Path, target: Path) -> Path:
        if source.name == "snapshot":
            shutil.copytree(source, target)
            raise FileExistsError(target)
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", competing_replace)
    workspace = manager.create(demo_task(), "concurrent-winner")
    assert (workspace.path / ".git").is_dir()


def test_agent_view_does_not_leak_gold_patch() -> None:
    task = TaskRecord(
        "x", "repo", "abc", "fix",
        eval_spec=EvalSpec(
            gold_patch="SECRET-GOLD",
            test_patch="SECRET-TEST-PATCH",
            test_command="python -m pytest SECRET-HIDDEN-TEST",
        ),
        metadata={"agent_test_command": "python -m pytest tests/test_visible.py"},
    )
    view = task.agent_view().to_dict()
    assert "gold_patch" not in view
    assert "SECRET-GOLD" not in json.dumps(view)
    assert "SECRET-TEST-PATCH" not in json.dumps(view)
    assert "SECRET-HIDDEN-TEST" not in json.dumps(view)
    assert view["allowed_test_command"] == "python -m pytest tests/test_visible.py"
    assert TaskRecord("y", "repo", "abc", "fix", eval_spec=task.eval_spec).agent_view().allowed_test_command == ""


def test_system_prompt_requires_new_symbol_import_check() -> None:
    prompt = AgentRuntime._system_prompt()
    assert "newly referenced name is defined or imported" in prompt


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


def test_budget_restore_cannot_erase_a_later_counter() -> None:
    from codeagentbench.harness.budget import BudgetSnapshot

    ledger = BudgetLedger(100, 100, 10, 10)
    ledger.consume(tokens=5, tool_calls=2)
    with pytest.raises(ValueError, match="erases spending"):
        ledger.restore(BudgetSnapshot(100, 100, 10, 10, 6, 0.0, 0.0, 1))


def test_budget_rejects_nonfinite_cost_and_mismatched_checkpoint() -> None:
    from codeagentbench.harness.budget import BudgetSnapshot

    ledger = BudgetLedger(100, 100, 10, 10)
    with pytest.raises(ValueError, match="finite"):
        ledger.consume(cost_usd=float("nan"))
    assert ledger.can_spend(cost_usd=float("nan")) is False
    with pytest.raises(ValueError, match="different limits"):
        ledger.restore(BudgetSnapshot(101, 100, 10, 10, 0, 0.0, 0.0, 0))


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
    assert result.visible_test_passed is True
    assert "left + right" in result.diff
    assert (tmp_path / "artifacts/runs/run-1/checkpoint.json").exists()
    assert (tmp_path / "artifacts/runs/run-1/actions.jsonl").exists()


def test_completion_without_post_edit_test_is_unverified(tmp_path: Path) -> None:
    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "untested")
    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(
        task,
        workspace,
        ScriptedModel([
            {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text().replace('left - right', 'left + right'))\""},
            {"command": "python -m pytest --version"},
            {"done": True},
        ]),
        RunConfig(max_steps=4, max_seconds=60, max_tool_calls=4),
        run_id="untested",
    )
    summary = json.loads((tmp_path / "artifacts/runs/untested/summary.json").read_text())
    assert result.status == "completed"
    assert result.visible_test_passed is False
    assert "unverified" in result.failure_reason
    assert summary["visible_test_passed"] is False


def test_edit_after_passing_test_invalidates_visible_verification(tmp_path: Path) -> None:
    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "edited-after-test")
    result = AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(
        task,
        workspace,
        ScriptedModel([
            {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text().replace('left - right', 'left + right'))\""},
            {"command": "python -m pytest -q"},
            {"command": "python -c \"from pathlib import Path; p=Path('src/calculator.py'); p.write_text(p.read_text() + '# late edit\\n')\""},
            {"done": True},
        ]),
        RunConfig(max_steps=5, max_seconds=60, max_tool_calls=5),
        run_id="edited-after-test",
    )
    assert result.status == "completed"
    assert result.visible_test_passed is False
    assert "unverified" in result.failure_reason


def test_no_patch_checkpoint_survives_context_compression(tmp_path: Path) -> None:
    class CapturingModel(ScriptedModel):
        def __init__(self) -> None:
            super().__init__(
                [{"command": f"python -c \"print('x' * 16000 + str({i}))\""} for i in range(6)]
                + [{"done": True}]
            )
            self.last_messages: list[dict[str, str]] = []

        def complete(self, messages: list[dict[str, str]], *, temperature: float = 0.0):
            self.last_messages = list(messages)
            return super().complete(messages, temperature=temperature)

    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "no-patch")
    model = CapturingModel()
    AgentRuntime(ArtifactStore(tmp_path / "artifacts")).run(
        task,
        workspace,
        model,
        RunConfig(max_steps=7, max_seconds=60, max_tool_calls=6),
        run_id="no-patch",
    )
    assert any("Evidence summary after context compression" in item["content"] for item in model.last_messages)
    assert "Harness checkpoint: 6 actions" in model.last_messages[-1]["content"]


def test_repeated_command_stops_even_when_output_changes(tmp_path: Path) -> None:
    task = demo_task()
    workspace = WorkspaceManager(tmp_path / "workspaces").create(task, "repeated")
    command = 'python -c "import uuid; print(uuid.uuid4())"'
    store = ArtifactStore(tmp_path / "artifacts")
    result = AgentRuntime(store).run(
        task,
        workspace,
        ScriptedModel([{"command": command}] * 3),
        RunConfig(max_steps=4, max_seconds=60, max_tool_calls=4),
        run_id="repeated",
    )
    events = [json.loads(line) for line in (tmp_path / "artifacts/runs/repeated/events.jsonl").read_text().splitlines()]
    assert result.status == "failed"
    assert "no progress" in result.failure_reason
    assert len([event for event in events if event["type"] == "tool"]) == 3
    assert len([event for event in events if event["type"] == "no_progress"]) == 1


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


def test_evaluator_rejects_no_remaining_time_before_workspace_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("workspace creation should not start")

    monkeypatch.setattr(WorkspaceManager, "create", forbidden)
    result = Evaluator(tmp_path / "evaluations").evaluate(
        demo_task(), Candidate("c1", "r1", "", "completed"), timeout_seconds=0,
    )
    assert result.verdict is Verdict.BLOCKED


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


def test_multi_rollout_does_not_call_model_or_evaluator_after_shared_budget_exhaustion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = RolloutCoordinator(str(tmp_path / "artifacts"))

    def forbidden(*args, **kwargs):
        raise AssertionError("budget-exhausted work must not start")

    monkeypatch.setattr(coordinator.evaluator, "evaluate", forbidden)
    group = coordinator.run(
        demo_task(), forbidden,
        RunConfig(candidate_count=2, max_tool_calls=0),
        group_id="no-budget",
    )
    assert len(group.candidates) == 1
    assert group.candidates[0].status == "blocked"
    assert group.candidates[0].evaluation.verdict is Verdict.BLOCKED
    assert group.metrics["oracle_coverage_at_k"] is False


def test_formal_evaluation_past_shared_deadline_is_not_counted_as_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    coordinator = RolloutCoordinator(str(tmp_path / "artifacts"))
    monkeypatch.setattr(
        coordinator.evaluator,
        "evaluate",
        lambda *args, **kwargs: EvaluationResult(Verdict.PASSED, 0, True, True, duration_seconds=11.0),
    )
    group = coordinator.run(
        demo_task(), lambda _: ScriptedModel([{"done": True}]),
        RunConfig(candidate_count=1, max_seconds=10),
        group_id="late-evaluation",
    )
    assert group.candidates[0].evaluation.verdict is Verdict.BLOCKED
    assert group.metrics["oracle_coverage_at_k"] is False
