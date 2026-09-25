"""Independent multi-rollout coordination with one shared task budget."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

from codeagentbench.adapters.model import ChatModel
from codeagentbench.evaluation.report import aggregate_results
from codeagentbench.harness.budget import BudgetExceeded, BudgetLedger
from codeagentbench.models import Candidate, EvaluationResult, RunConfig, TaskRecord, Verdict
from codeagentbench.rollout.selector import RuleCandidateSelector
from codeagentbench.runtime import AgentRuntime
from codeagentbench.sandbox.workspace import WorkspaceManager
from codeagentbench.storage.artifacts import ArtifactStore
from codeagentbench.verification.evaluator import Evaluator


@dataclass(frozen=True)
class RolloutGroupResult:
    group_id: str
    candidates: tuple[Candidate, ...]
    selection: object
    metrics: dict


class RolloutCoordinator:
    """Run k candidates on separate workspaces, select, then evaluate."""

    def __init__(self, artifact_root: str = "artifacts", selector: RuleCandidateSelector | None = None) -> None:
        self.artifact_root = artifact_root
        self.store = ArtifactStore(artifact_root)
        self.runtime = AgentRuntime(self.store)
        self.selector = selector or RuleCandidateSelector()
        self.evaluator = Evaluator(f"{artifact_root}/evaluations")

    def run(
        self,
        task: TaskRecord,
        model_factory: Callable[[int], ChatModel],
        config: RunConfig,
        *,
        group_id: str,
    ) -> RolloutGroupResult:
        count = max(1, min(config.candidate_count, 4))
        ledger = BudgetLedger(config.max_tokens, config.max_seconds, config.max_cost_usd, config.max_tool_calls)
        candidates: list[Candidate] = []
        manager = WorkspaceManager(self.artifact_root, cache_root=f"{self.artifact_root}/.repo_cache")
        for index in range(count):
            run_id = f"{group_id}-candidate-{index}"
            if (ledger.snapshot.remaining_tokens <= 0 or ledger.snapshot.remaining_seconds <= 0
                    or ledger.snapshot.tool_calls >= ledger.snapshot.tool_call_limit):
                candidates.append(Candidate(str(index), run_id, "", "blocked", metadata={"failure_reason": "shared task budget exhausted before rollout"}))
                break
            try:
                before = ledger.snapshot
                workspace = manager.create(task, run_id)
                result = self.runtime.run(task, workspace, model_factory(index), replace(config, candidate_count=1), run_id=run_id, ledger=ledger)
            except (BudgetExceeded, OSError, RuntimeError) as exc:
                candidates.append(Candidate(str(index), run_id, "", "blocked", metadata={"failure_reason": str(exc)}))
                break
            candidates.append(Candidate(
                candidate_id=str(index),
                run_id=run_id,
                diff=result.diff,
                status=result.status,
                visible_test_passed=result.visible_test_passed,
                changed_files=workspace.changed_files(),
                duration_seconds=ledger.snapshot.seconds - before.seconds,
                token_count=ledger.snapshot.tokens - before.tokens,
                cost_usd=ledger.snapshot.cost_usd - before.cost_usd,
                metadata={"failure_reason": result.failure_reason},
            ))
        selection = self.selector.select(candidates)
        evaluated: list[Candidate] = []
        for candidate in candidates:
            marked = replace(candidate, metadata={**candidate.metadata, "selected": candidate.candidate_id == selection.selected_candidate_id})
            if ledger.snapshot.remaining_seconds <= 0 or ledger.snapshot.tool_calls >= ledger.snapshot.tool_call_limit:
                evaluation = EvaluationResult(Verdict.BLOCKED, None, None, None, reason="shared task budget exhausted before formal evaluation")
            else:
                evaluation = self.evaluator.evaluate(task, marked, timeout_seconds=min(600.0, ledger.snapshot.remaining_seconds))
                try:
                    ledger.consume(seconds=evaluation.duration_seconds, tool_calls=1)
                except BudgetExceeded:
                    evaluation = replace(evaluation, verdict=Verdict.BLOCKED, reason="formal evaluation exceeded shared task budget")
            if evaluation.verdict is Verdict.BLOCKED:
                marked = replace(marked, metadata={**marked.metadata, "failure_reason": evaluation.reason})
            self.store.append_event(candidate.run_id, {"type": "evaluation", **evaluation.to_dict()})
            evaluated.append(replace(marked, evaluation=evaluation))
        metrics = aggregate_results(evaluated, candidate_count=count)
        self.store.save_summary(group_id, {"group_id": group_id, "selection": asdict(selection), "metrics": metrics, "candidates": [{"candidate_id": item.candidate_id, "evaluation": item.evaluation.to_dict() if item.evaluation else None} for item in evaluated]})
        return RolloutGroupResult(group_id, tuple(evaluated), selection, metrics)
