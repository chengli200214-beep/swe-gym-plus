"""Deterministic candidate selector that cannot see formal evaluator labels."""

from __future__ import annotations

from dataclasses import replace

from codeagentbench.models import Candidate, SelectionResult


class RuleCandidateSelector:
    """Rank candidates using visible evidence; evaluation is never a ranking input."""

    def score(self, candidate: Candidate) -> float:
        score = 0.0
        if candidate.status == "completed":
            score += 20.0
        if candidate.diff.strip():
            score += 15.0
        if candidate.visible_test_passed:
            score += 50.0
        score -= min(candidate.tool_errors, 10) * 5.0
        score -= min(len(candidate.changed_files), 10) * 0.75
        score -= min(candidate.duration_seconds / 600.0, 1.0) * 2.0
        return score

    def select(self, candidates: list[Candidate], *, require_visible_test: bool = False) -> SelectionResult:
        if not candidates:
            return SelectionResult(None, (), {}, "candidate pool is empty")
        unique: dict[str, Candidate] = {}
        for candidate in candidates:
            unique.setdefault(candidate.diff, candidate)
        ranked = sorted(unique.values(), key=lambda item: (-self.score(item), item.candidate_id))
        scores = {candidate.candidate_id: self.score(candidate) for candidate in ranked}
        if require_visible_test and not ranked[0].visible_test_passed:
            return SelectionResult(None, tuple(item.candidate_id for item in ranked), scores, "no candidate has visible passing evidence")
        if len(ranked) > 1 and scores[ranked[0].candidate_id] == scores[ranked[1].candidate_id]:
            return SelectionResult(None, tuple(item.candidate_id for item in ranked), scores, "top candidates are tied; selection rejected")
        return SelectionResult(ranked[0].candidate_id, tuple(item.candidate_id for item in ranked), scores, "selected highest evidence score")

    @staticmethod
    def with_evaluation(candidate: Candidate, evaluation) -> Candidate:
        """Attach a result after selection/evaluation for reporting only."""

        return replace(candidate, evaluation=evaluation)

    @staticmethod
    def oracle_coverage_at_k(candidates: list[Candidate]) -> bool:
        """Report candidate-pool upper bound separately from actual selection."""

        return any(candidate.evaluation is not None and candidate.evaluation.passed for candidate in candidates)
