"""Honest metrics for actual selection, oracle coverage and execution cost."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from codeagentbench.models import Candidate, Verdict
from codeagentbench.rollout.selector import RuleCandidateSelector


def aggregate_results(candidates: Iterable[Candidate], *, candidate_count: int | None = None) -> dict[str, Any]:
    """Aggregate results without conflating candidate-pool coverage with selection."""

    items = list(candidates)
    evaluated = [item for item in items if item.evaluation is not None]
    selected = [item for item in evaluated if item.metadata.get("selected") is True]
    passed = sum(item.evaluation.verdict is Verdict.PASSED for item in evaluated)
    selected_passed = sum(item.evaluation.verdict is Verdict.PASSED for item in selected)
    selector = RuleCandidateSelector()
    errors = Counter(item.metadata.get("failure_reason", "") for item in items if item.metadata.get("failure_reason"))
    return {
        "candidate_count": candidate_count or len(items),
        "evaluated_candidates": len(evaluated),
        "formal_pass_rate": passed / len(evaluated) if evaluated else None,
        "actual_selection_success_rate": selected_passed / len(selected) if selected else None,
        "oracle_coverage_at_k": selector.oracle_coverage_at_k(evaluated),
        "total_tokens": sum(item.token_count for item in items),
        "total_cost_usd": round(sum(item.cost_usd for item in items), 6),
        "total_duration_seconds": round(sum(item.duration_seconds for item in items), 3),
        "failure_categories": dict(errors),
    }


def write_report(results: dict[str, Any], path: str | Path) -> None:
    """Write a machine-readable report for a frozen evaluation run."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
