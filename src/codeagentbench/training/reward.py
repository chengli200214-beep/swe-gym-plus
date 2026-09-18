"""Binary formal reward used by a future GRPO rollout adapter."""

from __future__ import annotations

from codeagentbench.models import EvaluationResult, Verdict


def formal_reward(result: EvaluationResult) -> float:
    """Return 1 only for an independent formal pass, otherwise 0."""

    return 1.0 if result.verdict is Verdict.PASSED else 0.0
