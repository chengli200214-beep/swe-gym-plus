"""Whole-task budget accounting shared by rollouts, retries and evaluation."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetExceeded(RuntimeError):
    """Raised when an operation would exceed the task budget."""


@dataclass(frozen=True)
class BudgetSnapshot:
    token_limit: int
    seconds_limit: float
    cost_limit_usd: float
    tool_call_limit: int
    tokens: int
    seconds: float
    cost_usd: float
    tool_calls: int

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.token_limit - self.tokens)

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.seconds_limit - self.seconds)

    @property
    def remaining_cost_usd(self) -> float:
        return max(0.0, self.cost_limit_usd - self.cost_usd)


class BudgetLedger:
    """Monotonic ledger; recovery and retries never reset already-spent cost."""

    def __init__(self, token_limit: int, seconds_limit: float, cost_limit_usd: float, tool_call_limit: int) -> None:
        self._snapshot = BudgetSnapshot(token_limit, seconds_limit, cost_limit_usd, tool_call_limit, 0, 0.0, 0.0, 0)

    @property
    def snapshot(self) -> BudgetSnapshot:
        return self._snapshot

    def can_spend(self, *, tokens: int = 0, seconds: float = 0.0, cost_usd: float = 0.0, tool_calls: int = 0) -> bool:
        current = self._snapshot
        return (
            current.tokens + tokens <= current.token_limit
            and current.seconds + seconds <= current.seconds_limit
            and current.cost_usd + cost_usd <= current.cost_limit_usd
            and current.tool_calls + tool_calls <= current.tool_call_limit
        )

    def consume(self, *, tokens: int = 0, seconds: float = 0.0, cost_usd: float = 0.0, tool_calls: int = 0) -> BudgetSnapshot:
        if min(tokens, seconds, cost_usd, tool_calls) < 0:
            raise ValueError("budget consumption must be non-negative")
        if not self.can_spend(tokens=tokens, seconds=seconds, cost_usd=cost_usd, tool_calls=tool_calls):
            raise BudgetExceeded(f"task budget exceeded: {self._snapshot}")
        current = self._snapshot
        self._snapshot = BudgetSnapshot(
            current.token_limit,
            current.seconds_limit,
            current.cost_limit_usd,
            current.tool_call_limit,
            current.tokens + tokens,
            current.seconds + seconds,
            current.cost_usd + cost_usd,
            current.tool_calls + tool_calls,
        )
        return self._snapshot

    def restore(self, snapshot: BudgetSnapshot) -> None:
        """Restore only a checkpoint that is not behind the current ledger."""

        current = self._snapshot
        if (snapshot.tokens, snapshot.seconds, snapshot.cost_usd, snapshot.tool_calls) < (
            current.tokens,
            current.seconds,
            current.cost_usd,
            current.tool_calls,
        ):
            raise ValueError("cannot restore a budget checkpoint that erases spending")
        self._snapshot = snapshot
