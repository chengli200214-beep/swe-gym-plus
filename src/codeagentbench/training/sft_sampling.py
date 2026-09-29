"""Sampling weights for audited next-action records.

Weights change how often an existing record is drawn; they never create a new
source action or change the distinct-task count. The aligned-data readiness
gate must run on the unchanged source file before this module is used.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import random
from typing import Any

from codeagentbench.adapters.action import parse_action


def target_kind(record: dict[str, Any]) -> str:
    if record.get("next_action_only_loss") is not True:
        raise ValueError("action weighting requires audited next-action records")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("action weighting requires a final assistant target")
    action = parse_action(messages[-1]["content"])
    if action.done:
        return "done"
    for kind in ("edit", "read", "search"):
        if getattr(action, kind) is not None:
            return kind
    return "command"


def action_weight(record: dict[str, Any], done_weight: float, typed_weight: float = 1.0) -> float:
    kind = target_kind(record)
    if kind == "done":
        return done_weight
    if kind in {"edit", "read", "search"}:
        return typed_weight
    return 1.0


def covered_weighted_indices(weights: list[float], seed: int) -> list[int]:
    """One guaranteed pass over real records plus one seeded weighted pass.

    Replacement-only sampling can omit most unique examples in a small corpus.
    This schedule changes exposure, never the set of source records or labels.
    """
    if not weights or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError("coverage sampling requires positive finite weights")
    rng = random.Random(seed)
    indices = list(range(len(weights)))
    indices.extend(rng.choices(range(len(weights)), weights=weights, k=len(weights)))
    rng.shuffle(indices)
    return indices


class FixedIndexSampler:
    """Repeatable PyTorch DataLoader sampler for a frozen exposure schedule."""

    def __init__(self, indices: list[int]) -> None:
        self.indices = tuple(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def coverage_receipt(indices: list[int], kinds: list[str]) -> dict[str, Any]:
    """Record the exact train-only draw schedule without copying source text."""
    if not indices or len(set(indices)) != len(kinds) or any(i < 0 or i >= len(kinds) for i in indices):
        raise ValueError("coverage schedule omitted or exceeded a source record")
    return {
        "unique_source_records_covered": len(kinds),
        "drawn_action_kinds": dict(Counter(kinds[i] for i in indices)),
        "schedule_sha256": hashlib.sha256(json.dumps(indices).encode()).hexdigest(),
    }
