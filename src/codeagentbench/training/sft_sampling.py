"""Sampling weights for audited next-action records.

Weights change how often an existing record is drawn; they never create a new
source action or change the distinct-task count. The aligned-data readiness
gate must run on the unchanged source file before this module is used.
"""

from __future__ import annotations

from typing import Any

from codeagentbench.adapters.action import parse_action


def action_weight(record: dict[str, Any], done_weight: float) -> float:
    if record.get("next_action_only_loss") is not True:
        raise ValueError("action weighting requires audited next-action records")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "assistant":
        raise ValueError("action weighting requires a final assistant target")
    action = parse_action(messages[-1]["content"])
    return done_weight if action.done else 1.0
