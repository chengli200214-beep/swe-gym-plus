"""Sampling weights for audited next-action records.

Weights change how often an existing record is drawn; they never create a new
source action or change the distinct-task count. The aligned-data readiness
gate must run on the unchanged source file before this module is used.
"""

from __future__ import annotations

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
