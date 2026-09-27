"""Tie typed edits to genuine, versioned observations; no answer selection."""
from __future__ import annotations

import json

from codeagentbench.adapters.action import AgentAction


def grounded_command(action: AgentAction, observations: list[dict]) -> str:
    if action.edit is None:
        return action.tool_command()
    matching = [o for o in observations if o["path"] == action.edit.path
                and action.edit.before in o["text"]]
    if not matching:
        if any(o["path"] == action.edit.path for o in observations):
            raise ValueError("before text is not an exact substring of the successful read action; "
                             "preserve every leading space and newline from its decoded text field, "
                             "and use a small unique substring; no edit ran")
        raise ValueError("edit requires a successful read action of this exact path and before text; "
                         "search with line numbers, then read actual implementation source before editing")
    return action.tool_command(expected_sha256=matching[-1]["sha256"])


def observe_source(observations: list[dict], action: AgentAction, receipt: dict) -> None:
    """Call only for the actual executed action and its durable tool receipt."""
    if receipt["exit_code"] != 0 or receipt["timed_out"]:
        return
    if action.edit is not None:
        observations[:] = [o for o in observations if o["path"] != action.edit.path]
    if action.read is None:
        return
    result = json.loads(receipt["stdout"])
    required = {"path", "sha256", "text", "start_line", "end_line", "next_line"}
    if (not isinstance(result, dict) or set(result) != required
            or result["path"] != action.read.path
            or result["start_line"] != action.read.start_line
            or type(result["end_line"]) is not int
            or not result["start_line"] <= result["end_line"] <= action.read.end_line
            or not isinstance(result["text"], str) or not result["text"]
            or not isinstance(result["sha256"], str) or len(result["sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in result["sha256"])):
        raise ValueError("invalid actual source read receipt")
    # Retain at most eight observations and discard older versions of this file.
    observations[:] = [o for o in observations if o["path"] != result["path"]
                       or o["sha256"] == result["sha256"]][-7:] + [result]
