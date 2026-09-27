"""Bound real interaction history without discarding runtime interventions.

See docs/adr/0001-context-history-probe.md for the diagnostic tradeoff.
Tool output is observation data; only separate runtime messages are warnings.
"""
from __future__ import annotations

import json

from codeagentbench.adapters.action import parse_action

POLICIES = {"native", "last-action-v2", "recent-history-v3"}
INTERVENTIONS = ("Harness warning:", "Harness checkpoint:")


def recent_history(messages: list[dict[str, str]], *, turns: int = 4) -> list[dict[str, str]]:
    """Keep the issue, four interactions and currently active interventions.

    Latest stdout/stderr retain 4000 characters each; older outputs retain 1000
    each. Actions remain intact. Invalid action/receipt pairings fail closed.
    Repeated compaction is idempotent, including output-truncation flags.
    """
    if turns < 1 or len(messages) < 2 or [m["role"] for m in messages[:2]] != ["system", "user"]:
        raise ValueError("expected system + initial user task and a positive turn limit")
    result = [dict(m) for m in messages[:2]]
    receipts = [i for i, m in enumerate(messages[2:], 2)
                if m["role"] == "user" and m["content"].startswith(("Tool result:\n", "Protocol result:\n"))]
    previous_receipt = 1
    pairs = []
    for index in receipts:
        actions = [i for i in range(previous_receipt + 1, index) if messages[i]["role"] == "assistant"]
        if len(actions) != 1:
            raise ValueError("real receipt must follow exactly one assistant action")
        pairs.append((actions[0], index))
        previous_receipt = index
    if not pairs:
        return result + [dict(m) for m in messages[2:]]
    selected = pairs[-turns:]
    for position, (action_index, receipt_index) in enumerate(selected):
        if messages[receipt_index]["content"].startswith("Protocol result:\n"):
            rejection = json.loads(messages[receipt_index]["content"].split("\n", 1)[1])
            if not isinstance(rejection, dict) or not isinstance(rejection.get("error"), str):
                raise ValueError("invalid protocol rejection receipt")
            result.extend([dict(messages[action_index]), dict(messages[receipt_index])])
        else:
            result.extend(_tool_pair(messages[action_index], messages[receipt_index], position == len(selected) - 1))
        if position != len(selected) - 1:
            continue
        seen = set()
        for message in messages[receipt_index + 1:]:
            content = message["content"]
            if message["role"] == "user" and content.startswith(INTERVENTIONS) and content not in seen:
                result.append(dict(message))
                seen.add(content)
    return result


def _tool_pair(assistant: dict[str, str], user: dict[str, str], latest: bool) -> list[dict[str, str]]:
    action = parse_action(assistant["content"])
    if not action.executable:
        raise ValueError("real receipt must follow an executable action")
    receipt = json.loads(user["content"].split("\n", 1)[1])
    required = ("exit_code", "stdout", "stderr", "timed_out")
    if not isinstance(receipt, dict) or not set(required) <= receipt.keys():
        raise ValueError("incomplete real tool receipt")
    bounded = {key: receipt[key] for key in required}
    limit = 4000 if latest else 1000
    for key in ("stdout", "stderr"):
        text = str(bounded[key])
        bounded[key] = text[:limit]
        if len(text) > limit or receipt.get(key + "_truncated") is True:
            bounded[key + "_truncated"] = True
    return [
        {"role": "assistant", "content": json.dumps(
            action.to_dict(), ensure_ascii=False, separators=(",", ":"))},
        {"role": "user", "content": "Tool result:\n" + json.dumps(bounded, ensure_ascii=False)},
    ]


def prepare_context(messages: list[dict[str, str]], policy: str) -> list[dict[str, str]]:
    if policy == "recent-history-v3":
        return recent_history(messages)
    if policy == "last-action-v2":
        # This remains an actively used training format and experimental control,
        # not a fallback for the corrected policy. Remove after the decision gate.
        from codeagentbench.training.action_context import compact_context
        return compact_context(messages)
    if policy != "native":
        raise ValueError("unknown local prompt policy")
    return messages
