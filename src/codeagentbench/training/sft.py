"""Convert audited interaction trajectories to assistant-action SFT records."""

from __future__ import annotations

from typing import Any, Iterable


def trajectory_to_sft(events: Iterable[dict[str, Any]], *, successful_only: bool = False) -> dict[str, Any] | None:
    """Preserve issue → action → observation order for later TRL/PEFT training."""

    events = list(events)
    if successful_only and not any(event.get("type") == "evaluation" and event.get("verdict") == "passed" for event in events):
        return None
    messages: list[dict[str, str]] = []
    for event in events:
        event_type = event.get("type")
        if event_type == "prompt":
            messages.append({"role": "user", "content": str(event.get("content", ""))})
        elif event_type == "model":
            messages.append({"role": "assistant", "content": str(event.get("content", ""))})
        elif event_type == "tool":
            receipt = event.get("receipt", event)
            messages.append({"role": "tool", "content": str(receipt.get("stdout", "")) + str(receipt.get("stderr", ""))})
    if not any(message["role"] == "assistant" for message in messages):
        return None
    return {"messages": messages, "assistant_only_loss": True}
