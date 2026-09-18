"""Export audited run events into line-oriented SFT records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeagentbench.runtime import parse_action
from codeagentbench.training.sft import trajectory_to_sft


def export_run_to_sft(
    events_path: str | Path,
    output_path: str | Path,
    *,
    successful_only: bool = True,
) -> bool:
    """Write one SFT JSONL record, dropping malformed trailing model output."""

    events = [json.loads(line) for line in Path(events_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    parseable_events: list[dict[str, Any]] = []
    for event in events:
        if event.get("type") == "model":
            try:
                parse_action(str(event.get("content", "")))
            except ValueError:
                continue
        parseable_events.append(event)
    record = trajectory_to_sft(parseable_events, successful_only=successful_only)
    if record is None:
        return False
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return True
