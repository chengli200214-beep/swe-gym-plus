"""Export audited run events into line-oriented SFT records.

Runbook section 8 requires every training record to retain its provenance:
``task_id``, ``run_id``, ``messages``, ``assistant_only_loss``, the evaluation
verdict, the original events path, the patch, runtime, token usage and API cost.
Those facts already exist in the run directory, so this module collects them
instead of discarding them at export time.

A record stays usable when the sibling artifacts are missing: unavailable fields
are reported as ``None``/absent rather than invented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeagentbench.runtime import parse_action
from codeagentbench.training.sft import trajectory_to_sft


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read a sibling JSON artifact, tolerating absence and malformed content."""

    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def collect_provenance(events_path: str | Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the section 8 provenance block for one trajectory."""

    events_path = Path(events_path)
    run_dir = events_path.parent

    run_json = _read_json(run_dir / "run.json") or {}
    summary = _read_json(run_dir / "summary.json") or {}
    checkpoint = _read_json(run_dir / "checkpoint.json") or {}

    evaluation = next((event for event in events if event.get("type") == "evaluation"), None)

    prompt_tokens = sum(int(event.get("prompt_tokens") or 0) for event in events if event.get("type") == "model")
    completion_tokens = sum(int(event.get("completion_tokens") or 0) for event in events if event.get("type") == "model")
    cost_usd = sum(float(event.get("cost_usd") or 0.0) for event in events if event.get("type") == "model")
    tool_calls = sum(1 for event in events if event.get("type") == "tool")

    budget = summary.get("budget") or checkpoint.get("budget") or {}
    duration = None
    if isinstance(evaluation, dict) and evaluation.get("duration_seconds") is not None:
        duration = evaluation.get("duration_seconds")
    elif budget.get("seconds") is not None:
        duration = budget.get("seconds")

    return {
        "task_id": run_json.get("task_id") or summary.get("task_id"),
        "run_id": run_json.get("run_id") or summary.get("run_id"),
        "events_path": str(events_path),
        "evaluation_verdict": (evaluation or {}).get("verdict"),
        "patch": summary.get("diff"),
        "agent_status": summary.get("status"),
        "duration_seconds": duration,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost_usd,
        "tool_calls": tool_calls,
    }


def export_run_to_sft(
    events_path: str | Path,
    output_path: str | Path,
    *,
    successful_only: bool = True,
    provenance: bool = True,
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
    if provenance:
        record.update(collect_provenance(events_path, parseable_events))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return True
