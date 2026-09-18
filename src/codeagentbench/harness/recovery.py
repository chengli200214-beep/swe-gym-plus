"""Two-phase action journal and conservative worker recovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from codeagentbench.models import ToolIntent, ToolReceipt


@dataclass(frozen=True)
class RecoveryDecision:
    action_id: str
    decision: str
    reason: str


class ActionJournal:
    """Append-only intent/receipt journal.

    A side-effecting action with an intent but no receipt is never replayed
    blindly. The caller must compare the recorded pre-digest with the current
    workspace and stop or create a fresh attempt when the outcome is uncertain.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_intent(self, intent: ToolIntent) -> None:
        self._append({"type": "intent", **asdict(intent)})

    def record_receipt(self, receipt: ToolReceipt) -> None:
        self._append({"type": "receipt", **asdict(receipt)})

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def pending(self) -> list[dict[str, Any]]:
        records = self.records()
        intents = {item["action_id"]: item for item in records if item.get("type") == "intent"}
        receipts = {item["action_id"] for item in records if item.get("type") == "receipt"}
        return [intent for action_id, intent in intents.items() if action_id not in receipts]

    def decide_recovery(self, current_digest: str) -> list[RecoveryDecision]:
        decisions: list[RecoveryDecision] = []
        for intent in self.pending():
            action_id = intent["action_id"]
            if intent.get("side_effect", True):
                if intent.get("pre_digest") and intent["pre_digest"] != current_digest:
                    reason = "workspace changed after an unacknowledged side-effecting action; outcome is unknown"
                else:
                    reason = "side-effecting action has no receipt; outcome is unknown"
                decisions.append(RecoveryDecision(action_id, "stop", reason))
            else:
                decisions.append(RecoveryDecision(action_id, "retry", "read-only action has no receipt"))
        return decisions
