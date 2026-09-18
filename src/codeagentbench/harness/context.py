"""Evidence-aware context management with auditable compression."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class Evidence:
    """A message or observation with a retention priority."""

    evidence_id: str
    kind: str
    text: str
    source: str = ""
    priority: int = 50
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompressionRecord:
    before_chars: int
    after_chars: int
    retained_ids: tuple[str, ...]
    dropped_ids: tuple[str, ...]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"retained_ids": list(self.retained_ids), "dropped_ids": list(self.dropped_ids)}


class ContextManager:
    """Keep issue/diff/failure evidence and summarize low-value output first."""

    KEEP_KINDS = {"issue", "diff", "test_failure", "confirmed_location", "assumption"}

    def __init__(self, issue: str = "") -> None:
        self._items: list[Evidence] = []
        self.compressions: list[CompressionRecord] = []
        if issue:
            self.add("issue", issue, priority=100, confirmed=True, evidence_id="issue")

    @property
    def items(self) -> tuple[Evidence, ...]:
        return tuple(self._items)

    def add(self, kind: str, text: str, *, source: str = "", priority: int = 50, confirmed: bool = False, evidence_id: str | None = None) -> Evidence:
        identifier = evidence_id or sha256(f"{kind}:{source}:{text}".encode("utf-8")).hexdigest()[:16]
        item = Evidence(identifier, kind, text, source, priority, confirmed)
        self._items.append(item)
        return item

    def render(self, max_chars: int | None = None) -> str:
        text = "\n\n".join(f"[{item.kind}] {item.text}" for item in self._items)
        if max_chars is None or len(text) <= max_chars:
            return text
        self.compress(max_chars)
        return "\n\n".join(f"[{item.kind}] {item.text}" for item in self._items)

    def compress(self, max_chars: int) -> CompressionRecord:
        if max_chars < 128:
            raise ValueError("max_chars must leave room for meaningful evidence")
        before = self.render() if self._items else ""
        ranked = sorted(
            enumerate(self._items),
            key=lambda pair: (
                pair[1].kind in self.KEEP_KINDS,
                pair[1].confirmed,
                pair[1].priority,
                pair[0],
            ),
            reverse=True,
        )
        retained: list[Evidence] = []
        used = 0
        for _, item in ranked:
            rendered = f"[{item.kind}] {item.text}"
            extra = len(rendered) + (2 if retained else 0)
            if used + extra <= max_chars or (item.kind in self.KEEP_KINDS and not retained):
                retained.append(item)
                used += extra
        retained.sort(key=lambda item: self._items.index(item))
        retained_ids = tuple(item.evidence_id for item in retained)
        dropped_ids = tuple(item.evidence_id for item in self._items if item.evidence_id not in retained_ids)
        dropped_text = " ".join(item.text for item in self._items if item.evidence_id in dropped_ids)
        summary = f"Compressed {len(dropped_ids)} low-priority observations. Evidence log retained: {dropped_text[:240]}" if dropped_ids else "No observations dropped."
        self._items = retained
        after = self.render() if self._items else ""
        record = CompressionRecord(len(before), len(after), retained_ids, dropped_ids, summary)
        self.compressions.append(record)
        return record
