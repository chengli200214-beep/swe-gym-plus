"""Filesystem artifact store; metadata is small and logs stay out of the DB."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from codeagentbench.models import RunConfig, RunState


class ArtifactStore:
    """Persist run configuration, events, checkpoints and final summaries."""

    def __init__(self, root: str | Path = "artifacts") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,159}", run_id):
            raise ValueError("invalid run id")
        path = self.root / "runs" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start_run(self, run_id: str, task_id: str, config: RunConfig) -> Path:
        path = self.run_dir(run_id)
        if (path / "run.json").exists():
            raise ValueError("run already exists; resume it explicitly or choose a new run id")
        self._atomic_json(path / "run.json", {"run_id": run_id, "task_id": task_id, "config": config.to_dict()})
        return path

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        path = self.run_dir(run_id) / "events.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def save_checkpoint(self, state: RunState, *, budget: dict[str, Any], digest: str) -> None:
        path = self.run_dir(state.run_id) / "checkpoint.json"
        self._atomic_json(path, {"state": state.to_dict(), "budget": budget, "workspace_digest": digest})

    def load_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        path = self.run_dir(run_id) / "checkpoint.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_summary(self, run_id: str, summary: dict[str, Any]) -> None:
        self._atomic_json(self.run_dir(run_id) / "summary.json", summary)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
