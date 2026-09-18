"""SWE-Gym manifest normalization with deterministic, leakage-safe splits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from codeagentbench.models import TaskRecord


@dataclass(frozen=True)
class Manifest:
    """A versioned task manifest."""

    dataset: str
    revision: str
    tasks: tuple[TaskRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "tasks": [task.to_dict() for task in self.tasks],
        }


def _load_raw(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("YAML input requires `pip install pyyaml`") from exc
        return yaml.safe_load(text)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _records(raw: Any) -> list[TaskRecord]:
    if isinstance(raw, dict):
        rows = raw.get("tasks", raw.get("data", raw.get("train", [])))
    else:
        rows = raw
    if not isinstance(rows, list):
        raise ValueError("manifest must contain a list under `tasks` or `data`")
    return [TaskRecord.from_dict(row) for row in rows]


def load_manifest(path: str | Path) -> Manifest:
    """Load JSON/JSONL/YAML and normalize rows to stable contracts."""

    path = Path(path)
    raw = _load_raw(path)
    if isinstance(raw, dict):
        dataset = str(raw.get("dataset", "unknown"))
        revision = str(raw.get("revision", "unlocked"))
    else:
        dataset, revision = "unknown", "unlocked"
    manifest = Manifest(dataset, revision, tuple(_records(raw)))
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid manifest: " + "; ".join(errors))
    return manifest


def save_manifest(manifest: Manifest, path: str | Path) -> None:
    """Write a deterministic JSON manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def validate_manifest(manifest: Manifest) -> list[str]:
    """Return validation errors without attempting to run repositories."""

    errors: list[str] = []
    seen: set[str] = set()
    for index, task in enumerate(manifest.tasks):
        prefix = f"tasks[{index}]"
        if not task.instance_id:
            errors.append(f"{prefix}.instance_id is empty")
        elif task.instance_id in seen:
            errors.append(f"duplicate instance_id: {task.instance_id}")
        seen.add(task.instance_id)
        if not task.repo:
            errors.append(f"{prefix}.repo is empty")
        if not task.base_commit:
            errors.append(f"{prefix}.base_commit is empty")
        if not task.issue.strip():
            errors.append(f"{prefix}.issue is empty")
        if task.split not in {"smoke", "dev", "eval", "train", "unspecified"}:
            errors.append(f"{prefix}.split has unsupported value {task.split!r}")
    return errors


def grouped_split(
    tasks: Iterable[TaskRecord],
    *,
    smoke: int = 10,
    dev: int = 30,
    evaluation: int = 50,
) -> list[TaskRecord]:
    """Assign deterministic splits while keeping a repository/PR group together."""

    items = list(tasks)
    groups: dict[str, list[TaskRecord]] = {}
    for task in items:
        key = task.group_id or task.repo
        groups.setdefault(key, []).append(task)
    ordered = sorted(
        groups.items(),
        key=lambda pair: hashlib.sha256(pair[0].encode("utf-8")).hexdigest(),
    )
    limits = [("smoke", smoke), ("dev", dev), ("eval", evaluation)]
    counts = {name: 0 for name, _ in limits}
    result: list[TaskRecord] = []
    for _, group in ordered:
        split = "train"
        for name, limit in limits:
            if counts[name] + len(group) <= limit:
                split = name
                counts[name] += len(group)
                break
        result.extend(
            TaskRecord(
                instance_id=task.instance_id,
                repo=task.repo,
                base_commit=task.base_commit,
                issue=task.issue,
                split=split,
                group_id=task.group_id,
                eval_spec=task.eval_spec,
                metadata=task.metadata,
            )
            for task in group
        )
    return result


def import_swe_gym(source: str | Path, *, dataset: str = "SWE-Gym", revision: str, limit: int | None = None) -> Manifest:
    """Import a local export or a Hugging Face dataset revision.

    The optional HF path is deliberately an adapter: the normalized manifest is
    frozen locally before any agent rollout is allowed.
    """

    source_path = Path(source)
    if source_path.exists():
        raw = _load_raw(source_path)
        rows = _records(raw)
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "HF import requires `pip install codeagentbench[dataset]`"
            ) from exc
        from codeagentbench.adapters.swe_gym import normalize_swe_gym_row

        dataset_rows = load_dataset(str(source), revision=revision, split="train", streaming=limit is not None)
        rows = []
        for index, row in enumerate(dataset_rows):
            if limit is not None and index >= limit:
                break
            rows.append(normalize_swe_gym_row(dict(row)))
    rows = grouped_split(rows)
    manifest = Manifest(dataset, revision, tuple(rows))
    errors = validate_manifest(manifest)
    if errors:
        raise ValueError("invalid imported manifest: " + "; ".join(errors))
    return manifest
