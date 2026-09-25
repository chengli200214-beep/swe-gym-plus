"""Check task-level partitions and optional private training data for leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PARTITIONS = ("train", "dev", "test_candidates", "excluded_prior_train", "unassigned")


def validate_split(split_path: Path, manifests: list[Path], train_file: Path | None = None) -> dict[str, int]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    members: dict[str, str] = {}
    for partition in PARTITIONS:
        task_ids = split.get(partition)
        if not isinstance(task_ids, list) or not all(isinstance(task_id, str) and task_id for task_id in task_ids):
            raise ValueError(f"{partition} must be a list of non-empty task IDs")
        for task_id in task_ids:
            if task_id in members:
                raise ValueError(f"task {task_id} appears in both {members[task_id]} and {partition}")
            members[task_id] = partition

    imported: set[str] = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset") != split.get("dataset") or manifest.get("revision") != split.get("revision"):
            raise ValueError(f"dataset or revision mismatch: {manifest_path}")
        for task in manifest["tasks"]:
            task_id = task["instance_id"]
            if task_id in imported:
                raise ValueError(f"task {task_id} appears in multiple manifests")
            imported.add(task_id)
    missing = imported - members.keys()
    unknown = members.keys() - imported
    if missing or unknown:
        raise ValueError(f"split/manifest mismatch: missing={sorted(missing)} unknown={sorted(unknown)}")

    if train_file is not None:
        for line_number, line in enumerate(train_file.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            task_id = json.loads(line).get("task_id")
            if members.get(task_id) != "train":
                raise ValueError(f"training row {line_number} has non-train task: {task_id}")
    return {partition: len(split[partition]) for partition in PARTITIONS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("split", type=Path)
    parser.add_argument("manifests", type=Path, nargs="+")
    parser.add_argument("--train-file", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_split(args.split, args.manifests, args.train_file), sort_keys=True))


if __name__ == "__main__":
    main()
