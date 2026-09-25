from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.check_data_split import validate_split


ROOT = Path(__file__).parents[1]
SPLIT = ROOT / "data/splits/swegym-moto-v1.json"
MANIFESTS = [ROOT / "data/manifests/swegym-smoke.json", ROOT / "data/manifests/swegym-moto-candidates.json"]


def test_frozen_split_covers_all_imported_tasks() -> None:
    assert validate_split(SPLIT, MANIFESTS) == {
        "train": 6,
        "dev": 4,
        "test_candidates": 1,
        "excluded_prior_train": 1,
        "unassigned": 8,
    }


def test_training_rows_cannot_contain_dev_or_test_tasks(tmp_path: Path) -> None:
    train_file = tmp_path / "train.jsonl"
    train_file.write_text(json.dumps({"task_id": "getmoto__moto-5876"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-train task"):
        validate_split(SPLIT, MANIFESTS, train_file)


def test_prior_training_task_is_not_a_test_candidate() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    assert "getmoto__moto-4950" in split["excluded_prior_train"]
    assert "getmoto__moto-4950" not in split["test_candidates"]


def test_duplicate_task_partition_is_rejected(tmp_path: Path) -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    split["train"].append(split["dev"][0])
    path = tmp_path / "split.json"
    path.write_text(json.dumps(split), encoding="utf-8")
    with pytest.raises(ValueError, match="appears in both"):
        validate_split(path, MANIFESTS)
