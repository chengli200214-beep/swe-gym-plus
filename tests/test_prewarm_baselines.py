import hashlib
import json
from pathlib import Path

import pytest

from scripts.prewarm_baselines import eligible_tasks
from scripts.run_clean_training import EVALUATION_LIMITS, model_source_receipt, source_receipt, wait_for_baseline_prefetch


def test_prefetch_keeps_base_first_order_and_only_previously_admitted_tasks(tmp_path):
    (tmp_path / "quality").mkdir()
    for task in ("a", "b", "c", "d"):
        (tmp_path / "quality" / (task + ".json")).write_text(json.dumps({"admitted": task != "c"}))
    assert list(eligible_tasks({"dev": ["a", "b"], "eval": ["c", "d", "missing"]}, tmp_path)) == ["a"]


def test_absent_prefetch_does_not_change_existing_pipeline(tmp_path):
    wait_for_baseline_prefetch(tmp_path, tmp_path, tmp_path)


@pytest.mark.skipif(__import__("os").name == "nt", reason="Linux flock")
def test_prefetch_join_requires_frozen_identity_and_completed_receipt(tmp_path):
    for name in ("manifest", "split"):
        (tmp_path / (name + ".json")).write_text("{}")
    model = tmp_path / "model"; model.mkdir()
    (tmp_path / "base-prefetch.lock").touch()
    identity = {name + "_sha256": hashlib.sha256((tmp_path / (name + ".json")).read_bytes()).hexdigest() for name in ("manifest", "split")}
    identity.update(model_path=str(model.resolve()), model_sha256=model_source_receipt(model), limits=dict(EVALUATION_LIMITS), source_sha256=source_receipt(Path(__file__).resolve().parents[1]))
    (tmp_path / "base-prefetch.json").write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="incomplete"):
        wait_for_baseline_prefetch(tmp_path, tmp_path, model)
    (tmp_path / "base-prefetch-report.json").write_text('{"status":"completed"}')
    wait_for_baseline_prefetch(tmp_path, tmp_path, model)
    identity["split_sha256"] = "different"
    (tmp_path / "base-prefetch.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="split"):
        wait_for_baseline_prefetch(tmp_path, tmp_path, model)


def test_model_snapshot_detects_byte_changes_and_ignores_hidden_download_metadata(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"before")
    (tmp_path / ".download").write_bytes(b"metadata")
    receipt = model_source_receipt(tmp_path)
    assert list(receipt) == ["weights.bin"]
    (tmp_path / "weights.bin").write_bytes(b"after")
    assert model_source_receipt(tmp_path) != receipt
