import hashlib
import json

import pytest

from scripts.run_clean_training import ordered_pairs, paired_order, source_receipt, verify_preflight


def test_source_receipt_includes_uncommitted_source_and_ignores_private_data(tmp_path):
    source = tmp_path / "src/codeagentbench/runtime.py"
    source.parent.mkdir(parents=True)
    source.write_text("first")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/run.py").write_text("runner")
    (tmp_path / "private.jsonl").write_text("not public")
    before = source_receipt(tmp_path)
    assert set(before) == {"src/codeagentbench/runtime.py", "scripts/run.py"}
    source.write_text("second")
    assert source_receipt(tmp_path) != before


def test_preflight_requires_exact_frozen_manifest_and_split(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    receipt = {}
    for name in ("manifest", "split"):
        path = pool / (name + ".json")
        path.write_text("{}")
        receipt[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "quality-preflight.json").write_text(json.dumps(receipt))
    verify_preflight(tmp_path, pool)
    (pool / "manifest.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="different frozen manifest"):
        verify_preflight(tmp_path, pool)


def test_parallel_pairs_preserve_frozen_order_and_alternate_model_order():
    assert paired_order(0) == ["base", "sft"]
    assert paired_order(1) == ["sft", "base"]
    assert paired_order(2) == ["base", "sft"]
    completed = {"second": {"task_id": "second"}, "first": {"task_id": "first"}}
    assert ordered_pairs(["first", "second", "third"], completed) == [completed["first"], completed["second"]]
