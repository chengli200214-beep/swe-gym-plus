import json
import hashlib
from pathlib import Path

from codeagentbench.tasks.manifest import load_manifest
from scripts.freeze_expanded_pool import content_sha256


def test_frozen_expanded_pool_excludes_all_prior_exposure_from_eval():
    root = Path(__file__).parents[1]
    old = json.loads((root / "data/splits/swegym-moto-v1.json").read_text())
    new = json.loads((root / "data/expanded/moto-v2/split.json").read_text())
    known = {t for v in old.values() if isinstance(v, list) for t in v}
    assert len(new["eval"]) == 20
    assert not known.intersection(new["eval"])
    assert not set(new["train"]).intersection(new["eval"] + new["dev"])
    manifest = load_manifest(root / "data/expanded/moto-v2/manifest.json")
    assert {t.instance_id for t in manifest.tasks} == set(new["train"] + new["dev"] + new["eval"])
    assert all(not t.agent_view().allowed_test_command for t in manifest.tasks)


def test_freeze_receipt_survives_git_line_ending_conversion():
    pool = Path(__file__).parents[1] / "data/expanded/moto-v2"
    receipt = json.loads((pool / "receipt.json").read_text())
    for name in ("manifest", "split"):
        path = pool / (name + ".json")
        assert content_sha256(path) == receipt[name + "_content_sha256"]
        assert hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() == receipt[name + "_lf_sha256"]
