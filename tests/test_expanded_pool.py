import json
from pathlib import Path

from codeagentbench.tasks.manifest import load_manifest


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
