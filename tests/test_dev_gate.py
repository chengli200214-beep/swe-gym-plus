import json

import pytest

from codeagentbench.models import TaskRecord
from scripts.autodl_dev_gate import freeze, load_frozen


def source(path, splits=("dev", "dev", "dev")):
    tasks = [TaskRecord("task-" + str(i), "public/repo", "base-" + str(i), "fix", split=split)
             for i, split in enumerate(splits)]
    path.write_text(json.dumps({"dataset": "SWE-Gym", "revision": "pinned", "tasks": [t.to_dict() for t in tasks]}))
    return [t.instance_id for t in tasks]


def test_gate_freezes_three_tasks_without_outcome_replacement(tmp_path):
    manifest = tmp_path / "source.json"
    ids = source(manifest)
    root = tmp_path / "gate"
    identity = freeze(manifest, root, ids)
    assert identity["tasks"] == ids and identity["no_outcome_replacement"]
    assert [t.instance_id for t in load_frozen(root)] == ids
    with pytest.raises(FileExistsError):
        freeze(manifest, root, ids)
    changed = json.loads((root / "manifest.json").read_text())
    changed["tasks"][0]["issue"] = "different"
    (root / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="changed"):
        load_frozen(root)


def test_gate_does_not_relabel_eval_or_reuse_training_tasks(tmp_path):
    manifest = tmp_path / "source.json"
    ids = source(manifest, ("dev", "eval", "train"))
    with pytest.raises(ValueError, match="silently"):
        freeze(manifest, tmp_path / "gate", ids)
    with pytest.raises(ValueError, match="three distinct"):
        freeze(manifest, tmp_path / "gate", [ids[0]] * 3)
