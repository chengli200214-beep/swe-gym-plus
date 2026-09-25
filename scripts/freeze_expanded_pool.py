"""Freeze new task partitions before observing model outcomes (Moto-only pilot)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from codeagentbench.adapters.swe_gym import normalize_swe_gym_row
from codeagentbench.tasks.manifest import Manifest, save_manifest

REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, default=Path("data/splits/swegym-moto-v1.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("freeze output already exists; never silently replace a split")
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    known = {t for v in previous.values() if isinstance(v, list) for t in v}
    source = Path(hf_hub_download("SWE-Gym/SWE-Gym", "data/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION))
    rows = pq.read_table(source).to_pylist()
    # Only repository and test-count constraints. Do not inspect gold diff size,
    # reward, or evaluation outcomes when selecting the pilot population.
    fresh = [r for r in rows if r["repo"] == "getmoto/moto" and r["instance_id"] not in known
             and 1 <= len(r["FAIL_TO_PASS"]) <= 5 and len(r["PASS_TO_PASS"]) <= 30]
    fresh.sort(key=lambda r: hashlib.sha256(("cab-expanded-v1:" + r["instance_id"]).encode()).hexdigest())
    if len(fresh) < 80:
        raise ValueError(f"need 80 eligible new tasks, found {len(fresh)}")
    heldout, train = fresh[:20], fresh[20:80]
    ids = lambda items: [r["instance_id"] for r in items]
    train_ids = previous["train"] + ids(train)
    eval_ids = ids(heldout)
    selected = [r for r in rows if r["instance_id"] in set(train_ids + eval_ids + previous["dev"])]
    tasks = tuple(normalize_swe_gym_row(r, split="eval" if r["instance_id"] in eval_ids else "dev" if r["instance_id"] in previous["dev"] else "train") for r in selected)
    args.output.mkdir(parents=True)
    save_manifest(Manifest("SWE-Gym", REVISION, tasks), args.output / "manifest.json")
    split = {"dataset": "SWE-Gym", "revision": REVISION, "train": train_ids, "dev": previous["dev"], "eval": eval_ids,
             "excluded_prior_exposure": sorted(known - set(train_ids) - set(previous["dev"])),
             "notes": "Moto-only tractable pilot; not SWE-Gym overall performance. Split frozen before outcomes. 60 fresh training candidates plus 7 previous training IDs; 20 fresh evaluation IDs. Failed admission stays in reports; no outcome-based replacement."}
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    receipt = {"source_revision": REVISION, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "dataset_card_license": "MIT", "upstream_repository_licenses_still_apply": True,
               "selection": "repo=getmoto/moto; 1<=FAIL_TO_PASS count<=5; PASS_TO_PASS count<=30; hash seed cab-expanded-v1",
               "eligible_fresh": len(fresh), "train_candidates": len(train_ids), "eval": len(eval_ids),
               "manifest_sha256": hashlib.sha256((args.output / "manifest.json").read_bytes()).hexdigest(),
               "split_sha256": hashlib.sha256((args.output / "split.json").read_bytes()).hexdigest()}
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
