"""Freeze a small, reproducible SWE-Gym training-candidate manifest.

The source is imported from the pinned SWE-Gym revision. Candidate selection
uses task/test size for tractability, not model outcomes. The official patch
remains evaluation-only and is never exposed through AgentTaskView.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from codeagentbench.tasks.manifest import Manifest, load_manifest, save_manifest


REVISION = "bb94ed9e39bbeb96a7fcbfb533b80f25a7fd59cb"
CANDIDATE_IDS = (
    "getmoto__moto-6190",
    "getmoto__moto-7023",
    "getmoto__moto-6208",
    "getmoto__moto-6410",
    "getmoto__moto-5699",
    "getmoto__moto-6509",
    "getmoto__moto-5949",
    "getmoto__moto-7061",
    "getmoto__moto-7168",
    "getmoto__moto-6641",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = load_manifest(args.source)
    if source.revision != REVISION:
        raise ValueError(f"expected locked revision {REVISION}, got {source.revision}")
    by_id = {task.instance_id: task for task in source.tasks}
    missing = set(CANDIDATE_IDS) - by_id.keys()
    if missing:
        raise ValueError(f"missing candidates: {sorted(missing)}")
    selected = tuple(by_id[instance_id] for instance_id in CANDIDATE_IDS)
    if any(task.split != "train" or task.repo != "getmoto/moto" for task in selected):
        raise ValueError("all selected candidates must be getmoto train tasks")
    save_manifest(Manifest(source.dataset, source.revision, selected), args.output)
    print(f"saved {len(selected)} train candidates to {args.output}")


if __name__ == "__main__":
    main()
