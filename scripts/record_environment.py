"""Record package versions and selected private asset digests, never credentials.

This is an integrity inventory, not an external backup. Run with a private
--output path and explicit --assets directories after their writers finish.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys


def asset_digests(roots: list[Path]) -> dict:
    result = {}
    for root in roots:
        root = root.resolve(strict=True)
        candidates = [root] if root.is_file() else sorted(root.rglob("*"))
        files = {}
        for path in candidates:
            # Never traverse linked external caches or worktrees.
            relative = Path(path.name) if root.is_file() else path.relative_to(root)
            if any(part in {".git", ".repo_cache", "workspaces", "__pycache__"} for part in relative.parts):
                continue
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root if root.is_dir() else root.parent):
                continue
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"asset changed during inventory: {relative}")
            files[str(relative)] = {"bytes": after.st_size, "sha256": digest.hexdigest()}
        result[str(root)] = files
    return result


def environment() -> dict:
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        # Do not collect environment variables, pip configuration or direct URLs.
        "python_packages": sorted({(d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]}),
        "note": "Version inventory, not a fully reproducible binary lock or external backup",
    }
    if Path("/etc/os-release").is_file():
        result["os_release"] = Path("/etc/os-release").read_text()
    if sys.platform == "linux":
        run = subprocess.run(["dpkg-query", "-W", "-f=${Package}\t${Version}\n"], capture_output=True, text=True, timeout=30)
        result["debian_packages"] = run.stdout.splitlines() if run.returncode == 0 else {"returncode": run.returncode}
    try:
        import torch
        result["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "hip": torch.version.hip}
        if torch.cuda.is_available():
            result["torch"]["gpus"] = [{"name": torch.cuda.get_device_name(i), "vram_bytes": torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("inventory is append-only; choose a new output path")
    receipt = {"environment": environment(), "assets": asset_digests(args.assets)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "asset_roots": len(receipt["assets"]), "files": sum(map(len, receipt["assets"].values()))}))


if __name__ == "__main__":
    main()
