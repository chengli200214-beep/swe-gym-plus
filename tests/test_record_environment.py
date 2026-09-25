import hashlib
from pathlib import Path
import pytest

from scripts.record_environment import asset_digests


def test_inventory_hashes_selected_assets_and_skips_cache(tmp_path):
    (tmp_path / "weights.bin").write_bytes(b"weights")
    (tmp_path / ".repo_cache").mkdir()
    (tmp_path / ".repo_cache/ignored").write_text("large cache")
    result = asset_digests([tmp_path])[str(tmp_path.resolve())]
    assert result == {"weights.bin": {"bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}}


def test_inventory_does_not_follow_external_link(tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("do not include")
    try:
        (private / "link").symlink_to(outside)
    except OSError:
        pytest.skip("symlink unavailable")
    assert asset_digests([private])[str(private.resolve())] == {}
