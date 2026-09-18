"""Explicit adapter boundary for optional mini-swe-agent integration."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version


@dataclass(frozen=True)
class MiniSweAdapter:
    """Record the upstream agent choice without claiming to reimplement it."""

    package: str = "mini-swe-agent"
    requested_version: str = "0.15.0"

    def installed_version(self) -> str | None:
        try:
            return version(self.package)
        except PackageNotFoundError:
            return None

    def metadata(self) -> dict[str, str | None]:
        return {"package": self.package, "requested_version": self.requested_version, "installed_version": self.installed_version()}
