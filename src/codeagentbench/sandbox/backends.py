"""Shared, fail-closed executor selection for all execution paths."""
from __future__ import annotations

import os

BACKENDS = frozenset({"local", "bwrap", "nsjail"})
ISOLATED_BACKENDS = frozenset({"bwrap", "nsjail"})


def selected_backend() -> str:
    backend = os.getenv("CODEAGENTBENCH_EXECUTOR", "local")
    if backend not in BACKENDS:
        raise ValueError(f"unknown executor backend: {backend}")
    return backend
