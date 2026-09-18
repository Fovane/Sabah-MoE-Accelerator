"""Backing-store contracts used by the runtime.

The production implementation is intentionally small: ``ExpertBank`` is the
model-aware implementation and this module provides the explicit source
protocol plus a deterministic in-memory test source.  A file-backed source is
represented by the bank's read-only mmap mode; it never copies or rewrites the
GGUF.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from sabah.runtime.identity import ExpertId


class ExpertSource(Protocol):
    """Minimal source contract consumed by cache and executor code."""

    mode: str

    def read(self, identity: ExpertId, role: str) -> np.ndarray:
        """Return an exact, read-only uint8 view for one expert role."""


class MemoryExpertSource:
    """Small deterministic source used by unit tests and integrations."""

    mode = "ram"

    def __init__(self, values: dict[tuple[ExpertId, str], bytes]):
        self._values = {
            (identity, role): np.frombuffer(data, dtype=np.uint8)
            for (identity, role), data in values.items()
        }

    def read(self, identity: ExpertId, role: str) -> np.ndarray:
        try:
            return self._values[(identity, role)]
        except KeyError as exc:
            raise KeyError("missing expert source range: %s/%s" %
                           (identity, role)) from exc

