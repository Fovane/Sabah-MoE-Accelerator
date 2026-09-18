"""Stable identities for routed expert objects.

An expert number is only meaningful inside a block.  Keeping this identity in
one small value object prevents the easy-to-miss bug where expert 17 from one
block aliases expert 17 from another block or where a role is copied with the
wrong byte range.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class ExpertId:
    """Unambiguous identity of one routed expert."""

    block: int
    expert: int

    def __post_init__(self):
        if self.block < 0:
            raise ValueError("block must be non-negative")
        if self.expert < 0:
            raise ValueError("expert must be non-negative")

    def as_tuple(self) -> tuple[int, int]:
        return self.block, self.expert


@dataclass(frozen=True)
class ExpertRange:
    """One exact, read-only byte range in a GGUF shard."""

    identity: ExpertId
    role: str
    shard: int
    offset: int
    size: int
    qtype: str

    def __post_init__(self):
        if not self.role:
            raise ValueError("expert tensor role is required")
        if self.shard < 0 or self.offset < 0 or self.size <= 0:
            raise ValueError("invalid expert byte range")

    @property
    def end(self) -> int:
        return self.offset + self.size

