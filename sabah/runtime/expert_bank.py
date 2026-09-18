"""
Sabah / runtime / expert_bank

The backing store for expert weights.

The user's GGUF is opened READ-ONLY and is never written to, moved, rewritten
or repacked. Sabah reads per-expert byte ranges straight out of it, which is
only possible because `model_inspector` has already PROVEN that each expert's
slice is contiguous and quantization-block aligned. If that proof failed, this
class must not be constructed.

Two backing modes:

  mmap  - the file is memory-mapped and the OS page cache is the bank. Works on
          any machine, including ones where the bank is far larger than RAM.
          This is what STORAGE_BACKED planning means in practice.
  ram   - the selected blocks' experts are copied once into an anonymous RAM
          buffer (optionally pinned), removing the page-cache hop from the
          steady-state path. Only chosen when the planner says it fits.

Neither mode changes a single byte of the artifact.
"""
from __future__ import annotations

import os
import re
import ctypes
import numpy as np

from sabah.runtime import rt


def shard_paths(path: str) -> list:
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    base = path[:m.start()]
    return [("%s-%05d-of-%05d.gguf" % (base, i, n)) for i in range(1, n + 1)]


class ExpertBank:
    """Random access to (block, expert, role) weight slices."""

    ROLES = ("gate", "up", "down")

    def __init__(self, model_profile, mode: str = "mmap", blocks=None,
                 pinned: bool = False):
        if not model_profile.supported:
            raise ValueError("model is not supported; refusing to build a bank")
        if not model_profile.experts_contiguous:
            raise ValueError(
                "expert slices are not contiguous in this artifact; Sabah "
                "cannot address experts directly and must not guess")

        self.model = model_profile
        self.mode = mode
        self.pinned = pinned
        self.n_experts = model_profile.n_experts

        # index: (block, role) -> descriptor
        self.desc = {}
        for t in model_profile.expert_tensors:
            self.desc[(t["block"], t["role"])] = t
        self.blocks = sorted({b for (b, _) in self.desc})
        if blocks is not None:
            keep = set(blocks)
            missing = keep - set(self.blocks)
            if missing:
                raise ValueError("model has no blocks %s" % sorted(missing))
            self.blocks = sorted(keep)

        self._maps = {}
        self._ram = {}
        self._pinned_ptrs = []

        paths = shard_paths(model_profile.path)
        for t in self.desc.values():
            si = t["shard"]
            if si not in self._maps and si < len(paths):
                self._maps[si] = np.memmap(paths[si], dtype=np.uint8, mode="r")

        if mode == "ram":
            self._load_ram()
        elif mode != "mmap":
            raise ValueError("mode must be 'mmap' or 'ram'")

    # ------------------------------------------------------------------
    def qtype(self, block: int, role: str) -> int:
        return rt.QTYPE[self.desc[(block, role)]["qtype"]]

    def expert_bytes(self, block: int, role: str) -> int:
        return int(self.desc[(block, role)]["per_expert_bytes"])

    def block_expert_bytes(self, block: int) -> int:
        """Bytes for one expert's full FFN (gate + up + down) in this block."""
        return sum(self.expert_bytes(block, r) for r in self.ROLES)

    def bank_bytes(self, blocks=None) -> int:
        bs = self.blocks if blocks is None else blocks
        return sum(self.block_expert_bytes(b) * self.n_experts for b in bs)

    # ------------------------------------------------------------------
    def slice(self, block: int, expert: int, role: str) -> np.ndarray:
        """A read-only uint8 view of exactly one expert's weight slice."""
        if not (0 <= expert < self.n_experts):
            raise IndexError("expert %d out of range" % expert)
        key = (block, role)
        if self.mode == "ram":
            buf = self._ram[key]
            n = self.expert_bytes(block, role)
            return buf[expert * n:(expert + 1) * n]
        t = self.desc[key]
        n = int(t["per_expert_bytes"])
        off = int(t["offset"]) + expert * n
        return self._maps[t["shard"]][off:off + n]

    def expert_ptr_bytes(self, block: int, expert: int) -> list:
        """[(role, ndarray)] for one expert, in gate/up/down order."""
        return [(r, self.slice(block, expert, r)) for r in self.ROLES]

    # ------------------------------------------------------------------
    def _load_ram(self):
        total = self.bank_bytes()
        for b in self.blocks:
            for r in self.ROLES:
                t = self.desc[(b, r)]
                n = int(t["per_expert_bytes"]) * self.n_experts
                src = self._maps[t["shard"]]
                off = int(t["offset"])
                if self.pinned:
                    p = rt.check_ptr(rt.lib().sabah_host_alloc(n),
                                     "pinned host alloc %d B" % n)
                    self._pinned_ptrs.append(p)
                    buf = np.ctypeslib.as_array(
                        ctypes.cast(p, ctypes.POINTER(ctypes.c_uint8)), (n,))
                else:
                    buf = np.empty(n, dtype=np.uint8)
                buf[:] = src[off:off + n]
                self._ram[(b, r)] = buf
        self.loaded_bytes = total

    def close(self):
        for p in self._pinned_ptrs:
            rt.lib().sabah_host_free(p)
        self._pinned_ptrs = []
        self._ram.clear()
        self._maps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def describe(self) -> str:
        L = []
        L.append("backing    : %s%s" % (self.mode, " (pinned)" if self.pinned else ""))
        L.append("blocks     : %d (%s)"
                 % (len(self.blocks),
                    "%d..%d" % (self.blocks[0], self.blocks[-1]) if self.blocks else "-"))
        L.append("experts    : %d per block" % self.n_experts)
        qs = sorted({self.desc[(b, r)]["qtype"] for b in self.blocks for r in self.ROLES})
        L.append("quant mix  : %s" % ", ".join(qs))
        per = sorted({self.block_expert_bytes(b) for b in self.blocks})
        L.append("per expert : %s bytes" % ", ".join("{:,}".format(x) for x in per))
        L.append("bank       : %.3f GB" % (self.bank_bytes() / 1e9))
        return "\n".join(L)
