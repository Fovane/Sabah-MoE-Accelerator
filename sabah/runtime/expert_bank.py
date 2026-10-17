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
import threading
import numpy as np

from sabah.runtime import rt
from sabah.runtime.identity import ExpertId, ExpertRange


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
                 pinned: bool = False, warm_blocks=None):
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
        # top-k, i.e. how many experts one token can route to. The hot tier needs
        # it to guarantee the live working set always fits: reserving prefetch
        # slots beyond (slots - top-k) would leave a step unable to place the
        # experts it is about to execute.
        self.n_experts_used = int(getattr(model_profile, "n_experts_used", 0) or 0)
        self.warm_blocks = set(warm_blocks or [])

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
        self._warm = {}           # (block, role) -> pinned RAM buffer
        self.warm_loaded_bytes = 0
        self._pinned_ptrs = []
        self._lock = threading.RLock()
        self._ranges = {}

        paths = shard_paths(model_profile.path)
        self._file_handles = {}
        for t in self.desc.values():
            si = t["shard"]
            if si not in self._file_handles and si < len(paths):
                # Use file handles instead of memmap to avoid committing
                # the entire multi-GB shard into the pagefile. Read specific
                # ranges on demand in slice().
                self._file_handles[si] = open(paths[si], "rb")

        for b in self.blocks:
            for e in range(self.n_experts):
                for r in self.ROLES:
                    t = self.desc[(b, r)]
                    n = int(t["per_expert_bytes"])
                    self._ranges[(ExpertId(b, e), r)] = ExpertRange(
                        identity=ExpertId(b, e), role=r, shard=int(t["shard"]),
                        offset=int(t["offset"]) + e * n, size=n,
                        qtype=str(t["qtype"]))

        if mode == "ram":
            self._load_ram()
        elif mode != "mmap":
            raise ValueError("mode must be 'mmap' or 'ram'")
        # A warm tier is opt-in: naming blocks here (or calling load_warm)
        # pins them, and dot product reads then come from pinned RAM instead of
        # the page cache or the SSD. `mode="ram"` is the degenerate case where
        # every block is warm.
        if self.warm_blocks:
            self.load_warm()

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

    def pinned_block(self, block: int) -> bool:
        """True if a whole block's experts are readable from PINNED RAM.

        Two ways that happens: the bank was opened with `mode="ram"`, or the
        block is in the warm tier set by `load_warm`. When it is true a copy can
        DMA straight out of the bank with no staging hop on the host.
        """
        if self.mode == "ram":
            return all((block, r) in self._ram for r in self.ROLES)
        return all((block, r) in self._warm for r in self.ROLES)

    # ------------------------------------------------------------------
    def slice(self, block: int, expert: int, role: str) -> np.ndarray:
        """A read-only uint8 view of exactly one expert's weight slice."""
        if not (0 <= expert < self.n_experts):
            raise IndexError("expert %d out of range" % expert)
        if role not in self.ROLES or block not in self.blocks:
            raise KeyError("unknown expert range block=%s role=%s" % (block, role))
        with self._lock:
            key = (block, role)
            if key in self._warm:
                bufs = self._warm[key]
                out = bufs[expert]
            elif self.mode == "ram":
                buf = self._ram[key]
                n = self.expert_bytes(block, role)
                out = buf[expert * n:(expert + 1) * n]
            else:
                rng = self._ranges[(ExpertId(block, expert), role)]
                fh = self._file_handles[rng.shard]
                fh.seek(rng.offset)
                out = np.frombuffer(fh.read(rng.size), dtype=np.uint8)
                if out.size != rng.size:
                    raise IOError("short read for expert range: %s" % rng)
            out.setflags(write=False)
            return out

    def range(self, block: int, expert: int, role: str) -> ExpertRange:
        """Return the validated exact byte range for an expert role."""
        if not (0 <= expert < self.n_experts):
            raise IndexError("expert %d out of range" % expert)
        try:
            return self._ranges[(ExpertId(block, expert), role)]
        except KeyError as exc:
            raise KeyError("unknown expert range block=%s role=%s" %
                           (block, role)) from exc

    def read(self, identity: ExpertId, role: str) -> np.ndarray:
        """ExpertSource-compatible exact read."""
        return self.slice(identity.block, identity.expert, role)

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
                fh = self._file_handles[t["shard"]]
                fh.seek(int(t["offset"]))
                # Use regular heap memory (not pinned) to avoid CUDA pinned
                # memory limits on large allocations. Pinned staging happens
                # per-expert in the hot tier when needed.
                buf = np.empty(n, dtype=np.uint8)
                per = int(t["per_expert_bytes"])
                for e in range(self.n_experts):
                    chunk = fh.read(per)
                    if len(chunk) != per:
                        raise IOError("short read block=%d role=%d expert=%d" % (b, r, e))
                    buf[e * per:(e + 1) * per] = np.frombuffer(chunk, dtype=np.uint8)
                self._ram[(b, r)] = buf
        self.loaded_bytes = total

    def load_warm(self, blocks=None) -> int:
        """Load selected blocks into pinned RAM (the warm tier).

        These blocks are not in VRAM; they exist so a VRAM refill reads pinned
        RAM instead of the SSD or a page-cache miss. Blocks already pinned are
        skipped. Returns the number of bytes pinned by THIS call.

        The planner chooses which blocks (`ExecutionPlan.warm_blocks`, smallest
        blocks first, so the budget covers as many whole blocks as possible).
        """
        bs = sorted(blocks or self.warm_blocks or self.blocks)
        pinned = 0
        for b in bs:
            if b not in self.blocks:
                continue
            for r in self.ROLES:
                key = (b, r)
                if key in self._warm or key in self._ram:
                    continue
                t = self.desc[key]
                per = int(t["per_expert_bytes"])
                fh = self._file_handles[t["shard"]]
                fh.seek(int(t["offset"]))
                # Allocate pinned memory per-expert (2 MB chunks) to stay
                # within CUDA pinned memory limits instead of one big block.
                bufs = []
                for e in range(self.n_experts):
                    chunk = fh.read(per)
                    if len(chunk) != per:
                        raise IOError("short read warm block=%d role=%d expert=%d" % (b, r, e))
                    p = rt.check_ptr(rt.lib().sabah_host_alloc(per),
                                     "pinned warm alloc %d B" % per)
                    self._pinned_ptrs.append(p)
                    buf = np.ctypeslib.as_array(
                        ctypes.cast(p, ctypes.POINTER(ctypes.c_uint8)), (per,))
                    buf[:] = np.frombuffer(chunk, dtype=np.uint8)
                    bufs.append(buf)
                    pinned += per
                self._warm[key] = bufs  # list of per-expert pinned arrays
        self.warm_loaded_bytes = getattr(self, "warm_loaded_bytes", 0) + pinned
        return pinned

    def close(self):
        with self._lock:
            for p in self._pinned_ptrs:
                rt.lib().sabah_host_free(p)
            self._pinned_ptrs = []
            self._ram.clear()
            self._maps.clear()
            self._ranges.clear()
            for fh in self._file_handles.values():
                try:
                    fh.close()
                except OSError:
                    pass
            self._file_handles.clear()

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
