"""
Sabah / runtime / residency

Splits one device's VRAM into the two things that want it, in the only order
that makes sense:

    1. FIXED PATH first.  Attention, SSM/GDN state, the router, the embedding
       and the output head are read EVERY token. A miss on any of them is a
       stall with no way to hide it, and their total is small (~4-5 GB). They
       are pinned once and never move.

    2. EXPERT HOT TIER gets everything left. Behind PCIe/SSD, the expert bank
       is what does not fit, so the only question worth asking is "how many
       experts fit after the fixed path is safe", and the answer is
       `vram_usable - fixed - kv/workspace`.

This module is the runtime side of that decision. The planner *predicts* it
(`ExecutionPlan.fixed_bytes`, `.expert_capacity_total`); this code *does* it, so
a standalone runner can pin the fixed path without re-deriving the arithmetic.

The n-gram lookup table is deliberately absent: it is a sparse gather and stays
on disk (see `ngram_prefetch`). Nothing here touches it.

The original GGUF is opened read-only and is not modified.
"""
from __future__ import annotations

import os
import ctypes
import threading
from typing import Optional, List

import numpy as np

from sabah.runtime import rt
from sabah.runtime.rt import SabahCudaError
from sabah.runtime.expert_bank import shard_paths


def expert_capacity_bytes(vram_usable: int, fixed_bytes: int,
                          kv_workspace_bytes: int) -> int:
    """VRAM left for experts after the fixed path and KV/workspace reserves."""
    return max(0, int(vram_usable) - int(fixed_bytes) - int(kv_workspace_bytes))


def apply_plan(plan_obj, bank) -> dict:
    """Apply an ExecutionPlan's residency decisions to a live expert bank.

    Pins two things:

      1. The FIXED PATH (attention, router, embedding, output head — read
         every token) into VRAM via `FixedPathResident`. This is a one-time
         H2D; after it, every token reads these tensors from VRAM at GPU
         memory bandwidth instead of re-reading them from storage. Without
         this the fixed path alone is a ~2 s/token storage tax and no
         acceleration is possible.
      2. The WARM tier (piece B): `plan.warm_blocks` pinned in RAM so VRAM
         refills for those blocks read pinned memory instead of the SSD.

    Returns a small dict for logging/testing.
    """
    from sabah.runtime.residency import FixedPathResident

    warm = list(getattr(plan_obj, "warm_blocks", []) or [])
    pinned = bank.load_warm(warm) if warm else 0
    fixed_resident = 0
    if getattr(plan_obj, "fixed_resident", False):
        fpr = FixedPathResident(bank.model)
        fixed_resident = fpr.load()
    return dict(warm_blocks=warm, warm_bytes=int(pinned),
                fixed_resident_bytes=int(fixed_resident))


class FixedPathResident:
    """Pins a model's fixed-path tensors into VRAM, in order, to a byte budget.

    Tensors are copied as raw quantised bytes (no dequantisation), so residency
    never changes a value. If the budget is smaller than the full fixed path the
    method copies as much as fits in the order given and reports what is left
    out; it does not silently drop a tensor that later code assumes is resident.
    """

    def __init__(self, model_profile, device: int = 0):
        self.profile = model_profile
        self.device = device
        self._ptrs = {}                # name -> device pointer
        self._maps = {}
        self._pinned = []
        self._lock = threading.RLock()
        self.resident_bytes = 0
        self.missing: List[str] = []

    # ------------------------------------------------------------------
    def tensors(self) -> list:
        """Fixed-path tensors, largest first (bigger pins buy more stall back)."""
        ts = [t for t in getattr(self.profile, "fixed_tensors", [])]
        if not ts:
            # No per-tensor table (older profile): fall back to nothing to pin.
            return []
        return sorted(ts, key=lambda t: -int(t["bytes"]))

    def plan(self, budget_bytes: Optional[int] = None) -> dict:
        """What would be pinned, and what would be left out, for a budget."""
        total = sum(int(t["bytes"]) for t in self.tensors())
        if budget_bytes is None:
            budget_bytes = total
        keep, taken = [], 0
        for t in self.tensors():
            if taken + int(t["bytes"]) <= budget_bytes:
                keep.append(t["name"])
                taken += int(t["bytes"])
        return dict(total_bytes=total, budget_bytes=int(budget_bytes),
                    pinned_bytes=taken, pinned=keep,
                    left_out=[t["name"] for t in self.tensors()
                              if t["name"] not in keep])

    # ------------------------------------------------------------------
    def load(self, budget_bytes: Optional[int] = None) -> int:
        """Copy the fixed path into VRAM. Returns bytes made resident.

        Uses a single reusable staging buffer (allocated via cudaHostAlloc)
        to avoid repeated host_register/unregister and alignment issues.
        """
        L = rt.lib()
        names = None if budget_bytes is None else set(self.plan(budget_bytes)["pinned"])
        paths = shard_paths(self.profile.path)

        # Single staging buffer: max tensor size, allocated once via cudaHostAlloc
        max_tensor_bytes = max(int(t["bytes"]) for t in self.tensors()) if self.tensors() else 0
        stage = None
        stage_ptr = None
        if max_tensor_bytes > 0:
            stage = rt.check_ptr(L.sabah_host_alloc(max_tensor_bytes),
                                 "fixed-path staging buffer")
            stage_ptr = ctypes.c_void_p(stage)
            stage_arr = np.ctypeslib.as_array(
                ctypes.cast(stage_ptr, ctypes.POINTER(ctypes.c_uint8)), (max_tensor_bytes,))

        with self._lock:
            for t in self.tensors():
                if names is not None and t["name"] not in names:
                    self.missing.append(t["name"])
                    continue
                si = int(t["shard"])
                if si not in self._maps:
                    self._maps[si] = np.memmap(paths[si], dtype=np.uint8, mode="r")
                off, nbytes = int(t["offset"]), int(t["bytes"])
                src = self._maps[si][off:off + nbytes]

                # Copy into pre-allocated staging buffer
                stage_arr[:nbytes] = src

                try:
                    dev = rt.check_ptr(L.sabah_dev_alloc(nbytes),
                                       "fixed-path VRAM %s" % t["name"])
                except SabahCudaError as e:
                    if "out of memory" in str(e).lower():
                        self.missing.append(t["name"])
                        continue
                    raise
                rt.check(L.sabah_memcpy_h2d(ctypes.c_void_p(dev),
                                            stage_ptr, nbytes),
                         "fixed-path h2d %s" % t["name"])
                self._ptrs[t["name"]] = dev
                self.resident_bytes += nbytes

            L.sabah_sync_all()

        # Cleanup staging buffer
        if stage_ptr is not None:
            L.sabah_host_free(stage_ptr)
        return self.resident_bytes

    def ptr(self, name: str) -> Optional[int]:
        return self._ptrs.get(name)

    def free(self):
        with self._lock:
            L = rt.lib()
            for p in self._ptrs.values():
                L.sabah_dev_free(ctypes.c_void_p(p))
            self._ptrs.clear()
            # Staging buffers were unregistered immediately after copy in load();
            # this loop is a safety net in case any remain.
            for s in self._pinned:
                rt.host_unregister(s)
            self._pinned = []
            self._maps.clear()
            self.resident_bytes = 0

    def describe(self) -> str:
        plan = self.plan()
        return ("fixed path : %d tensors, %.3f GB total\n"
                "resident   : %.3f GB in VRAM\n"
                "left out   : %d tensor(s)"
                % (len(self.tensors()), plan["total_bytes"] / 1e9,
                   self.resident_bytes / 1e9, len(self.missing)))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.free()
