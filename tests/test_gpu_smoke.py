"""Model-free GPU smoke tests for the acceleration layers (pieces B, C, D).

These tests need a GPU and the built ``sabah_rt`` runtime, but NOT a GGUF model,
so they run on a machine whose quant mix Sabah cannot execute yet. They cover
exactly the native surface the acceleration work touched:

  * the device slab allocator (best-fit free list + coalescing) and its
    ``destroy`` entry point,
  * ``sabah_copy_wait_event`` (the precise copy<->compute dependency),
  * the HotTier: direct-DMA slot exactness, the prefetch reserve, prefetch
    telemetry, and that eviction never corrupts a requested expert.

They prove the transfer machinery is byte-exact on real hardware. They do NOT
prove matvec numerics; that is what ``tests/test_prefetch_equivalence.py`` and
``sabah/tests/test_quant_kernels.py`` do, and both need a runnable model.
"""
from __future__ import annotations

import os
import ctypes

import numpy as np
import pytest

from sabah.runtime import rt
from sabah.runtime.hot_tier import HotTier

MIB = 1 << 20
ROLE_BYTES = 256
N_SLOTS = 8          # capacity / slot_bytes, chosen by the tier's own maths


def _cuda_ready() -> bool:
    return rt.available() and rt.device_count() >= 1


needs_gpu = pytest.mark.skipif(
    not _cuda_ready(), reason="CUDA runtime or GPU not available")


@pytest.fixture(scope="module", autouse=True)
def _runtime():
    if not _cuda_ready():
        yield False
        return
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")
    yield True
    rt.check(rt.lib().sabah_rt_shutdown(), "rt_shutdown")


# ---------------------------------------------------------------------------
# device slab allocator
# ---------------------------------------------------------------------------
@needs_gpu
def test_slab_best_fit_reuses_a_freed_range():
    """The second 1 MiB request must land back in the 1 MiB hole, not the tail."""
    a = rt.slab_alloc(MIB)
    b = rt.slab_alloc(MIB)
    c = rt.slab_alloc(MIB)
    rt.slab_free(b)
    d = rt.slab_alloc(MIB)
    try:
        # best fit picks the smallest range that fits -- the freed MIB -- so the
        # allocator returns the exact same address rather than carving the tail.
        assert d == b, "best-fit did not reuse the freed 1 MiB range"
        assert d not in (a, c)
    finally:
        rt.slab_free(a)
        rt.slab_free(c)
        rt.slab_free(d)


@needs_gpu
def test_slab_coalesces_adjacent_frees():
    """Three adjacent 1 MiB frees must merge into one 3 MiB range at offset 0."""
    a = rt.slab_alloc(MIB)
    b = rt.slab_alloc(MIB)
    c = rt.slab_alloc(MIB)
    rt.slab_free(a)
    rt.slab_free(b)
    rt.slab_free(c)
    big = rt.slab_alloc(3 * MIB)
    try:
        # If the frees had not coalesced, no 3 MiB *hole* would exist at offset 0
        # and best-fit would hand back a slice of the untouched tail instead.
        assert big == a, "adjacent frees did not coalesce back into one range"
    finally:
        rt.slab_free(big)


@needs_gpu
def test_slab_returns_null_when_no_range_fits():
    """An impossible request must return NULL, never over-commit the pool."""
    p = rt.lib().sabah_slab_alloc(ctypes.c_size_t(1 << 62))
    assert not p, "slab over-committed a request larger than the whole pool"


@needs_gpu
def test_slab_destroy_and_reinit():
    """destroy() must release the pool so a fresh init() can size it again."""
    cap = int(os.environ.get("SABAH_LLAMA_HOT_BYTES") or 0) or (2 << 30)
    rt.lib().sabah_slab_destroy()
    try:
        rt.slab_init(4 * MIB)
        p = rt.slab_alloc(MIB)
        assert p, "alloc after re-init returned NULL"
        rt.slab_free(p)
        # a 4 MiB pool cannot honour 8 MiB: proves the new size took effect
        assert not rt.lib().sabah_slab_alloc(ctypes.c_size_t(8 * MIB))
    finally:
        rt.lib().sabah_slab_destroy()
        rt.slab_init(cap)          # restore what sabah_rt_init() had set up


# ---------------------------------------------------------------------------
# copy-stream / compute-stream dependency
# ---------------------------------------------------------------------------
@needs_gpu
def test_copy_wait_event_contract():
    """A recorded compute event must be a legal wait for the copy stream."""
    n = 4096
    src = np.arange(n, dtype=np.uint8)
    d = rt.check_ptr(rt.lib().sabah_dev_alloc(ctypes.c_size_t(n)), "dev alloc")
    host = rt.check_ptr(rt.lib().sabah_host_alloc(ctypes.c_size_t(n)), "host alloc")
    ev = rt.check_ptr(rt.lib().sabah_event_create(), "event create")
    try:
        rt.check(rt.lib().sabah_event_record_compute(ctypes.c_void_p(ev)),
                 "record compute event")
        rt.check(rt.lib().sabah_copy_wait_event(ctypes.c_void_p(ev)),
                 "copy wait event")
        rt.check(rt.lib().sabah_h2d_async(ctypes.c_void_p(d),
                                          ctypes.c_void_p(src.ctypes.data), n),
                 "h2d async")
        rt.check(rt.lib().sabah_sync_all(), "sync all")
        rt.check(rt.lib().sabah_memcpy_d2h(ctypes.c_void_p(host),
                                           ctypes.c_void_p(d), n), "d2h")
        got = np.ctypeslib.as_array(
            ctypes.cast(host, ctypes.POINTER(ctypes.c_uint8)), (n,))
        assert np.array_equal(got, src), "bytes moved after a copy_wait_event differ"
    finally:
        rt.lib().sabah_event_destroy(ctypes.c_void_p(ev))
        rt.lib().sabah_host_free(ctypes.c_void_p(host))
        rt.lib().sabah_dev_free(ctypes.c_void_p(d))


# ---------------------------------------------------------------------------
# HotTier: exactness, prefetch reserve, telemetry, eviction
# ---------------------------------------------------------------------------
class FakeBank:
    """A RAM-resident, pinned, synthetic expert bank.

    Same shape as the real ExpertBank for everything HotTier touches, but the
    expert bytes are a deterministic pattern instead of a quantized tensor.
    """

    mode = "ram"
    pinned = True
    ROLES = ("gate", "up", "down")

    def __init__(self, n_experts: int = 64, role_bytes: int = ROLE_BYTES):
        self.role_bytes = role_bytes
        self.n_experts = n_experts
        self.blocks = [0]
        self._bufs = {}

    def expert_bytes(self, block, role):
        return self.role_bytes

    def block_expert_bytes(self, block):
        return self.role_bytes * len(self.ROLES)

    def pinned_block(self, block):
        return True

    def pattern(self, block, expert, role):
        base = (expert * 7 + self.ROLES.index(role) * 29 + block * 3) & 0xFF
        return ((np.arange(self.role_bytes, dtype=np.int32) + base) & 0xFF
                ).astype(np.uint8)

    def slice(self, block, expert, role):
        key = (block, expert, role)
        v = self._bufs.get(key)
        if v is None:
            ptr = rt.check_ptr(rt.lib().sabah_host_alloc(
                ctypes.c_size_t(self.role_bytes)), "fake pinned expert")
            arr = np.ctypeslib.as_array(
                ctypes.cast(ptr, ctypes.POINTER(ctypes.c_uint8)),
                (self.role_bytes,))
            arr[:] = self.pattern(block, expert, role)
            v = (ptr, arr)
            self._bufs[key] = v
        return v[1]

    def expected(self, block, expert):
        return b"".join(bytes(self.pattern(block, expert, r)) for r in self.ROLES)

    def free(self):
        for ptr, _ in self._bufs.values():
            rt.lib().sabah_host_free(ctypes.c_void_p(ptr))
        self._bufs.clear()


def _make_tier(prefetch_slots: int = 0, n_experts: int = 64):
    bank = FakeBank(n_experts=n_experts)
    slot_bytes = bank.block_expert_bytes(0)
    tier = HotTier(bank, slot_bytes * N_SLOTS, blocks=[0],
                   prefetch_slots=prefetch_slots)
    assert tier.slots_per_block == N_SLOTS, tier.slots_per_block
    return bank, tier


def _slot_bytes(tier, block, slot) -> bytes:
    p = tier.pools[block]
    n = p.slot_bytes
    host = rt.check_ptr(rt.lib().sabah_host_alloc(ctypes.c_size_t(n)),
                        "verify host buf")
    try:
        rt.check(rt.lib().sabah_memcpy_d2h(
            ctypes.c_void_p(host), ctypes.c_void_p(p.base + slot * n), n),
            "verify d2h")
        arr = np.ctypeslib.as_array(
            ctypes.cast(host, ctypes.POINTER(ctypes.c_uint8)), (n,))
        return bytes(arr)
    finally:
        rt.lib().sabah_host_free(ctypes.c_void_p(host))


def _assert_exact(tier, bank, block, want, slots):
    rt.check(rt.lib().sabah_sync_all(), "sync before verify")
    for e, slot in zip(want, slots):
        got = _slot_bytes(tier, block, slot)
        assert got == bank.expected(block, e), \
            "expert %d differs from its source bytes" % e


@needs_gpu
def test_hot_tier_direct_dma_slots_are_exact():
    """A pinned RAM bank must DMA in place and land byte-exact in each slot."""
    bank, tier = _make_tier()
    try:
        want = [3, 11, 40, 7]
        slots = tier.ensure(0, want)
        assert len(set(slots)) == len(want), "two experts share one slot"
        _assert_exact(tier, bank, 0, want, slots)
        assert tier.tel.misses == len(want)
        assert len(tier.resident(0)) == len(want)
    finally:
        tier.free()
        bank.free()


@needs_gpu
def test_hot_tier_prefetch_fills_the_reserve_and_is_counted():
    """Prefetch must only use the reserve, and a later hit must be counted."""
    bank, tier = _make_tier(prefetch_slots=2)
    try:
        tier.ensure(0, [0, 1, 2, 3, 4, 5])
        # 6 resident, 2 slots deliberately kept free for prefetch
        assert len(tier.pools[0].free_slots) == 2

        issued = tier.prefetch(0, [20, 21, 22])   # 22 has nowhere to go
        assert issued == 2, "prefetch used more slots than the reserve"
        assert tier.tel.prefetch_issued == 2

        slots = tier.ensure(0, [20, 21])
        assert tier.tel.prefetch_hits == 2, "prefetched experts were not hits"
        _assert_exact(tier, bank, 0, [20, 21], slots)
    finally:
        tier.free()
        bank.free()


@needs_gpu
def test_hot_tier_prefetch_never_evicts_a_live_slot():
    """With no free slot left, prefetch must decline rather than evict."""
    bank, tier = _make_tier(prefetch_slots=2)
    try:
        tier.ensure(0, [0, 1, 2, 3, 4, 5])   # leaves exactly the reserve free
        resident_before = set(tier.resident(0))
        # fill the reserve
        assert tier.prefetch(0, [20, 21]) == 2
        # pool is now full: nothing may be evicted to satisfy advice
        assert tier.prefetch(0, [30, 31]) == 0
        assert resident_before <= set(tier.resident(0)), \
            "prefetch evicted live work"
    finally:
        tier.free()
        bank.free()


@needs_gpu
def test_hot_tier_eviction_keeps_requested_experts_exact():
    """Evicting to make room must never corrupt the experts actually requested."""
    bank, tier = _make_tier(prefetch_slots=2)
    try:
        tier.ensure(0, [0, 1, 2, 3, 4, 5])
        tier.prefetch(0, [20, 21])
        rt.check(rt.lib().sabah_sync_all(), "sync")

        # the pool is full of {0..5,20,21}; asking for {0..7} must evict the two
        # prefetched-but-unrequested experts and keep every requested one exact
        want = list(range(8))
        slots = tier.ensure(0, want)
        _assert_exact(tier, bank, 0, want, slots)
        assert 20 not in tier.resident(0) and 21 not in tier.resident(0)
    finally:
        tier.free()
        bank.free()
