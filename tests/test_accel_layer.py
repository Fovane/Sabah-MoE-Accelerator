"""Tests for the acceleration layers that do NOT need a GPU.

Covers the parts whose correctness is pure logic, so they run on any machine:

  * the n-gram prefetch layer (row addressing, gather, best-effort page hints,
    index function, the streaming driver)
  * the warm-tier block selection and the planner's fixed/warm split
  * the expert prefetcher's advice contract, against a fake tier
  * the fixed-path / expert-capacity arithmetic

The GPU-side equivalence (prefetch changes timing, never values) lives in
`tests/test_prefetch_equivalence.py` and is skipped when CUDA is unavailable.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sabah.core.model_inspector import ModelProfile
from sabah.hardware.profile import HardwareProfile, GpuInfo
from sabah.planner.planner import ExecClass, _select_warm_blocks, plan
from sabah.runtime.mtp_prefetch import (
    ExpertPrefetcher,
    ExternalPredictor,
    FrequencyPredictor,
    NullPredictor,
)
from sabah.runtime.ngram_prefetch import (
    NgramPrefetcher,
    NgramTable,
    rolling_hash_indices,
    row_shape,
)
from sabah.runtime.residency import expert_capacity_bytes

GB = 1 << 30


# ---------------------------------------------------------------------------
# n-gram prefetch layer
# ---------------------------------------------------------------------------
def _ngram_fixture(tmp_path, n_rows=8, row_bytes=32, pad=16):
    """A tiny shard with one n-gram lookup tensor at a non-zero offset."""
    raw = bytes(((i * 7) + 3) & 0xFF for i in range(n_rows * row_bytes))
    f = tmp_path / "model-00001-of-00001.gguf"
    f.write_bytes(b"\xAA" * pad + raw)
    prof = ModelProfile()
    prof.path = str(f)
    prof.lookup_tensors = [dict(name="blk.1.per_layer_token_embd.weight", shard=0,
                                offset=pad, bytes=n_rows * row_bytes,
                                qtype="IQ4_NL", shape=[row_bytes, n_rows])]
    return f, raw, prof


def test_row_shape_is_bytes_over_rows():
    n, elems, rb = row_shape(dict(shape=[640, 1000], bytes=1000 * 88))
    assert (n, elems, rb) == (1000, 640, 88)


def test_ngram_row_and_gather_are_exact_and_offset_aware(tmp_path):
    f, raw, prof = _ngram_fixture(tmp_path)
    t = NgramTable(prof)
    try:
        assert t.n_rows == 8
        assert t.row_bytes == 32
        # the tensor starts at `pad`, so row 3 is NOT bytes[96:128]
        assert bytes(t.row(3)) == raw[3 * 32:4 * 32]
        out = t.gather([0, 5, 2])
        assert out.shape == (3, 32)
        assert bytes(out[0]) == raw[0:32]
        assert bytes(out[1]) == raw[5 * 32:6 * 32]
        assert bytes(out[2]) == raw[2 * 32:3 * 32]
        with pytest.raises(IndexError):
            t.row(99)
    finally:
        t.close()


def test_ngram_rows_are_read_only(tmp_path):
    f, raw, prof = _ngram_fixture(tmp_path)
    t = NgramTable(prof)
    try:
        assert not t.row(1).flags.writeable
    finally:
        t.close()


def test_ngram_page_hints_are_best_effort_never_raise(tmp_path):
    f, raw, prof = _ngram_fixture(tmp_path)
    t = NgramTable(prof)
    try:
        a = t.prefetch([1, 2, 3])
        b = t.prefetch([0, 7])          # non-contiguous -> several hints
        c = t.release([1, 2, 3])
        assert isinstance(a, int) and a >= 0
        assert isinstance(b, int) and b >= 0
        assert isinstance(c, int) and c >= 0
        # empty input is a no-op, not an error
        assert t.prefetch([]) == 0
        assert t.release([]) == 0
    finally:
        t.close()


def test_runs_are_page_aligned_and_clamped():
    """A hint must cover the needed pages and must not run past the mapping."""
    from sabah.runtime.ngram_prefetch import _PAGE, _runs

    base = 0x10000                                  # page aligned by construction
    limit = base + 3 * _PAGE
    assert _runs([0, 1, 2], _PAGE, base, limit) == [(base, 3 * _PAGE)]
    # the last row stops at the limit instead of rounding past it
    assert _runs([2], _PAGE, base, limit) == [(base + 2 * _PAGE, _PAGE)]
    # a row entirely past the mapping is dropped
    assert _runs([5], _PAGE, base, limit) == []
    # non-contiguous rows stay separate hints
    assert len(_runs([0, 4], 64, base, base + 64 * 8)) == 2
    # no rows, no hints
    assert _runs([], _PAGE, base, limit) == []


def test_rolling_hash_indices_stay_in_range():
    for rows in (1, 8, 4096, 351_000):
        idx = rolling_hash_indices([1, 2, 3, 4, 5, 6], table_rows=rows, n=4)
        assert len(idx) == 6
        assert all(0 <= i < rows for i in idx)
    assert rolling_hash_indices([1, 2], table_rows=0) == []


def test_prefetcher_step_drives_the_table(tmp_path):
    f, raw, prof = _ngram_fixture(tmp_path)
    t = NgramTable(prof)
    pf = NgramPrefetcher(t, n=4)
    try:
        a = pf.step([10, 11, 12])
        b = pf.step([13, 14])
        assert len(a) == 3 and len(b) == 2
        assert all(0 <= i < t.n_rows for i in a + b)
        st = pf.stats()
        assert st["steps"] == 2 and st["tokens"] == 5 and st["rows"] == 8
        assert st["table_gb"] > 0
    finally:
        pf.close()


def test_prefetcher_accepts_an_injected_index_function(tmp_path):
    f, raw, prof = _ngram_fixture(tmp_path)
    t = NgramTable(prof)
    pf = NgramPrefetcher(t, index_fn=lambda toks: [0, 7])
    try:
        assert pf.step([10]) == [0, 7]
        assert pf.step([11]) == [0, 7]
    finally:
        pf.close()


# ---------------------------------------------------------------------------
# warm tier selection + planner split
# ---------------------------------------------------------------------------
def _flash_next_profile() -> ModelProfile:
    m = ModelProfile()
    m.path = "/models/qwen3.8-flash-next.gguf"
    m.architecture = "qwen4exp"
    m.supported = True
    m.experts_contiguous = True
    m.n_blocks, m.d_model = 48, 2560
    m.n_experts, m.n_experts_used, m.expert_ff = 512, 10, 640
    m.expert_bank_bytes = 50_000_000_000
    m.fixed_bytes = 4_830_200_000
    m.lookup_bytes = 28_800_000_000
    m.per_token_expert_bytes = 10 * 2_000_000
    m.per_block_expert_bytes = {i: 2_000_000 for i in range(48)}
    return m


def test_select_warm_blocks_prefers_smallest_first():
    m = ModelProfile()
    m.n_experts = 4
    m.per_block_expert_bytes = {0: 100, 1: 50, 2: 200}
    # block bytes are per_expert * n_experts: 400, 200, 800
    assert _select_warm_blocks(m, 0) == []
    assert _select_warm_blocks(m, 199) == []
    assert _select_warm_blocks(m, 200) == [1]
    assert _select_warm_blocks(m, 600) == [0, 1]
    assert _select_warm_blocks(m, 2000) == [0, 1, 2]


def test_select_warm_blocks_is_empty_without_geometry():
    m = ModelProfile()
    m.n_experts = 4
    m.per_block_expert_bytes = {}
    assert _select_warm_blocks(m, 10 ** 12) == []


def test_planner_reserves_the_fixed_path_and_hands_the_rest_to_experts():
    m = _flash_next_profile()
    hw = HardwareProfile(host="dev", os="mock")
    total = int(24 * GB)
    hw.gpus.append(GpuInfo(index=0, name="mock", vram_total=total,
                           vram_free=total, vram_usable=int(total * 0.94) - (512 << 20),
                           h2d_solo_gbps=12.8))
    hw.cuda_available = True
    hw.h2d_sum_solo_gbps = 12.8
    hw.h2d_simultaneous_gbps = 12.8
    hw.h2d_contention_factor = 1.0
    hw.ram_total = int(16 * GB)
    hw.ram_usable = int(16 * GB * 0.88) - (4 * GB)
    hw.ram_bandwidth_gbps = 60.0
    hw.storage_read_gbps = 2.3

    p = plan(m, hw, context=8192, concurrency=1)
    # the fixed path is always marked resident, and it is charged to VRAM before
    # experts get anything
    assert p.fixed_resident is True
    assert p.fixed_bytes == m.fixed_bytes
    for d in p.devices:
        assert d.fixed_bytes + d.expert_capacity + d.kv_workspace_bytes <= d.vram_usable + 1
    # the bank does not fit in VRAM, so experts are a hot tier, not resident
    assert p.exec_class != ExecClass.FULL_GPU
    assert p.expert_capacity_total < m.expert_bank_bytes
    # and it does not fit in RAM either: warm tier is a subset of blocks, not all
    assert p.warm_bytes > 0
    assert 0 < len(p.warm_blocks) < m.n_blocks
    # every warm block must be a real block and the set well formed
    assert set(p.warm_blocks) <= set(range(m.n_blocks))
    assert p.warm_blocks == sorted(p.warm_blocks)
    budget = sum(m.per_block_expert_bytes[b] * m.n_experts for b in p.warm_blocks)
    assert budget <= p.warm_bytes


def test_planner_does_not_warm_when_the_whole_bank_fits_in_ram():
    m = _flash_next_profile()
    hw = HardwareProfile(host="big", os="mock")
    hw.gpus.append(GpuInfo(index=0, name="mock", vram_total=int(24 * GB),
                           vram_free=int(24 * GB),
                           vram_usable=int(24 * GB * 0.94) - (512 << 20),
                           h2d_solo_gbps=12.8))
    hw.cuda_available = True
    hw.h2d_sum_solo_gbps = 12.8
    hw.h2d_simultaneous_gbps = 12.8
    hw.h2d_contention_factor = 1.0
    hw.ram_total = int(128 * GB)
    hw.ram_usable = int(128 * GB * 0.88) - (4 * GB)
    hw.ram_bandwidth_gbps = 60.0
    hw.storage_read_gbps = 2.3

    p = plan(m, hw, context=8192, concurrency=1)
    assert p.ram_bank_resident is True
    # nothing to warm: the bank is already resident as a whole
    assert p.warm_bytes == 0
    assert p.warm_blocks == []


def test_planner_never_lets_capacity_exceed_the_bank():
    m = _flash_next_profile()
    hw = HardwareProfile(host="huge", os="mock")
    hw.gpus.append(GpuInfo(index=0, name="mock", vram_total=int(512 * GB),
                           vram_free=int(512 * GB),
                           vram_usable=int(512 * GB),
                           h2d_solo_gbps=64.0))
    hw.cuda_available = True
    hw.h2d_sum_solo_gbps = 64.0
    hw.h2d_simultaneous_gbps = 64.0
    hw.h2d_contention_factor = 1.0
    hw.ram_total = int(512 * GB)
    hw.ram_usable = int(512 * GB)
    hw.ram_bandwidth_gbps = 200.0
    hw.storage_read_gbps = 7.0

    p = plan(m, hw, context=8192, concurrency=1)
    assert p.expert_capacity_total <= m.expert_bank_bytes
    assert p.exec_class == ExecClass.FULL_GPU
    assert p.hit_rate == 1.0


# ---------------------------------------------------------------------------
# fixed-path / expert-capacity arithmetic
# ---------------------------------------------------------------------------
def test_expert_capacity_never_goes_negative():
    assert expert_capacity_bytes(1000, 400, 100) == 500
    assert expert_capacity_bytes(1000, 1000, 0) == 0
    assert expert_capacity_bytes(1000, 2000, 500) == 0
    assert expert_capacity_bytes(0, 0, 0) == 0


def test_apply_plan_pins_exactly_the_planners_warm_blocks():
    from sabah.runtime.residency import apply_plan

    seen = []

    class FakeBank:
        def load_warm(self, blocks):
            seen.append(list(blocks))
            return 42 * len(blocks)

    class FakePlan:
        warm_blocks = [3, 7]

    out = apply_plan(FakePlan(), FakeBank())
    assert seen == [[3, 7]]
    assert out == {"warm_blocks": [3, 7], "warm_bytes": 84,
                   "fixed_resident_bytes": 0}


def test_apply_plan_is_a_noop_without_warm_blocks():
    from sabah.runtime.residency import apply_plan

    class FakeBank:
        def load_warm(self, blocks):          # must never be called
            raise AssertionError("load_warm called with no warm blocks")

    class FakePlan:
        warm_blocks = []

    assert apply_plan(FakePlan(), FakeBank()) == {"warm_blocks": [],
                                                  "warm_bytes": 0,
                                                  "fixed_resident_bytes": 0}


def test_pinned_block_is_true_only_for_a_fully_pinned_block():
    """`pinned_block` gates the no-staging DMA path, so a partial warm block
    must NOT be treated as pinned (the missing roles would be read from mmap)."""
    from sabah.runtime.expert_bank import ExpertBank

    b = object.__new__(ExpertBank)
    b.ROLES = ("gate", "up", "down")
    b.mode = "mmap"
    b._ram = {}
    b._warm = {}

    assert b.pinned_block(0) is False
    b._warm = {(0, "gate"): 1, (0, "up"): 1}
    assert b.pinned_block(0) is False            # down is still on disk
    b._warm[(0, "down")] = 1
    assert b.pinned_block(0) is True
    assert b.pinned_block(1) is False

    b.mode = "ram"
    b._ram = {}
    assert b.pinned_block(0) is False
    b._ram = {(0, r): 1 for r in b.ROLES}
    assert b.pinned_block(0) is True


# ---------------------------------------------------------------------------
# the prefetcher's advice contract (against a fake tier, no GPU)
# ---------------------------------------------------------------------------
class FakeTier:
    def __init__(self, n_experts=64):
        self.n_experts = n_experts
        self.calls = []
        self.tel = type("T", (), {"prefetch_issued": 0, "prefetch_hits": 0})()

    def prefetch(self, block, experts):
        experts = [int(e) for e in experts if 0 <= int(e) < self.n_experts]
        self.calls.append((block, list(experts)))
        self.tel.prefetch_issued += len(experts)
        return len(experts)


def test_null_predictor_is_a_true_baseline():
    tier = FakeTier()
    pf = ExpertPrefetcher(tier, NullPredictor())
    pf.observe(0, [1, 2, 3])
    assert pf.warm(0) == 0
    assert tier.calls == []
    assert pf.stats()["predictor"] == "NullPredictor"


def test_disabled_prefetcher_never_touches_the_tier():
    tier = FakeTier()
    pf = ExpertPrefetcher(tier, FrequencyPredictor(top_k=2), enabled=False)
    pf.observe(0, [1, 2, 3, 4])
    assert pf.warm(0) == 0
    assert tier.calls == []


def test_frequency_predictor_returns_the_recent_hot_set():
    tier = FakeTier()
    pred = FrequencyPredictor(top_k=3)
    for _ in range(5):
        pred.observe(0, [7, 7, 8])
    pred.observe(0, [8])
    pred.observe(0, [9])
    cand = pred.candidates(0)
    assert len(cand) == 3
    # 7 seen 10x, 8 seen 6x, 9 seen 1x
    assert cand[0] == 7 and cand[1] == 8
    assert set(cand) == {7, 8, 9}


def test_frequency_predictor_is_per_block():
    pred = FrequencyPredictor(top_k=2)
    pred.observe(0, [5, 5, 5])
    pred.observe(1, [6, 6, 6])
    assert pred.candidates(0) == [5]
    assert pred.candidates(1) == [6]
    assert pred.candidates(2) == []
    pred.reset()
    assert pred.candidates(0) == []


def test_prefetcher_issues_only_the_predictor_candidates():
    tier = FakeTier()
    pred = FrequencyPredictor(top_k=2, blocks=[0])
    pf = ExpertPrefetcher(tier, pred)
    for _ in range(3):
        pf.observe(0, [11, 12, 13])
    pf.warm_ahead(0)
    assert tier.calls == [(0, pred.candidates(0))]
    assert pf.stats()["issued"] == 2
    assert pf.stats()["prefetch_issued"] == 2


def test_external_predictor_is_the_mtp_hook_and_is_capped():
    tier = FakeTier()
    seen = {}

    def mtp(block, ctx):
        seen["ctx"] = ctx
        return [1, 2, 3, 4, 5]

    pf = ExpertPrefetcher(tier, ExternalPredictor(mtp, top_k=3))
    n = pf.warm(2, context="hidden")
    assert seen["ctx"] == "hidden"
    assert n == 3
    assert tier.calls == [(2, [1, 2, 3])]


def test_external_predictor_tolerates_a_none_result():
    tier = FakeTier()
    pf = ExpertPrefetcher(tier, ExternalPredictor(lambda b, c: None))
    assert pf.warm(0) == 0
    assert tier.calls == []
