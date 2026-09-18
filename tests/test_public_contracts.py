from __future__ import annotations

import json

import numpy as np
import pytest

from sabah.core.model_inspector import ModelProfile
from sabah.planner.planner import ExecClass, plan
from sabah.hardware.profile import GpuInfo, HardwareProfile
from sabah.runtime.identity import ExpertId, ExpertRange
from sabah.runtime.source import MemoryExpertSource
from sabah.runtime.hot_tier import Telemetry


def test_expert_identity_never_aliases_blocks():
    a = ExpertId(4, 17)
    b = ExpertId(5, 17)
    assert a != b
    assert a.as_tuple() == (4, 17)
    assert a < b


def test_expert_range_validates_exact_bounds():
    key = ExpertId(2, 17)
    r = ExpertRange(key, "gate", 1, 4096, 1024, "Q4_K")
    assert r.end == 5120
    with pytest.raises(ValueError):
        ExpertRange(key, "gate", 1, 0, 0, "Q4_K")


def test_memory_source_is_deterministic_and_read_only():
    key = ExpertId(0, 3)
    source = MemoryExpertSource({(key, "gate"): b"abcd"})
    view = source.read(key, "gate")
    assert bytes(view) == b"abcd"
    assert view.flags.writeable is False
    with pytest.raises(KeyError):
        source.read(ExpertId(1, 3), "gate")


def test_telemetry_exposes_byte_weighted_metrics():
    t = Telemetry()
    t.hits = 2
    t.misses = 1
    t.byte_hits = 20
    t.byte_misses = 10
    payload = t.as_dict()
    assert payload["hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert payload["byte_hit_rate"] == pytest.approx(2 / 3, abs=1e-6)
    assert json.loads(json.dumps(payload))["byte_misses"] == 10


def test_planner_never_claims_full_gpu_without_capacity():
    model = ModelProfile(architecture="qwen4exp", supported=True,
                         experts_contiguous=True, n_blocks=2, n_experts=4,
                         n_experts_used=2, expert_ff=64,
                         expert_bank_bytes=400, fixed_bytes=100,
                         lookup_bytes=20, per_token_expert_bytes=200,
                         per_block_expert_bytes={0: 50, 1: 50})
    hw = HardwareProfile(host="test", os="test", ram_total=1024,
                         ram_usable=1024, ram_bandwidth_gbps=20)
    hw.gpus = [GpuInfo(index=0, name="test", vram_total=256,
                       vram_free=256, vram_usable=120, h2d_solo_gbps=10)]
    hw.cuda_available = True
    hw.h2d_sum_solo_gbps = 10
    hw.h2d_simultaneous_gbps = 10
    assert plan(model, hw).exec_class != ExecClass.FULL_GPU


def test_storage_backed_plan_is_not_recommended_without_measurement():
    model = ModelProfile(architecture="qwen4exp", supported=True,
                         experts_contiguous=True, n_blocks=2, n_experts=4,
                         n_experts_used=2, expert_ff=64,
                         expert_bank_bytes=10_000, fixed_bytes=50,
                         lookup_bytes=20, per_token_expert_bytes=5_000,
                         per_block_expert_bytes={0: 1_250, 1: 1_250})
    hw = HardwareProfile(host="test", os="test", ram_total=2_000,
                         ram_usable=2_000, ram_bandwidth_gbps=20,
                         storage_read_gbps=1)
    hw.gpus = [GpuInfo(index=0, name="test", vram_total=1_000,
                       vram_free=1_000, vram_usable=900, h2d_solo_gbps=10)]
    hw.cuda_available = True
    hw.h2d_sum_solo_gbps = 10
    hw.h2d_simultaneous_gbps = 10
    result = plan(model, hw)
    assert result.exec_class.value == "STORAGE_BACKED"
    assert result.recommended is False
