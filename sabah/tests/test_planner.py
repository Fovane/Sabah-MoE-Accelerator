"""
Synthetic planner tests across machine classes.

The 3-GPU machine does not exist here, so multi-device behaviour is exercised
with mock HardwareProfiles. These are PLANNER tests: they check that the plan
is sane and safe, not that any throughput number is achievable.
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sabah.hardware.profile import HardwareProfile, GpuInfo
from sabah.planner.planner import plan, summarize, ExecClass
from sabah.core.model_inspector import ModelProfile

GB = 1 << 30


def flash_next_profile() -> ModelProfile:
    """The measured Qwen3.8-Flash-Next geometry, so planner tests do not
    require the 111 GB artifact to be present."""
    m = ModelProfile()
    m.path = "/models/qwen3.8-flash-next-Q4_K_XL.gguf"
    m.architecture = "qwen4exp"
    m.supported = True
    m.experts_contiguous = True
    m.n_blocks, m.d_model = 48, 2560
    m.n_experts, m.n_experts_used, m.expert_ff = 512, 10, 640
    m.expert_bank_bytes = 77_017_907_200
    m.fixed_bytes = 4_830_200_000
    m.lookup_bytes = 29_475_600_000
    m.per_token_expert_bytes = 1_504_256_000
    m.per_block_expert_bytes = {i: 3_072_000 for i in range(48)}
    return m


def mock_hw(name, n_gpu, vram_gb, ram_gb, h2d_solo, contention=1.0,
            ram_bw=60.0, storage=2.3):
    hw = HardwareProfile(host=name, os="mock")
    for i in range(n_gpu):
        total = int(vram_gb * GB)
        hw.gpus.append(GpuInfo(index=i, name="mock-gpu-%d" % i,
                               vram_total=total, vram_free=total,
                               vram_usable=int(total * 0.94) - (512 << 20),
                               h2d_solo_gbps=h2d_solo))
    hw.cuda_available = n_gpu > 0
    hw.h2d_sum_solo_gbps = h2d_solo * n_gpu
    hw.h2d_simultaneous_gbps = h2d_solo * n_gpu * contention
    hw.h2d_contention_factor = contention
    hw.ram_total = int(ram_gb * GB)
    hw.ram_usable = int(ram_gb * GB * 0.88) - (4 * GB)
    hw.ram_bandwidth_gbps = ram_bw
    hw.storage_read_gbps = storage
    return hw


CASES = [
    ("1 GPU 6GB / 28GB RAM (this dev box)", mock_hw("dev", 1, 6.44, 29.8, 12.8)),
    ("1 GPU 6GB / 128GB RAM", mock_hw("a", 1, 6.44, 128, 12.8)),
    ("1 GPU 24GB / 128GB RAM", mock_hw("b", 1, 24, 128, 12.8)),
    ("3 GPU 12GB / 128GB, PCIe scales", mock_hw("c", 3, 12, 128, 11.0, 1.0)),
    ("3 GPU 12GB / 128GB, PCIe CONTENDED", mock_hw("d", 3, 12, 128, 11.0, 0.55)),
    ("heterogeneous 24+12+12GB", None),
    ("4 GPU 24GB / 256GB RAM", mock_hw("f", 4, 24, 256, 20.0, 0.9)),
    ("1 GPU 8GB / 16GB RAM (storage backed)", mock_hw("g", 1, 8, 16, 12.8)),
    ("no GPU, 128GB RAM", mock_hw("h", 0, 0, 128, 0.0)),
]

# heterogeneous case needs hand-built devices
_het = mock_hw("e", 0, 0, 128, 0.0)
for i, gb in enumerate((24, 12, 12)):
    t = int(gb * GB)
    _het.gpus.append(GpuInfo(index=i, name="het-%d" % i, vram_total=t,
                             vram_free=t, vram_usable=int(t * 0.94) - (512 << 20),
                             h2d_solo_gbps=11.0))
_het.cuda_available = True
_het.h2d_sum_solo_gbps = 33.0
_het.h2d_simultaneous_gbps = 28.0
_het.h2d_contention_factor = 28.0 / 33.0
CASES[5] = ("heterogeneous 24+12+12GB", _het)


def main():
    m = flash_next_profile()
    print("=" * 100)
    print("SABAH PLANNER — SYNTHETIC MACHINE CLASSES (projections, not measurements)")
    print("=" * 100)
    print("%-38s %-15s %9s %9s %11s %s"
          % ("machine", "class", "expertGB", "hit", "PROJ tok/s", "rec"))
    print("-" * 100)
    failures = []
    for label, hw in CASES:
        p = plan(m, hw, context=8192, concurrency=1)
        print("%-38s %-15s %9.1f %9.4f %11s %s"
              % (label[:38], p.exec_class.value, p.expert_capacity_total / 1e9,
                 p.hit_rate,
                 "%.1f-%.1f" % (p.projected_tok_s_low, p.projected_tok_s_high),
                 "yes" if p.recommended else "NO"))

        # ---- invariants the planner must never violate ------------------
        for d in p.devices:
            if d.expert_capacity + d.fixed_bytes + d.kv_workspace_bytes > d.vram_usable + 1:
                failures.append("%s: GPU%d placement exceeds usable VRAM"
                                % (label, d.index))
        if p.expert_capacity_total > m.expert_bank_bytes:
            failures.append("%s: expert capacity exceeds the bank" % label)
        if p.hit_rate < 0 or p.hit_rate > 1:
            failures.append("%s: hit rate out of range" % label)
        if p.exec_class == ExecClass.FULL_GPU and p.hit_rate != 1.0:
            failures.append("%s: FULL_GPU must have hit rate 1.0" % label)
        if not hw.gpus and p.exec_class in (ExecClass.FULL_GPU, ExecClass.GPU_HOT_TIER):
            failures.append("%s: GPU class chosen with no GPU" % label)

    print("\n--- concurrency sweep, 3 GPU / PCIe scales ---")
    hw = CASES[3][1]
    print("%-6s %-15s %11s %11s" % ("B", "class", "PROJ tok/s", "per-stream"))
    for B in (1, 4, 8, 16):
        p = plan(m, hw, context=8192, concurrency=B)
        mid = 0.5 * (p.projected_tok_s_low + p.projected_tok_s_high)
        print("%-6d %-15s %11s %11.1f"
              % (B, p.exec_class.value,
                 "%.1f-%.1f" % (p.projected_tok_s_low, p.projected_tok_s_high),
                 mid / B))

    print("\n--- the contended-PCIe machine must be planned against the "
          "AGGREGATE figure ---")
    good, bad = CASES[3][1], CASES[4][1]
    pg, pb = plan(m, good, concurrency=1), plan(m, bad, concurrency=1)
    print("  scaling board : effective H2D %.1f GB/s -> %.1f-%.1f tok/s"
          % (good.effective_h2d_gbps, pg.projected_tok_s_low, pg.projected_tok_s_high))
    print("  contended     : effective H2D %.1f GB/s -> %.1f-%.1f tok/s"
          % (bad.effective_h2d_gbps, pb.projected_tok_s_low, pb.projected_tok_s_high))
    mid_g = 0.5 * (pg.projected_tok_s_low + pg.projected_tok_s_high)
    mid_b = 0.5 * (pb.projected_tok_s_low + pb.projected_tok_s_high)
    if mid_b >= mid_g * 0.95:
        failures.append("contended PCIe was not penalised (%.1f vs %.1f)"
                        % (mid_b, mid_g))

    print("\n" + "=" * 100)
    if failures:
        print("FAILURES (%d):" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("all planner invariants hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
