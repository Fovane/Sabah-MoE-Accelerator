"""
Sabah / planner

Takes a ModelProfile and a HardwareProfile and produces an ExecutionPlan.

Design rules this module enforces:

* **Aggregate VRAM is not unified memory.** Every placement must fit each
  physical device separately. Capacity is computed per GPU and only then summed.
* **No perfect cross-layer prefetch.** Block L's router consumes block L-1's
  output, so exact expert IDs for L are not known early. Only intra-block
  transfer/compute overlap is modelled.
* **The planner is allowed to say "no speedup".** If the predicted plan is not
  faster than the reference path, it says so and recommends REFERENCE.
* Estimates are ranges with a stated confidence, never single fabricated
  numbers, and they are labelled PROJECTED until a benchmark replaces them.
"""
from __future__ import annotations

import os
import json
import math
from dataclasses import dataclass, field, asdict
from enum import Enum

PLAN_VERSION = 1


class ExecClass(str, Enum):
    FULL_GPU = "FULL_GPU"              # bank fits in VRAM
    GPU_HOT_TIER = "GPU_HOT_TIER"      # bank in RAM, hot experts in VRAM
    CPU_EXPERT = "CPU_EXPERT"          # experts execute on CPU from RAM
    STORAGE_BACKED = "STORAGE_BACKED"  # bank does not fit RAM
    REFERENCE = "REFERENCE"            # Sabah cannot help; use stock runtime


# ---------------------------------------------------------------------------
# Measured hit-rate calibration.
#
# Provenance: Sabah v4 Q4, 13-family family-interleaved routing trace of
# Qwen3.8-Flash-Next UD-Q4_K_XL, 18,696 analysed tokens / 48 prompts /
# 12 families, source-disjoint prompts, all 48 blocks, Top-10 verified.
# Static popularity-ordered residency, byte-weighted hit rate.
#
# This is MEASURED for this architecture on a mixed workload. It is not a
# universal constant and must not be applied to another architecture.
# ---------------------------------------------------------------------------
HIT_CURVES = {
    "qwen4exp": {
        "provenance": "sabah v4 Q4, 18696 tokens, 12 families, mixed workload; "
                      "LRU curve re-measured in v5 on the same trace",

        # ------------------------------------------------------------------
        # `lru` is the curve the RUNTIME actually produces: seeded from a
        # disjoint 20% warm-up window, then plain LRU, byte-weighted, 48
        # blocks, capacity split evenly into per-block slots. Measured on
        # 15,168 unseen tokens (sabah/tools/calib_check.py).
        #
        # `static_deployable` is the same experiment with a FIXED hot set
        # placed once by warm-up popularity and never moved. It is the floor,
        # and it is what v4's `mixed` curve measured.
        #
        # LRU beats an ORACLE static placement by ~0.13-0.19 across the range.
        # Planning against the static curve therefore under-predicts the
        # runtime and refuses machines it could serve, so the planner reports
        # both: the LRU value as the operating point, the static value as the
        # conservative floor.
        # ------------------------------------------------------------------
        "lru": [(2, 0.2444), (4, 0.4458), (6, 0.5530), (8, 0.6206),
                (12, 0.7157), (16, 0.7809), (20, 0.8296), (24, 0.8660),
                (28, 0.8925), (32, 0.9117), (36, 0.9270), (40, 0.9385),
                (48, 0.9552), (56, 0.9668), (64, 0.9756), (72, 0.9832),
                (77, 1.0)],
        "static_deployable": [(2, 0.1092), (4, 0.1859), (6, 0.2621),
                              (8, 0.3207), (12, 0.4246), (16, 0.4998),
                              (20, 0.5627), (24, 0.6149), (28, 0.6601),
                              (32, 0.6976), (36, 0.7331), (40, 0.7640),
                              (48, 0.8184), (56, 0.8659), (64, 0.9043),
                              (72, 0.9366), (77, 1.0)],

        # v4's original static-oracle curve, kept so older plans stay readable
        "mixed": [(8, 0.4256), (12, 0.5372), (16, 0.6210), (17, 0.6389),
                  (20, 0.6869), (24, 0.7407), (28, 0.7864), (32, 0.8252),
                  (36, 0.8585), (40, 0.8871), (48, 0.9331), (60, 0.9778)],
        # per-family medians at the same capacities are markedly better; a
        # workload profile is worth roughly +18 points at 17 GB
        "specialised_bonus": 0.18,
        # cross-request weight reuse R_B, measured
        "reuse_mixed": {1: 1.0, 2: 1.019, 4: 1.057, 8: 1.134, 16: 1.307, 32: 1.651},
        "reuse_same_domain": {1: 1.0, 2: 1.071, 4: 1.193, 8: 1.401,
                              16: 1.762, 32: 2.385},
        # static hot set learned on past traffic loses this much vs an oracle
        # hot set on unseen source-disjoint prompts
        "generalisation_gap": 0.0477,
    }
}

# CPU expert-kernel efficiency: measured 32.16 GB/s effective against ~60 GB/s
# raw RAM bandwidth on the v3/v4 development machine (Q4_K/Q5_1 mul_mat_id).
# Used only until `sabah bench --cpu-expert` measures the actual machine.
CPU_EXPERT_BW_FRACTION = 0.54
# Sustained GPU read efficiency for the expert kernel, as a fraction of the
# device's nominal bandwidth. 145 GB/s achieved on a 192 GB/s part.
GPU_BW_EFFICIENCY = 0.75
# Runtime overhead not captured by the byte model (graph, launch, sync).
OVERHEAD_FRAC = 0.12


@dataclass
class DevicePlan:
    index: int = 0
    name: str = ""
    vram_usable: int = 0
    fixed_bytes: int = 0
    kv_workspace_bytes: int = 0
    expert_capacity: int = 0
    n_experts_resident: int = 0


@dataclass
class ExecutionPlan:
    version: int = PLAN_VERSION
    exec_class: str = ExecClass.REFERENCE
    model: str = ""
    architecture: str = ""

    devices: list = field(default_factory=list)
    expert_capacity_total: int = 0
    expert_capacity_frac_of_bank: float = 0.0
    fixed_replicated: bool = True

    ram_bank_resident: bool = False
    ram_bank_bytes: int = 0

    hit_rate: float = 0.0
    hit_rate_floor: float = 0.0       # same capacity under static placement
    hit_rate_source: str = ""
    miss_bytes_per_token: int = 0

    t_fixed_ms: float = 0.0
    t_expert_compute_ms: float = 0.0
    t_expert_fetch_ms: float = 0.0
    t_token_ms: float = 0.0

    projected_tok_s_low: float = 0.0
    projected_tok_s_high: float = 0.0
    confidence: str = "low"
    bottleneck: str = ""

    reference_tok_s_estimate: float = 0.0
    projected_speedup_low: float = 0.0
    projected_speedup_high: float = 0.0
    recommended: bool = False

    concurrency: int = 1
    context: int = 8192
    notes: list = field(default_factory=list)
    limitations: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)


def _interp_hit(curve, gb):
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    if gb <= xs[0]:
        return ys[0] * (gb / xs[0]) if xs[0] else 0.0
    if gb >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if gb <= xs[i]:
            f = (gb - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def _kv_bytes(model, context, concurrency):
    """KV + state estimate. Deliberately generous: under-reserving OOMs."""
    # hybrid attention: assume a KV-bearing block every 4th, 2 KV heads,
    # head_dim 256, 2 bytes per element, K and V.
    kv_blocks = max(1, model.n_blocks // 4)
    per_tok = kv_blocks * 2 * 256 * 2 * 2
    kv = per_tok * context * max(1, concurrency)
    ssm_state = 192 << 20        # recurrent state, roughly constant
    workspace = 512 << 20        # activations, graph scratch
    return int(kv + ssm_state + workspace)


def plan(model, hw, context: int = 8192, concurrency: int = 1,
         workload: str = "auto", allow_cpu_expert: bool = True) -> ExecutionPlan:
    p = ExecutionPlan(model=os.path.basename(model.path),
                      architecture=model.architecture,
                      context=context, concurrency=concurrency)

    if not model.supported:
        p.exec_class = ExecClass.REFERENCE
        p.notes.append("model not supported by Sabah: %s"
                       % "; ".join(model.notes[-1:]))
        return p

    cal = HIT_CURVES.get(model.architecture)
    if cal is None:
        p.exec_class = ExecClass.REFERENCE
        p.notes.append("no measured hit-rate calibration for architecture %r; "
                       "Sabah will not guess one" % model.architecture)
        return p

    bank = model.expert_bank_bytes
    fixed = model.fixed_bytes
    kvws = _kv_bytes(model, context, concurrency)

    # ---- per-device capacity. Aggregate VRAM is NOT unified memory. -------
    usable_total = 0
    for g in hw.gpus:
        d = DevicePlan(index=g.index, name=g.name, vram_usable=g.vram_usable,
                       fixed_bytes=fixed, kv_workspace_bytes=kvws)
        free = g.vram_usable - fixed - kvws
        d.expert_capacity = max(0, free)
        usable_total += d.expert_capacity
        p.devices.append(d)
    p.expert_capacity_total = usable_total
    p.expert_capacity_frac_of_bank = usable_total / bank if bank else 0.0
    p.fixed_replicated = True

    # every device must independently hold fixed + kv/workspace
    starved = [d.index for d in p.devices if d.expert_capacity <= 0]
    if starved and hw.gpus:
        p.notes.append(
            "GPU(s) %s cannot hold the %.2f GB fixed path plus %.2f GB of "
            "KV/workspace at context %d; no expert capacity there"
            % (starved, fixed / 1e9, kvws / 1e9, context))

    # ---- where does the bank live? ---------------------------------------
    p.ram_bank_bytes = bank
    p.ram_bank_resident = hw.ram_usable >= bank
    if not p.ram_bank_resident:
        p.notes.append(
            "expert bank is %.1f GB but only %.1f GB of RAM is usable; the "
            "remainder is served from storage at %.2f GB/s"
            % (bank / 1e9, hw.ram_usable / 1e9, hw.storage_read_gbps or 0.0))

    # ---- choose execution class ------------------------------------------
    gpu_bw = 0.0
    for g in hw.gpus:
        # nominal device bandwidth is not exposed; approximate from H2D class
        gpu_bw += max(g.h2d_solo_gbps * 12.0, 100.0) * 1e9 * GPU_BW_EFFICIENCY
    h2d = hw.effective_h2d_gbps * 1e9
    cpu_expert_bw = (hw.ram_bandwidth_gbps * 1e9) * CPU_EXPERT_BW_FRACTION

    if usable_total >= bank and hw.gpus:
        p.exec_class = ExecClass.FULL_GPU
        p.hit_rate = 1.0
        p.hit_rate_source = "entire bank resident"
    elif hw.gpus and usable_total > 0:
        p.exec_class = ExecClass.GPU_HOT_TIER
        cap_gb = usable_total / 1e9
        base = _interp_hit(cal["lru"], cap_gb)
        p.hit_rate_floor = _interp_hit(cal["static_deployable"], cap_gb)
        if workload not in ("auto", "mixed", "general"):
            base = min(0.995, base + cal["specialised_bonus"])
            p.hit_rate_source = ("measured LRU curve + %.2f specialisation "
                                 "bonus for profile %r"
                                 % (cal["specialised_bonus"], workload))
        else:
            p.hit_rate_source = "measured LRU curve (the runtime's own policy)"
        # No generalisation-gap subtraction here: the LRU curve was already
        # measured on tokens the warm-up window never saw, so the gap is
        # inside the number rather than on top of it.
        p.hit_rate = max(0.0, min(1.0, base))
        p.hit_rate_source += ("; static-placement floor at this capacity %.4f"
                              % p.hit_rate_floor)
    elif allow_cpu_expert and cpu_expert_bw > 0:
        p.exec_class = ExecClass.CPU_EXPERT
        p.hit_rate = 1.0 if p.ram_bank_resident else 0.0
        p.hit_rate_source = "experts execute on CPU from RAM"
    else:
        p.exec_class = ExecClass.REFERENCE
        p.notes.append("no usable GPU and no CPU expert path")
        return p

    if not p.ram_bank_resident and p.exec_class != ExecClass.FULL_GPU:
        p.exec_class = ExecClass.STORAGE_BACKED

    # ---- timing model (no perfect cross-layer prefetch) ------------------
    reuse = (cal["reuse_same_domain"] if workload not in ("auto", "mixed", "general")
             else cal["reuse_mixed"])
    R = reuse.get(concurrency) or reuse[max(k for k in reuse if k <= concurrency)]
    # Bytes for the WHOLE batch. Weights are shared across the B tokens of a
    # pass (that is what R measures), arithmetic is not. A batch does not cost
    # the same as a single token.
    expert_bytes_batch = model.per_token_expert_bytes * concurrency / R
    expert_bytes_tok = expert_bytes_batch / concurrency

    if p.exec_class in (ExecClass.FULL_GPU, ExecClass.GPU_HOT_TIER):
        t_fixed = fixed / gpu_bw            # weights read once per pass
        t_comp = expert_bytes_batch / gpu_bw
        miss = expert_bytes_batch * (1.0 - p.hit_rate)
        t_fetch = miss / h2d if h2d else float("inf")
        p.bottleneck = ("RAM->GPU expert streaming" if t_fetch > max(t_fixed, t_comp)
                        else "GPU fixed path")
    elif p.exec_class == ExecClass.CPU_EXPERT:
        t_fixed = fixed / gpu_bw if hw.gpus else fixed / max(cpu_expert_bw, 1)
        t_comp = expert_bytes_batch / max(cpu_expert_bw, 1)
        t_fetch = 0.0
        miss = 0
        p.bottleneck = "CPU expert bandwidth"
    else:  # STORAGE_BACKED
        t_fixed = fixed / gpu_bw if hw.gpus else 0.0
        t_comp = expert_bytes_batch / max(gpu_bw, 1)
        resident_frac = (hw.ram_usable / bank) if bank else 0.0
        miss = expert_bytes_batch * max(0.0, 1.0 - resident_frac)
        ssd = (hw.storage_read_gbps or 0.5) * 1e9
        t_fetch = miss / ssd
        p.bottleneck = "storage-backed expert reads"

    # The low bound of the band is not an arbitrary percentage: for a hot tier
    # it is the SAME model re-evaluated at the static-placement hit rate, i.e.
    # what Sabah would achieve if LRU bought nothing at all.
    t_pass_floor = None
    if p.exec_class == ExecClass.GPU_HOT_TIER and p.hit_rate_floor > 0:
        miss_floor = expert_bytes_batch * (1.0 - p.hit_rate_floor)
        t_fetch_floor = miss_floor / h2d if h2d else float("inf")
        t_pass_floor = ((t_fixed + max(t_fetch_floor, t_comp))
                        * (1.0 + OVERHEAD_FRAC))

    p.miss_bytes_per_token = int(miss / max(1, concurrency))
    # reported per token so the numbers stay comparable across concurrency
    p.t_fixed_ms = 1e3 * t_fixed / max(1, concurrency)
    p.t_expert_compute_ms = 1e3 * t_comp / max(1, concurrency)
    p.t_expert_fetch_ms = 1e3 * t_fetch / max(1, concurrency)

    # intra-block overlap only: the expert path costs max(fetch, compute).
    # t_pass is the time for one forward pass carrying `concurrency` tokens.
    t_pass = (t_fixed + max(t_fetch, t_comp)) * (1.0 + OVERHEAD_FRAC)
    p.t_token_ms = 1e3 * t_pass / max(1, concurrency)

    agg = concurrency / t_pass if t_pass > 0 else 0.0
    # uncertainty band: tighter when the plan is compute/fixed bound, wider
    # when it depends on a predicted hit rate
    spread = 0.10 if p.exec_class == ExecClass.FULL_GPU else 0.18
    if t_pass_floor and t_pass_floor > 0:
        agg_floor = concurrency / t_pass_floor
        p.projected_tok_s_low = min(agg_floor, agg * (1 - spread))
        p.projected_tok_s_high = agg * (1 + spread)
    else:
        p.projected_tok_s_low = agg * (1 - spread)
        p.projected_tok_s_high = agg * (1 + spread)
    p.confidence = ("high" if p.exec_class == ExecClass.FULL_GPU else
                    "medium" if p.exec_class == ExecClass.GPU_HOT_TIER else "low")

    # ---- reference estimate and the honest recommendation ----------------
    ref_bw = cpu_expert_bw if cpu_expert_bw > 0 else 1e9
    t_ref = (fixed / gpu_bw if hw.gpus else fixed / ref_bw) + \
            model.per_token_expert_bytes / ref_bw
    if not p.ram_bank_resident:
        ssd = (hw.storage_read_gbps or 0.5) * 1e9
        t_ref += model.per_token_expert_bytes * max(0.0, 1 - hw.ram_usable / bank) / ssd
    t_ref *= (1.0 + OVERHEAD_FRAC)
    p.reference_tok_s_estimate = 1.0 / t_ref if t_ref > 0 else 0.0
    if p.reference_tok_s_estimate > 0:
        p.projected_speedup_low = p.projected_tok_s_low / p.reference_tok_s_estimate
        p.projected_speedup_high = p.projected_tok_s_high / p.reference_tok_s_estimate

    p.recommended = p.projected_speedup_low >= 1.10
    if p.projected_tok_s_high < 5.0:
        p.notes.append(
            "absolute throughput is very low (%.1f-%.1f tok/s) whatever the "
            "speedup: this machine is undersized for a %.0f GB artifact. More "
            "RAM (to hold the %.0f GB expert bank) is the single biggest win."
            % (p.projected_tok_s_low, p.projected_tok_s_high,
               (bank + fixed + model.lookup_bytes) / 1e9, bank / 1e9))
    if not p.recommended:
        p.exec_class = ExecClass.REFERENCE if p.projected_speedup_high < 1.0 \
            else p.exec_class
        p.notes.append(
            "projected speedup %.2f-%.2fx does not clear the 1.10x bar; "
            "reference runtime recommended"
            % (p.projected_speedup_low, p.projected_speedup_high))

    p.limitations.append(
        "hit rate from a measured mixed-workload curve (%s); actual traffic "
        "may differ" % cal["provenance"])
    p.limitations.append(
        "PROJECTED, not measured. Replace with `sabah bench` results.")
    if len(hw.gpus) > 1 and hw.h2d_contention_factor < 0.8:
        p.limitations.append(
            "PCIe root complex scales at %.2f; the aggregate figure was used"
            % hw.h2d_contention_factor)
    return p


def summarize(p: ExecutionPlan) -> str:
    L, a = [], lambda s: L.append(s)
    a("execution  : %s" % p.exec_class.value)
    a("model      : %s (%s)" % (p.model, p.architecture))
    if p.devices:
        for d in p.devices:
            a("  GPU%d %-30s expert capacity %6.2f GB  (of %.2f usable)"
              % (d.index, d.name[:30], d.expert_capacity / 1e9, d.vram_usable / 1e9))
    a("expert tier: %.2f GB total = %.1f%% of the bank"
      % (p.expert_capacity_total / 1e9, 100 * p.expert_capacity_frac_of_bank))
    a("ram bank   : %s (%.1f GB)"
      % ("resident" if p.ram_bank_resident else "DOES NOT FIT", p.ram_bank_bytes / 1e9))
    a("hit rate   : %.4f  [%s]" % (p.hit_rate, p.hit_rate_source))
    if p.hit_rate_floor and p.hit_rate_floor < p.hit_rate:
        a("             (%.4f static / %.4f LRU: the spread between them is "
          "what caching buys)" % (p.hit_rate_floor, p.hit_rate))
    a("per token  : fixed %.2f ms | expert compute %.2f ms | expert fetch %.2f ms"
      % (p.t_fixed_ms, p.t_expert_compute_ms, p.t_expert_fetch_ms))
    a("             miss %.1f MB/token -> %.2f ms/token total"
      % (p.miss_bytes_per_token / 1e6, p.t_token_ms))
    a("bottleneck : %s" % p.bottleneck)
    a("PROJECTED  : %.1f-%.1f tok/s at concurrency %d (confidence: %s)"
      % (p.projected_tok_s_low, p.projected_tok_s_high, p.concurrency, p.confidence))
    a("             vs reference estimate %.1f tok/s -> %.2f-%.2fx PROJECTED"
      % (p.reference_tok_s_estimate, p.projected_speedup_low, p.projected_speedup_high))
    a("recommended: %s" % ("YES" if p.recommended else "NO — use reference runtime"))
    for n in p.notes:
        a("note       : %s" % n)
    for n in p.limitations:
        a("limitation : %s" % n)
    return "\n".join(L)
