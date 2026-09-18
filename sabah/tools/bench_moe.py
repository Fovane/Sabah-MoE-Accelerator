"""
Sabah - trace-driven MoE block benchmark.

This is the first thing in the whole Sabah project that produces MEASURED
throughput rather than a projection. What it measures is precise and limited:

    the cost of executing ONE routed-MoE block, on real weights, for a real
    routing sequence, at a given VRAM residency.

It does NOT measure end-to-end tokens/s for the full model. Attention, the
fixed path, sampling and the other 47 blocks are absent. Any figure derived
from this by multiplication is labelled PROJECTED, never measured.

Two things it answers that no simulator can:

  1. Does the measured hit rate match the hit curve v4 simulated? The planner
     is calibrated on that curve; if hardware disagrees, the planner is wrong.

  2. The M1 A/B: the same block, same routing, run fully resident (no transfer
     at all) and streamed (hot tier smaller than the working set). The
     difference is the streaming penalty, isolated from kernel cost.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sabah.core.model_inspector import inspect_model
from sabah.runtime import rt
from sabah.runtime.expert_bank import ExpertBank
from sabah.runtime.hot_tier import HotTier
from sabah.runtime.executor import MoEExecutor

DEFAULT_MODEL = (r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                 r"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
DEFAULT_TRACE = r"D:/sabah_scaling/v3/traces/main2"


def load_trace(trace_dir: str, block: int, topk: int) -> np.ndarray:
    """Routing ids for one block: [n_tokens, topk]."""
    p = os.path.join(trace_dir, "ffn_moe_topk_L%d.i32" % block)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    a = np.fromfile(p, dtype=np.int32)
    if a.size % topk:
        raise ValueError("%s holds %d ids, not a multiple of top-%d"
                         % (os.path.basename(p), a.size, topk))
    return a.reshape(-1, topk)


def popularity(ids: np.ndarray, n_experts: int) -> np.ndarray:
    c = np.bincount(ids.reshape(-1), minlength=n_experts)
    return np.argsort(-c, kind="stable")


def run_one(bank, block, ids, capacity_slots, warm_order, x,
            max_used, repeat_sync=True):
    """Replay `ids` through a tier of `capacity_slots` and time it."""
    slot_bytes = bank.block_expert_bytes(block)
    tier = HotTier(bank, capacity_slots * slot_bytes, blocks=[block])
    if tier.slots_per_block != capacity_slots:
        tier.free()
        raise RuntimeError("asked for %d slots, tier built %d"
                           % (capacity_slots, tier.slots_per_block))
    ex = MoEExecutor(bank, tier, max_used=max_used)
    try:
        tier.preload(block, warm_order[:capacity_slots].tolist())
        ex.upload_x(x)
        # one untimed token so allocation and first-touch are not in the result
        ex.run_block(block, ids[0].tolist(), np.full(ids.shape[1], 0.1, np.float32))
        ex.sync()
        tier.drain_stalls()
        tier.tel.reset()

        w = np.full(ids.shape[1], 1.0 / ids.shape[1], dtype=np.float32)
        rt.check(rt.lib().sabah_sync_all(), "pre-timer sync")
        t0 = time.perf_counter()
        for t in range(ids.shape[0]):
            ex.run_block(block, ids[t].tolist(), w)
        ex.sync()
        dt = time.perf_counter() - t0
        tier.drain_stalls()
        tier.tel.tokens = ids.shape[0]

        res = tier.tel.as_dict()
        res.update(capacity_slots=capacity_slots,
                   vram_bytes=tier.capacity_bytes,
                   wall_s=dt,
                   ms_per_token=1000.0 * dt / ids.shape[0])
        return res
    finally:
        ex.free()
        tier.free()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah bench-moe")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--block", type=int, default=0)
    ap.add_argument("--tokens", type=int, default=2000)
    ap.add_argument("--warm-frac", type=float, default=0.2)
    ap.add_argument("--slots", default="16,32,64,128,256,512")
    ap.add_argument("--bank-mode", default="mmap", choices=["mmap", "ram"],
                    help="mmap: page-cache backed. ram: copy the block's experts "
                         "into pinned RAM first, so the DMA engine reads in place.")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    print("=" * 84)
    print("SABAH - MEASURED MoE block benchmark (one block, real weights, real trace)")
    print("=" * 84)

    if not rt.available() or rt.device_count() < 1:
        print("CUDA runtime unavailable."); return 2
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")

    prof = inspect_model(args.model)
    if not prof.supported:
        print("unsupported model"); return 2

    need = sum(t["per_expert_bytes"] for t in prof.expert_tensors
               if t["block"] == args.block) * prof.n_experts
    if args.bank_mode == "ram":
        print("loading block %d's expert bank into pinned RAM (%.3f GB)..."
              % (args.block, need / 1e9))
    bank = ExpertBank(prof, mode=args.bank_mode, blocks=[args.block],
                      pinned=(args.bank_mode == "ram"))
    ids_all = load_trace(args.trace, args.block, prof.n_experts_used)

    n_warm = int(ids_all.shape[0] * args.warm_frac)
    warm_order = popularity(ids_all[:n_warm], prof.n_experts)
    ids = ids_all[n_warm:n_warm + args.tokens]
    if ids.shape[0] < args.tokens:
        print("note: trace supplies only %d tokens after warm-up" % ids.shape[0])

    qmix = "%s/%s/%s" % (bank.desc[(args.block, "gate")]["qtype"],
                         bank.desc[(args.block, "up")]["qtype"],
                         bank.desc[(args.block, "down")]["qtype"])
    print("bank       : %s%s"
          % (bank.mode, " (pinned, DMA in place)" if bank.pinned else
             " (page-cache backed, staged through pinned buffers)"))
    print("block      : %d   quant %s   %s bytes/expert"
          % (args.block, qmix, "{:,}".format(bank.block_expert_bytes(args.block))))
    print("trace      : %s  (%d tokens total, %d used for popularity, %d measured)"
          % (os.path.basename(args.trace.rstrip("/\\")), ids_all.shape[0],
             n_warm, ids.shape[0]))
    print("distinct   : %d of %d experts appear in the measured window"
          % (len(np.unique(ids)), prof.n_experts))
    print()

    x = (np.random.default_rng(7).standard_normal(prof.d_model)
         .astype(np.float32) * 0.05)

    sweep = [int(s) for s in args.slots.split(",") if s.strip()]
    rows = []
    print("%8s %8s %8s %10s %11s %11s %11s %10s"
          % ("slots", "VRAM GB", "hit", "ms/token", "gpu stall", "host stage",
             "fetch MB", "blk tok/s"))
    print("%8s %8s %8s %10s %11s %11s %11s %10s"
          % ("", "", "", "", "ms/tok", "ms/tok", "per tok", ""))
    print("-" * 84)
    for C in sweep:
        if C > prof.n_experts:
            continue
        try:
            r = run_one(bank, args.block, ids, C, warm_order, x,
                        max_used=prof.n_experts_used)
        except rt.SabahCudaError as e:
            print("%8d  skipped: %s" % (C, e))
            continue
        rows.append(r)
        print("%8d %8.3f %8.4f %10.3f %11.3f %11.3f %11.2f %10.1f"
              % (C, r["vram_bytes"] / 1e9, r["hit_rate"], r["ms_per_token"],
                 r["gpu_wait_expert_ms"] / r["tokens"],
                 r["host_stage_ms"] / r["tokens"],
                 r["bytes_fetched"] / r["tokens"] / 1e6,
                 1000.0 / r["ms_per_token"]))

    print()
    # ---- M1 A/B ---------------------------------------------------------
    full = next((r for r in rows if r["capacity_slots"] >= prof.n_experts), None)
    if full and len(rows) > 1:
        print("M1 A/B - forced-resident vs streamed")
        print("  fully resident (%d slots): %.3f ms/token, hit %.4f, stall %.3f ms/tok"
              % (full["capacity_slots"], full["ms_per_token"], full["hit_rate"],
                 full["gpu_wait_expert_ms"] / full["tokens"]))
        print("  %-28s %10s %10s %10s" % ("streamed at", "ms/token", "penalty", "x slower"))
        for r in rows:
            if r is full:
                continue
            pen = r["ms_per_token"] - full["ms_per_token"]
            print("  %-28s %10.3f %10.3f %10.2f"
                  % ("%d slots (%.0f%% of bank)"
                     % (r["capacity_slots"],
                        100.0 * r["capacity_slots"] / prof.n_experts),
                     r["ms_per_token"], pen,
                     r["ms_per_token"] / full["ms_per_token"]))
        print()
        print("  The fully-resident row is the kernel cost with ZERO expert")
        print("  transfer. Everything above it is the streaming penalty, which")
        print("  is what the hot tier exists to shrink.")
        print()

    print("SCOPE: these are measurements of ONE block. They are not tokens/s for")
    print("the model. Multiplying by %d blocks ignores attention, the fixed path"
          % prof.n_blocks)
    print("and sampling, and would be a PROJECTION, not a measurement.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(schema=1, model=os.path.basename(args.model),
                           block=args.block, quant=qmix,
                           trace=args.trace, tokens=int(ids.shape[0]),
                           n_experts=prof.n_experts,
                           topk=prof.n_experts_used, rows=rows), f, indent=1)
        print("\nwrote %s" % args.json_out)
    bank.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
