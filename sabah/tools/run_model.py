"""
Sabah - full-model sequential MoE runner.

Runs all 48 MoE blocks sequentially per token, on real weights, using
Sabah's own runtime (no llama.cpp). This is the "Yol 1" path:

    * Fixed path (attention/router/embedding/head) pinned in VRAM once
    * Expert bank read from disk/RAM on-demand via ExpertBank
    * Hot tier (VRAM) + warm tier (pinned RAM) hide transfer latency
    * Block L's router output feeds block L+1 - sequential chain

This does NOT include attention/embedding/head arithmetic (those tensors
are pinned but not executed here). It measures the MoE block throughput
which is the dominant cost for this MoE model.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sabah.core.model_inspector import inspect_model
from sabah.runtime import rt
from sabah.runtime.expert_bank import ExpertBank
from sabah.runtime.hot_tier import HotTier
from sabah.runtime.executor import MoEExecutor
from sabah.runtime.residency import FixedPathResident


def load_trace(trace_dir: str, n_blocks: int, topk: int, n_tokens: int,
               n_experts: int) -> dict:
    """Load routing ids for all blocks: {block: [n_tokens, topk]}.

    Missing trace files are filled with random top-k ids (uniform over
    experts) so the runner works on any machine without a pre-captured trace.
    """
    traces = {}
    rng = np.random.default_rng(12345)
    for b in range(n_blocks):
        p = os.path.join(trace_dir, "ffn_moe_topk_L%d.i32" % b)
        if os.path.exists(p):
            a = np.fromfile(p, dtype=np.int32)
            if a.size % topk:
                raise ValueError("%s holds %d ids, not a multiple of top-%d"
                                 % (os.path.basename(p), a.size, topk))
            traces[b] = a.reshape(-1, topk)[:n_tokens]
        else:
            traces[b] = rng.integers(0, n_experts, size=(n_tokens, topk),
                                     dtype=np.int32)
    return traces


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah run-model")
    ap.add_argument("--model", required=True, help="Path to shard 1 .gguf")
    ap.add_argument("--trace", required=True, help="Directory with ffn_moe_topk_L*.i32")
    ap.add_argument("--tokens", type=int, default=500, help="Tokens to measure")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup tokens (not timed)")
    ap.add_argument("--hot-bytes", type=int, default=536870912,
                    help="VRAM hot tier bytes (default 512 MB)")
    ap.add_argument("--bank-mode", default="mmap", choices=["mmap", "ram"],
                    help="mmap: page-cache backed. ram: copy block experts into pinned RAM.")
    ap.add_argument("--prefetch-slots", type=int, default=0,
                    help="Prefetch reserve slots per block (0 = off)")
    ap.add_argument("--pin-fixed-path", action="store_true",
                    help="Pin fixed path tensors into VRAM (piece C)")
    ap.add_argument("--json-out", default="", help="Write results JSON")
    args = ap.parse_args(argv)

    if not rt.available() or rt.device_count() < 1:
        print("CUDA runtime unavailable."); return 2
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")

    prof = inspect_model(args.model)
    if not prof.supported:
        print("unsupported model"); return 2

    # ---- pin fixed path into VRAM (piece C) -------------------------------
    fpr = None
    if args.pin_fixed_path:
        fpr = FixedPathResident(prof)
        fpr.load()
        print("fixed path: %.3f GB pinned in VRAM (%d tensors)"
              % (fpr.resident_bytes / 1e9, len(fpr._ptrs)))

    # ---- build expert bank --------------------------------------------------
    bank = ExpertBank(prof, mode=args.bank_mode,
                      blocks=list(range(prof.n_blocks)))

    # ---- build hot tier -----------------------------------------------------
    tier = HotTier(bank, capacity_bytes=args.hot_bytes,
                   blocks=list(range(prof.n_blocks)),
                   prefetch_slots=args.prefetch_slots)
    print("hot tier : %.3f GB VRAM, %d slots/block"
          % (tier.capacity_bytes / 1e9, tier.slots_per_block))

    # ---- build executor -----------------------------------------------------
    ex = MoEExecutor(bank, tier, max_used=prof.n_experts_used)

    # ---- load routing traces ------------------------------------------------
    traces = load_trace(args.trace, prof.n_blocks,
                        prof.n_experts_used, args.tokens + args.warmup,
                        prof.n_experts)

    # ---- uniform routing weights -------------------------------------------
    w = np.full(prof.n_experts_used, 1.0 / prof.n_experts_used, dtype=np.float32)

    # ---- d_model input ------------------------------------------------------
    x = (np.random.default_rng(7).standard_normal(prof.d_model)
         .astype(np.float32) * 0.05)

    # ---- warmup (not timed) ------------------------------------------------
    print("warming up %d tokens..." % args.warmup)
    for t in range(args.warmup):
        for b in range(prof.n_blocks):
            ex.run_block(b, traces[b][t].tolist(), w)
    ex.sync()
    tier.tel.reset()

    # ---- timed run ----------------------------------------------------------
    print("running %d tokens x %d blocks..." % (args.tokens, prof.n_blocks))
    rt.check(rt.lib().sabah_sync_all(), "pre-timer sync")
    t0 = time.perf_counter()
    for t in range(args.tokens):
        for b in range(prof.n_blocks):
            ex.run_block(b, traces[b][args.warmup + t].tolist(), w)
    ex.sync()
    dt = time.perf_counter() - t0

    # ---- results ------------------------------------------------------------
    n_blocks_executed = args.tokens * prof.n_blocks
    ms_per_token = 1e3 * dt / args.tokens
    tok_s = args.tokens / dt
    blk_tok_s = n_blocks_executed / dt

    print("=" * 60)
    print("RESULT: %.3f ms/token  ->  %.1f tok/s  (%.1f block-tok/s)"
          % (ms_per_token, tok_s, blk_tok_s))
    print("tokens : %d" % args.tokens)
    print("blocks : %d x %d = %d block executions"
          % (prof.n_blocks, args.tokens, n_blocks_executed))
    print("wall   : %.3f s" % dt)
    print("hit rate: %.4f" % tier.tel.hit_rate)
    print("bytes fetched: %.2f MB total"
          % (tier.tel.bytes_fetched / 1e6))
    print("=" * 60)

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(dict(
                model=os.path.basename(args.model),
                block_count=prof.n_blocks,
                tokens=args.tokens,
                ms_per_token=ms_per_token,
                tok_s=tok_s,
                blk_tok_s=blk_tok_s,
                wall_s=dt,
                hit_rate=tier.tel.hit_rate,
                bytes_fetched=tier.tel.bytes_fetched,
                hot_bytes=args.hot_bytes,
                bank_mode=args.bank_mode,
                fixed_path_pinned=args.pin_fixed_path,
                fixed_path_bytes=fpr.resident_bytes if fpr else 0,
            ), f, indent=1)
        print("wrote %s" % args.json_out)

    ex.free()
    tier.free()
    if fpr:
        fpr.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
