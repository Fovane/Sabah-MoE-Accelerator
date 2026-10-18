"""
Sabah - locality test: same experts per block across all tokens.
Tests if Sabah can hit 10-15 tok/s when routing has expert locality.
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah run-locality")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--hot-bytes", type=int, default=4294967296)
    ap.add_argument("--bank-mode", default="ram", choices=["mmap", "ram"])
    ap.add_argument("--pin-fixed-path", action="store_true")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    if not rt.available() or rt.device_count() < 1:
        print("CUDA runtime unavailable."); return 2
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")

    prof = inspect_model(args.model)
    if not prof.supported:
        print("unsupported model"); return 2

    fpr = None
    if args.pin_fixed_path:
        fpr = FixedPathResident(prof)
        fpr.load()
        print("fixed path: %.3f GB pinned in VRAM (%d tensors)"
              % (fpr.resident_bytes / 1e9, len(fpr._ptrs)))

    bank = ExpertBank(prof, mode=args.bank_mode,
                      blocks=list(range(prof.n_blocks)))

    tier = HotTier(bank, capacity_bytes=args.hot_bytes,
                   blocks=list(range(prof.n_blocks)),
                   prefetch_slots=0)
    print("hot tier : %.3f GB VRAM, %d slots/block"
          % (tier.capacity_bytes / 1e9, tier.slots_per_block))

    ex = MoEExecutor(bank, tier, max_used=prof.n_experts_used)

    # ---- LOCALITY: same 10 experts per block for all tokens -------------
    # Pick experts 0..9 for every block, every token.
    # This guarantees 100% hit rate after warmup.
    n_tokens = args.tokens + args.warmup
    traces = {}
    for b in range(prof.n_blocks):
        traces[b] = np.tile(np.arange(prof.n_experts_used, dtype=np.int32),
                            (n_tokens, 1))

    w = np.full(prof.n_experts_used, 1.0 / prof.n_experts_used, dtype=np.float32)
    x = (np.random.default_rng(7).standard_normal(prof.d_model)
         .astype(np.float32) * 0.05)

    # warmup
    print("warming up %d tokens..." % args.warmup)
    for t in range(args.warmup):
        for b in range(prof.n_blocks):
            ex.run_block(b, traces[b][t].tolist(), w)
    ex.sync()
    tier.tel.reset()

    # timed
    print("running %d tokens x %d blocks..." % (args.tokens, prof.n_blocks))
    rt.check(rt.lib().sabah_sync_all(), "pre-timer sync")
    t0 = time.perf_counter()
    for t in range(args.tokens):
        for b in range(prof.n_blocks):
            ex.run_block(b, traces[b][args.warmup + t].tolist(), w)
    ex.sync()
    dt = time.perf_counter() - t0

    n_blocks_executed = args.tokens * prof.n_blocks
    ms_per_token = 1e3 * dt / args.tokens
    tok_s = args.tokens / dt
    blk_tok_s = n_blocks_executed / dt

    print("=" * 60)
    print("RESULT (LOCALITY): %.3f ms/token  ->  %.1f tok/s"
          % (ms_per_token, tok_s))
    print("tokens : %d" % args.tokens)
    print("blocks : %d x %d = %d" % (prof.n_blocks, args.tokens, n_blocks_executed))
    print("wall   : %.3f s" % dt)
    print("hit rate: %.4f" % tier.tel.hit_rate)
    print("bytes fetched: %.2f MB total" % (tier.tel.bytes_fetched / 1e6))
    print("=" * 60)

    if args.json_out:
        import json
        with open(args.json_out, "w") as f:
            json.dump(dict(
                model=os.path.basename(args.model),
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
            ), f, indent=1)

    ex.free()
    tier.free()
    if fpr:
        fpr.free()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
