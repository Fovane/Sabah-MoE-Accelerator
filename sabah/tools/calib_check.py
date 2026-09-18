"""
Sabah - is the planner's hit curve calibrated for the policy the runtime runs?

The planner's embedded HIT_CURVES came from v4, where the hot set was STATIC:
experts were placed once by global popularity and never moved. The Phase B
runtime does not do that. It seeds by popularity and then runs LRU, evicting
and admitting as routing demands.

If LRU beats static, every plan the planner emits is conservative - it will
refuse machines the runtime could serve. If LRU loses, the planner is
over-promising, which is worse. Either way the number has to be measured, on
the same trace, with the same byte weighting and the same capacity budget.

Three policies, one eval window:

  static-oracle     popularity computed on the EVALUATION tokens themselves.
                    Not deployable; it is the best a static placement could
                    ever do, and it is what v4's curve measured.
  static-deployable popularity computed on an earlier warm-up window only.
                    This is what a shipped static placement would achieve.
  lru               seeded from the same warm-up window, then LRU.
                    This is what sabah/runtime/hot_tier.py actually does.
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import collections
import numpy as np
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sabah.core.model_inspector import inspect_model

DEFAULT_MODEL = (r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                 r"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
DEFAULT_TRACE = r"D:/sabah_scaling/v3/traces/main2"

# the planner's current calibration, for direct comparison
PLANNER_CURVE = [(8, 0.4256), (12, 0.5372), (16, 0.6210), (20, 0.6869),
                 (24, 0.7407), (28, 0.7864), (32, 0.8252), (36, 0.8585),
                 (40, 0.8871), (48, 0.9331), (60, 0.9778)]


def interp(curve, x):
    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    return float(np.interp(x, xs, ys))


def load(trace_dir, blocks, topk):
    out = {}
    for b in blocks:
        p = os.path.join(trace_dir, "ffn_moe_topk_L%d.i32" % b)
        out[b] = np.fromfile(p, dtype=np.int32).reshape(-1, topk)
    return out


def counts(ids, n_experts):
    return np.bincount(ids.reshape(-1), minlength=n_experts)


def byte_hit_static(ids, resident: set, per_block_bytes: int):
    """Byte-weighted hit against a fixed resident set."""
    hit = tot = 0
    for e in ids.reshape(-1):
        tot += per_block_bytes
        if int(e) in resident:
            hit += per_block_bytes
    return hit, tot


def byte_hit_lru(ids, seed, slots, per_block_bytes: int):
    cache = OrderedDict((int(e), None) for e in seed[:slots])
    hit = tot = 0
    for row in ids:
        for e in row:
            e = int(e)
            tot += per_block_bytes
            if e in cache:
                cache.move_to_end(e)
                hit += per_block_bytes
            else:
                if len(cache) >= slots:
                    cache.popitem(last=False)
                cache[e] = None
    return hit, tot


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah calib-check")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--trace", default=DEFAULT_TRACE)
    ap.add_argument("--warm-frac", type=float, default=0.2)
    ap.add_argument("--caps", default="4,8,12,16,24,32,40,48")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)

    prof = inspect_model(args.model)
    per = {b: sum(t["per_expert_bytes"] for t in prof.expert_tensors
                  if t["block"] == b)
           for b in range(prof.n_blocks)}
    blocks = sorted(per)
    biggest = max(per.values())

    print("=" * 96)
    print("SABAH - planner hit-curve calibration check (static placement vs LRU)")
    print("=" * 96)
    tr = load(args.trace, blocks, prof.n_experts_used)
    n_tok = tr[blocks[0]].shape[0]
    n_warm = int(n_tok * args.warm_frac)
    print("trace   : %d tokens, %d blocks, top-%d; warm-up %d, evaluated %d"
          % (n_tok, len(blocks), prof.n_experts_used, n_warm, n_tok - n_warm))
    print("slots   : allocated evenly per block, %s bytes/slot (largest block)"
          % "{:,}".format(biggest))
    print()

    seeds, evals, oracle = {}, {}, {}
    for b in blocks:
        a = tr[b]
        seeds[b] = np.argsort(-counts(a[:n_warm], prof.n_experts), kind="stable")
        evals[b] = a[n_warm:]
        oracle[b] = np.argsort(-counts(evals[b], prof.n_experts), kind="stable")

    caps = [float(c) for c in args.caps.split(",") if c.strip()]
    rows = []
    print("%8s %7s %14s %18s %10s %12s %10s"
          % ("cap GB", "slots", "static-oracle", "static-deployable", "LRU",
             "planner", "LRU-planner"))
    print("-" * 96)
    for cap in caps:
        slots = int(cap * 1e9 // (biggest * len(blocks)))
        slots = max(0, min(slots, prof.n_experts))
        if slots == 0:
            continue
        h_o = t_o = h_d = t_d = h_l = t_l = 0
        for b in blocks:
            pb = per[b]
            a, b_ = evals[b], None
            x, y = byte_hit_static(a, set(oracle[b][:slots].tolist()), pb)
            h_o += x; t_o += y
            x, y = byte_hit_static(a, set(seeds[b][:slots].tolist()), pb)
            h_d += x; t_d += y
            x, y = byte_hit_lru(a, seeds[b], slots, pb)
            h_l += x; t_l += y
        so, sd, lru = h_o / t_o, h_d / t_d, h_l / t_l
        pl = interp(PLANNER_CURVE, cap)
        rows.append(dict(cap_gb=cap, slots=slots, static_oracle=so,
                         static_deployable=sd, lru=lru, planner=pl))
        print("%8.0f %7d %14.4f %18.4f %10.4f %12.4f %+10.4f"
              % (cap, slots, so, sd, lru, pl, lru - pl))

    print()
    if rows:
        dl = np.mean([r["lru"] - r["planner"] for r in rows])
        dso = np.mean([r["lru"] - r["static_oracle"] for r in rows])
        dsd = np.mean([r["lru"] - r["static_deployable"] for r in rows])
        print("mean advantage of LRU over an ORACLE static placement : %+.4f" % dso)
        print("mean advantage of LRU over a DEPLOYABLE static placement: %+.4f" % dsd)
        print("mean gap between LRU and the planner's curve           : %+.4f" % dl)
        print()
        if dl > 0.02:
            print("VERDICT: the planner is CONSERVATIVE. It is calibrated on static")
            print("placement while the runtime runs LRU, so it under-predicts the")
            print("hit rate by ~%.2f and will refuse machines the runtime could" % dl)
            print("serve. Safe direction, but it should be recalibrated.")
        elif dl < -0.02:
            print("VERDICT: the planner is OPTIMISTIC by ~%.2f. This is the unsafe"
                  % (-dl))
            print("direction and must be corrected before any plan is trusted.")
        else:
            print("VERDICT: the planner's curve matches the runtime policy.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(schema=1, trace=args.trace, n_tokens=int(n_tok),
                           warm_frac=args.warm_frac, rows=rows), f, indent=1)
        print("\nwrote %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
