#!/usr/bin/env python3
"""Decode-token timeline from an Nsight Systems SQLite export.

    python tools/nsys_decode.py <report.sqlite> [--groups N] [--group-size 144]

Segments the k_mul_mat_id launch stream into decode tokens (144 launches each:
48 layers x gate/up/down), then reports per token:

  * GPU busy (union of kernel intervals) vs idle
  * kernel time by category and by Sabah tensor (gate/up rows=640, down rows=2560)
  * cudaStreamSynchronize count and duration, cudaMemcpyAsync by direction
  * kernel launches per token
  * per-layer busy/idle, using the three expert launches as layer markers

The trace's own overhead inflates host gaps; the script is a measurement aid,
not a projection.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter, defaultdict


def load_kernels(c):
    q = """select k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ,
                  k.blockX, k.streamId, k.registersPerThread
           from CUPTI_ACTIVITY_KIND_KERNEL k
           join StringIds s on s.id = k.demangledName
           order by k.start"""
    return list(c.execute(q))


def load_runtime(c):
    q = """select r.start, r.end, s.value
           from CUPTI_ACTIVITY_KIND_RUNTIME r
           join StringIds s on s.id = r.nameId
           order by r.start"""
    return list(c.execute(q))


def load_memcpy(c):
    q = """select start, end, bytes, copyKind
           from CUPTI_ACTIVITY_KIND_MEMCPY order by start"""
    return list(c.execute(q))


def union_busy(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        elif e > ce:
            ce = e
    total += ce - cs
    return total


def category(name, grid_x):
    n = name.lower()
    if "k_mul_mat_id" in n:
        if grid_x == 640:
            return "sabah gate+up"
        if grid_x == 2560:
            return "sabah down"
        return "sabah other"
    if any(k in n for k in ("flash_attn", "attention", "soft_max", "rope", "wkv",
                            "kv_", "attn")):
        return "attention"
    if any(k in n for k in ("ssm", "gated_delta", "conv", "delta_net")):
        return "ssm"
    if "k_get_rows" in n or "get_rows" in n:
        return "ple/embedding"
    if "mul_mat" in n or "gemm" in n or "cutlass" in n:
        return "dense matmul"
    if any(k in n for k in ("rms_norm", "norm", "scale_f32", "unary", "k_bin",
                            "quantize", "convert", "moe_weighted", "topk",
                            "concat", "copy", "set_rows", "sum", "arg")):
        return "elementwise/router"
    return "other"


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--groups", type=int, default=8,
                    help="trailing decode groups to summarize")
    ap.add_argument("--group-size", type=int, default=144)
    ap.add_argument("--split-gap-ms", type=float, default=1000.0,
                    help="gap between decode launches that starts a new item")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    kernels = load_kernels(c)
    runtime = load_runtime(c)
    memcpy = load_memcpy(c)

    mm = [k for k in kernels if "k_mul_mat_id" in k[2] and k[5] == 1]
    print("kernels=%d runtime=%d memcpy=%d  decode_mmid=%d" %
          (len(kernels), len(runtime), len(memcpy), len(mm)))
    if len(mm) < args.group_size:
        print("not enough decode launches")
        return 1

    # Split at item boundaries (the warm-tier replay clears the KV and runs the
    # next prompt; the gap between two decode launches then holds the whole
    # prefill). Grouping every 144 launches across that boundary misaligns.
    segs = []
    cur = [mm[0]]
    for k in mm[1:]:
        if (k[0] - cur[-1][0]) / 1e6 > args.split_gap_ms:
            segs.append(cur)
            cur = []
        cur.append(k)
    segs.append(cur)
    print("items:", [len(s) for s in segs])
    groups = []
    for seg in segs:
        for i in range(0, len(seg) - args.group_size + 1, args.group_size):
            groups.append(seg[i:i + args.group_size])
    sizes = Counter(len(g) for g in groups)
    print("groups:", sizes, "using last", args.groups)
    groups = groups[-args.groups:]

    import json
    doc = {"db": args.db, "tokens": []}

    for gi, g in enumerate(groups):
        a = g[0][0]
        b = g[-1][1]
        # include the tail after the last expert kernel, up to the next group's
        # first launch (final norm / logits read) where one exists
        if gi + 1 < len(groups):
            b = groups[gi + 1][0][0]
        else:
            tail = [k for k in kernels if k[0] >= b]
            b = min(tail[0][0], b + 10_000_000) if tail else b
        span = b - a
        ks = [k for k in kernels if a <= k[0] < b]
        busy = union_busy([(k[0], min(k[1], b)) for k in ks])
        by = defaultdict(int)
        cnt = Counter()
        by_name = defaultdict(int)
        for k in ks:
            key = category(k[2], k[3])
            by[key] += min(k[1], b) - k[0]
            cnt[key] += 1
            by_name[k[2]] += min(k[1], b) - k[0]
        rt = [r for r in runtime if a <= r[0] < b]
        sync = [r for r in rt if "StreamSynchronize" in r[2]]
        launch = [r for r in rt if "LaunchKernel" in r[2] or r[2] == "cuLaunchKernel"]
        cp = [m for m in memcpy if a <= m[0] < b]
        cp_by = Counter()
        cp_n = Counter()
        for m in cp:
            cp_by[(m[3], m[2])] += m[1] - m[0]
            cp_n[(m[3], m[2])] += 1

        # per-layer: four groups of 3 launches = 48 layers
        layers = []
        for L in range(len(g) // 3):
            la = g[3 * L][0]
            lb = g[3 * L + 3][0] if 3 * L + 3 < len(g) else b
            lks = [k for k in ks if la <= k[0] < lb]
            lbusy = union_busy([(k[0], min(k[1], lb)) for k in lks])
            layers.append({"layer": L, "span_ms": (lb - la) / 1e6,
                           "busy_ms": lbusy / 1e6,
                           "busy_pct": pct(lbusy, lb - la),
                           "kernels": len(lks)})

        rec = {
            "token": gi,
            "start_ns": a, "end_ns": b,
            "span_ms": span / 1e6,
            "gpu_busy_ms": busy / 1e6,
            "gpu_busy_pct": pct(busy, span),
            "idle_ms": (span - busy) / 1e6,
            "kernels": len(ks),
            "kernel_time_ms": {k: v / 1e6 for k, v in by.items()},
            "kernel_count": dict(cnt),
            "kernel_time_top": {k: v / 1e6 for k, v in
                                sorted(by_name.items(), key=lambda x: -x[1])[:15]},
            "launch_calls": len(launch),
            "runtime_calls": len(rt),
            "stream_sync_calls": len(sync),
            "stream_sync_ms": sum(r[1] - r[0] for r in sync) / 1e6,
            "stream_sync_max_ms": (max((r[1] - r[0] for r in sync), default=0) / 1e6),
            "memcpy": {f"kind{k}": v / 1e6 for (k, _b), v in cp_by.items()},
            "memcpy_n": {f"kind{k}": v for (k, _b), v in cp_n.items()},
            "layers": layers,
        }
        doc["tokens"].append(rec)
        print("\n== token %d: span %.2f ms  busy %.2f ms (%.1f%%)  idle %.2f ms  kernels %d" %
              (gi, rec["span_ms"], rec["gpu_busy_ms"], rec["gpu_busy_pct"],
               rec["idle_ms"], rec["kernels"]))
        print("   launches %d  sync %d / %.2f ms (max %.2f)  memcpy %s" %
              (rec["launch_calls"], rec["stream_sync_calls"],
               rec["stream_sync_ms"], rec["stream_sync_max_ms"],
               ", ".join("%s:%.2fms/%d" % (k, v, rec["memcpy_n"][k])
                         for k, v in sorted(rec["memcpy"].items()))))
        tot = float(sum(by.values()))
        for k, v in sorted(by.items(), key=lambda x: -x[1]):
            print("   %-20s %8.2f ms  %5.1f%%  n=%d" % (k, v / 1e6,
                                                        100.0 * v / max(tot, 1), cnt[k]))
        print("   top kernels:")
        for k, v in sorted(by_name.items(), key=lambda x: -x[1])[:8]:
            print("      %-60s %7.2f ms" % (str(k)[:60], v / 1e6))
        if layers:
            lv = [l["busy_pct"] for l in layers]
            lv.sort()
            print("   layer busy%%: min %.0f  p50 %.0f  p90 %.0f  max %.0f" %
                  (lv[0], lv[len(lv) // 2], lv[int(len(lv) * 0.9)], lv[-1]))
            print("   layer spans: " + " ".join("%.1f" % l["span_ms"] for l in layers[:12]) + " ...")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=2)
            f.write("\n")
        print("\nwrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
