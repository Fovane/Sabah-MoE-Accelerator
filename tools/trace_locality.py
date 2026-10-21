"""Trace locality analysis for the routed-MoE expert bank.

Answers one question: with a *global* expert cache of capacity C objects fed by
a real routing trace, what fraction of the per-token routed bytes must come from
the SSD, and how fast does that let the model run?

Trace format is Sabah's own: ``ffn_moe_topk_L<block>.i32``, i32 little endian,
``steps * top_k`` expert ids per block (top_k = 10 for qwen4exp).

Usage:
    python tools/trace_locality.py --trace $DRIVE\\LMStudio\\scratchpads\\dk\\trace_full
    python tools/trace_locality.py --trace <dir> --model <shard1> [--model <shard2>]
                                   --disk-gbps 2.2,1.56,1.0 --json out.json
"""

from __future__ import annotations

import argparse
import array
import heapq
import json
import os
import re
import sys
from collections import OrderedDict, defaultdict


# --------------------------------------------------------------------------- #
def load_trace(trace_dir: str, top_k: int) -> dict[int, list[list[int]]]:
    out: dict[int, list[list[int]]] = {}
    pat = re.compile(r"^ffn_moe_topk_L(\d+)\.i32$")
    for name in sorted(os.listdir(trace_dir)):
        m = pat.match(name)
        if not m:
            continue
        block = int(m.group(1))
        vals = array.array("i")
        vals.frombytes(open(os.path.join(trace_dir, name), "rb").read())
        if len(vals) % top_k:
            raise SystemExit("%s: %d ids not a multiple of top_k=%d"
                             % (name, len(vals), top_k))
        out[block] = [list(vals[i:i + top_k]) for i in range(0, len(vals), top_k)]
    if not out:
        raise SystemExit("no ffn_moe_topk_L*.i32 in %s" % trace_dir)
    return out


def interleave(trace: dict[int, list[list[int]]]) -> tuple[list[tuple[int, int]], int]:
    """Real execution order: for each step, each block in ascending order."""
    blocks = sorted(trace)
    steps = min(len(trace[b]) for b in blocks)
    seq: list[tuple[int, int]] = []
    for s in range(steps):
        for b in blocks:
            for e in trace[b][s]:
                seq.append((b, e))
    return seq, steps


def expert_bytes_per_block(shards: list[str]) -> dict[int, int]:
    """Mean per-object bytes (one expert of one projection) per block.

    The runtime's cache object is a single (expert, projection) tensor, so the
    three roles are averaged: gate/up are 0.61 MB and down is 0.84 MB.
    """
    sys.path.insert(0, r"$DRIVE\LMStudio\scratchpads\dk")
    from gguf_probe import parse  # type: ignore

    per_block: dict[int, int] = defaultdict(int)
    roles: dict[int, set[str]] = defaultdict(set)
    for path in shards:
        _, _, tensors, _ = parse(path)
        fsize = os.path.getsize(path)
        ts = sorted(tensors, key=lambda t: t[3])
        for i, (name, dims, tt, off) in enumerate(ts):
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", name)
            if not m:
                continue
            end = ts[i + 1][3] if i + 1 < len(ts) else fsize
            block, role = int(m.group(1)), m.group(2)
            per_block[block] += (end - off) // 512   # 512 experts per tensor
            roles[block].add(role)
    missing = [b for b in sorted(per_block) if len(roles[b]) != 3]
    if missing:
        print("WARN: blocks missing an expert role: %s" % missing, file=sys.stderr)
    return {b: v // 3 for b, v in per_block.items()}


# --------------------------------------------------------------------------- #
def lru_hits(seq, capacity, weights):
    """Exact LRU, byte-weighted. O(n) with an OrderedDict."""
    resident: OrderedDict = OrderedDict()
    hit_b = miss_b = 0.0
    for key in seq:
        w = weights[key[0]]
        if key in resident:
            hit_b += w
            resident.move_to_end(key)
        else:
            miss_b += w
            resident[key] = None
            if len(resident) > capacity:
                resident.popitem(last=False)
    return hit_b / (hit_b + miss_b), miss_b


def lru_hits_bytes(seq, capacity_bytes, weights):
    """Exact LRU with a byte budget, as the runtime's slab actually works."""
    resident: OrderedDict = OrderedDict()
    used = 0.0
    hit_b = miss_b = 0.0
    miss_n = 0
    for key in seq:
        w = weights[key[0]]
        if key in resident:
            hit_b += w
            resident.move_to_end(key)
            continue
        miss_b += w
        miss_n += 1
        resident[key] = w
        used += w
        while used > capacity_bytes and resident:
            _, v = resident.popitem(last=False)
            used -= v
    total = hit_b + miss_b
    return (hit_b / total if total else 0.0), miss_b, miss_n


def static_topk_hits(seq, capacity, weights):
    """Offline static placement: the `capacity` most frequent objects, preloaded."""
    freq: dict = defaultdict(int)
    for key in seq:
        freq[key] += 1
    top = set(sorted(freq, key=lambda k: -freq[k])[:capacity])
    hit_b = miss_b = 0.0
    for key in seq:
        w = weights[key[0]]
        if key in top:
            hit_b += w
        else:
            miss_b += w
    return hit_b / (hit_b + miss_b), miss_b


def belady_hits(seq, capacity, weights):
    """Offline optimal (farthest-in-future): an upper bound for any policy."""
    n = len(seq)
    nxt = [n] * n
    last: dict = {}
    for i in range(n - 1, -1, -1):
        k = seq[i]
        nxt[i] = last.get(k, n)
        last[k] = i

    resident: set = set()
    cur: dict = {}
    heap: list = []
    hit_b = miss_b = 0.0
    for i, k in enumerate(seq):
        w = weights[k[0]]
        if k in resident:
            hit_b += w
            cur[k] = nxt[i]
            heapq.heappush(heap, (-nxt[i], k))
            continue
        miss_b += w
        if len(resident) < capacity:
            resident.add(k)
        else:
            while heap:
                neg, victim = heapq.heappop(heap)
                if victim in resident and cur.get(victim) == -neg:
                    break
            resident.discard(victim)
            cur.pop(victim, None)
            resident.add(k)
        cur[k] = nxt[i]
        heapq.heappush(heap, (-nxt[i], k))
    return hit_b / (hit_b + miss_b), miss_b


def reuse_distances(seq) -> list[int]:
    last: dict = {}
    out: list[int] = []
    for i, key in enumerate(seq):
        out.append(i - last[key] - 1 if key in last else -1)
        last[key] = i
    return out


def pct(vals: list[int], q: float) -> int:
    if not vals:
        return 0
    return vals[min(len(vals) - 1, int(q * len(vals)))]


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(prog="trace_locality")
    ap.add_argument("--trace", required=True)
    ap.add_argument("--model", action="append", default=[])
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--disk-gbps", default="2.2,1.56,1.0")
    ap.add_argument("--json", default="")
    ap.add_argument("--caps-gb", default="",
                    help="comma list of byte budgets, e.g. 6.0,7.0,8.0")
    args = ap.parse_args()

    GB = 1e9
    disk = [float(x) for x in args.disk_gbps.split(",")]
    trace = load_trace(args.trace, args.top_k)
    blocks = sorted(trace)
    seq, steps = interleave(trace)
    n_objects = len(set(seq))

    eb = (expert_bytes_per_block(args.model) if args.model
          else {b: 2_046_000 for b in blocks})
    eb = {b: float(eb.get(b, next(iter(eb.values())))) for b in blocks}

    bytes_per_token = sum(args.top_k * eb[b] for b in blocks)
    mean_expert = sum(eb[b] for b in blocks) / len(blocks)

    print("=" * 78)
    print("trace   : %s" % args.trace)
    print("blocks=%d steps=%d top_k=%d requests=%d unique objects=%d"
          % (len(blocks), steps, args.top_k, len(seq), n_objects))
    print("routed bytes/token at 100%% miss = %.4f GB   (this is W)"
          % (bytes_per_token / GB))
    print("expert size: min %.3f MB  max %.3f MB  mean %.3f MB"
          % (min(eb.values()) / 1e6, max(eb.values()) / 1e6, mean_expert / 1e6))
    print("=" * 78)

    if args.caps_gb:
        print("\n-- GLOBAL LRU with a byte budget (the slab), real execution order --")
        print("%8s %8s %10s %11s %12s"
              % ("slab_GB", "hit%", "obj/token", "MB/token", "bytes/token"))
        for gb in [float(x) for x in args.caps_gb.split(",")]:
            hr, miss_b, miss_n = lru_hits_bytes(seq, gb * 1e9, eb)
            bpt = miss_b / steps
            print("%8.2f %8.2f %10.2f %11.1f %12.0f"
                  % (gb, 100.0 * hr, miss_n / steps, bpt / 1e6, bpt))

    print("\n-- per-block: unique experts, and static top-K coverage --")
    print("blk  uniq  top64  top128 top256 top384")
    for b in blocks:
        flat = [e for st in trace[b] for e in st]
        freq: dict = defaultdict(int)
        for e in flat:
            freq[e] += 1
        ranked = sorted(freq.values(), reverse=True)
        tot = len(flat)
        cov = [sum(ranked[:k]) / tot for k in (64, 128, 256, 384)]
        print("%3d  %4d  %.3f  %.3f  %.3f  %.3f"
              % (b, len(set(flat)), cov[0], cov[1], cov[2], cov[3]))

    rd = reuse_distances(seq)
    cold = sum(1 for d in rd if d < 0)
    warm = sorted(d for d in rd if d >= 0)
    print("\n-- reuse distance --")
    print("cold (first touch): %d / %d = %.1f%%"
          % (cold, len(rd), 100.0 * cold / len(rd)))
    print("warm " + "  ".join("p%d=%d" % (int(q * 100), pct(warm, q))
                              for q in (0.25, 0.5, 0.75, 0.9, 0.99)))

    print("\n-- expert bank: %.2f GB, %d objects --"
          % (sum(eb[b] * 512 for b in blocks) / GB, n_objects))

    fracs = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00)
    caps = [max(1, int(n_objects * f)) for f in fracs]

    print("\n-- GLOBAL LRU over all blocks, in real execution order --")
    print("cap_GB  fill   hit%%   fetch_MB/tok   " +
          "  ".join("tok/s@%.2f" % d for d in disk))
    rows = []
    for C in caps:
        hr, miss_b = lru_hits(seq, C, eb)
        fpt = miss_b / steps
        tps = [d * GB / fpt for d in disk]
        print("%6.2f  %4.1f%%  %6.2f  %10.1f   %s"
              % (C * mean_expert / GB, 100.0 * C / n_objects, 100.0 * hr,
                 fpt / 1e6, "  ".join("%9.1f" % t for t in tps)))
        rows.append({"capacity_objects": C, "capacity_gb": C * mean_expert / GB,
                     "fill": C / n_objects, "hit_rate": hr,
                     "fetch_bytes_per_token": fpt,
                     "tokens_per_s": dict(zip([str(d) for d in disk], tps))})

    print("\n-- offline static top-K (most-frequent experts preloaded) --")
    print("cap_GB  fill   hit%%   fetch_MB/tok")
    for C in caps:
        hr, miss_b = static_topk_hits(seq, C, eb)
        print("%6.2f  %4.1f%%  %6.2f  %10.1f"
              % (C * mean_expert / GB, 100.0 * C / n_objects, 100.0 * hr,
                 miss_b / steps / 1e6))

    print("\n-- PARTITIONED (each block gets total/48 bytes) vs GLOBAL (same total) --")
    print("total_GB  global_hit%%  part_hit%%   delta    part_fetch_MB/tok")
    tot_part = sum(eb[b] * args.top_k * steps for b in blocks)
    for C in (int(n_objects * f) for f in (0.15, 0.20, 0.30, 0.40, 0.50, 0.60)):
        g_hr, _ = lru_hits(seq, C, eb)
        per = max(1, C // len(blocks))
        mb = 0.0
        for b in blocks:
            sub = [(b, e) for st in trace[b] for e in st]
            _, miss_b = lru_hits(sub, per, eb)
            mb += miss_b
        part_hr = 1 - mb / tot_part
        print("%8.2f  %9.2f  %9.2f  %+7.2f  %14.1f"
              % (C * mean_expert / GB, 100.0 * g_hr, 100.0 * part_hr,
                 100.0 * (part_hr - g_hr), mb / steps / 1e6))

    warmup = max(1, steps // 5)
    print("\n-- DEPLOYABLE static top-K (rank on first %d steps, score on the rest) --"
          % warmup)
    print("cap_GB  fill   hit%%(scored)   fetch_MB/tok")
    for C in caps:
        freq: dict = defaultdict(int)
        for s in range(warmup):
            for b in blocks:
                for e in trace[b][s]:
                    freq[(b, e)] += 1
        top = set(sorted(freq, key=lambda k: -freq[k])[:C])
        hit_b = miss_b = 0.0
        n_scored = 0
        for s in range(warmup, steps):
            for b in blocks:
                for e in trace[b][s]:
                    w = eb[b]
                    n_scored += 1
                    if (b, e) in top:
                        hit_b += w
                    else:
                        miss_b += w
        print("%6.2f  %4.1f%%  %6.2f  %10.1f"
              % (C * mean_expert / GB, 100.0 * C / n_objects,
                 100.0 * hit_b / (hit_b + miss_b), miss_b / (steps - warmup) / 1e6))

    print("\n-- batch / speculative-decode amplification (union of experts per layer"
          " per pass) --")
    print("batch  union/step(avg)  union_per_token  bytes/tok  bytes/tok ratio")
    base = None
    for B in (1, 2, 3, 4, 6, 8):
        n_passes = steps // B
        tot_union = 0
        for p in range(n_passes):
            for b in blocks:
                s: set = set()
                for j in range(B):
                    s.update(trace[b][p * B + j])
                tot_union += len(s)
        avg_union = tot_union / n_passes
        per_tok = avg_union / B
        bytes_tok = avg_union * mean_expert / B
        if base is None:
            base = bytes_tok
        print("%5d  %12.2f  %15.2f  %8.1f MB  %.3f"
              % (B, avg_union, per_tok, bytes_tok / 1e6, bytes_tok / base))

    print("\n-- Belady (offline optimal, upper bound) --")
    for C in (int(n_objects * f) for f in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)):
        hr, miss_b = belady_hits(seq, C, eb)
        print("  fill %4.1f%% (%5.2f GB)  hit %6.2f%%  fetch %7.1f MB/tok  %6.1f tok/s@2.2"
              % (100.0 * C / n_objects, C * mean_expert / GB, 100.0 * hr,
                 miss_b / steps / 1e6, 2.2 * GB / (miss_b / steps)))

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"trace": args.trace, "blocks": len(blocks), "steps": steps,
                       "objects": n_objects, "bytes_per_token": bytes_per_token,
                       "lru": rows}, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
