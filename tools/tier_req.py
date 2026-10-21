"""Phase 2 requirement table: what each tier needs, from the measured numbers.

The disk is the wall in the deep-context regime, so a tier is reachable only if
the bytes per token it must fetch fit in the token's time budget:

    ms/token = compute_floor + (MB/token / disk_GBs) * 1000 / overlap

where `overlap` is the measured read_ms / wall ratio for that regime. Both the
compute floor and the overlap differ by an order of magnitude between the short
prompt and a deep KV, so they are inputs, not constants, and every row names the
run it came from.

    python tools/tier_req.py
"""
from __future__ import annotations

import argparse

TIERS = (10.0, 20.0, 30.0)

# label, ms/token, MB/token, misses/token, compute floor ms, overlap
REGIMES = [
    ("chat, 53-tok prompt, 128K ctx, slab 6.0e9, SEG=48", 30.36, 1.9, 3.0, 27.0, 3.06),
    ("chat, 53-tok prompt, 128K ctx, SEG=0", 42.74, 1.9, 3.0, 38.0, 3.06),
    ("deep KV ~30K, slab 6.0e9, tail 256", 309.95, 343.5, 504.0, 180.0, 1.10),
    ("coding trace, slab 6.0e9, tail 256", 236.38, 252.0, 373.5, 180.0, 1.10),
    ("coding trace, slab 9.8e9, tail 256", 223.97, 116.2, 174.0, 180.0, 1.10),
]

DISK_GBS = 1.05  # measured fresh-offset random read rate, MSI M371 (REPORT 15.2)


def disk_ms(mb, gbs, overlap):
    """Disk service time in ms for a MB-per-token budget.

    gbs is GB/s, so the rate in MB/s is gbs*1000 and
    time_ms = mb / (gbs*1000) * 1000 / overlap = mb / gbs / overlap.
    """
    return mb / gbs / overlap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disk-gbs", type=float, default=DISK_GBS)
    args = ap.parse_args()
    gbs = args.disk_gbs

    print("disk service rate: %.2f GB/s (measured, MSI M371 fresh offsets)\n" % gbs)

    print("tier | tok/s | token budget | disk budget no overlap | disk budget with 1.1x | with 3.06x")
    print("-----+-------+--------------+-----------------------+----------------------+------------")
    for t in TIERS:
        budget = 1000.0 / t
        print("%4.0f | %5.0f | %9.1f ms | %19.0f MB | %19.0f MB | %10.0f MB"
              % (t, t, budget,
                 budget * gbs,
                 budget * gbs * 1.1,
                 budget * gbs * 3.06))
    print("\n(the compute work has to fit inside the same budget; the measured")
    print("compute component is ~27 ms at short context and far larger at a deep")
    print("KV, so the budget is a joint bound, not a disk-only one.)")

    # tier A's disk budget at the conservative (measured) 1.1x overlap
    budget_mb_a = 1000.0 / min(TIERS) * gbs * 1.1

    print("\nMeasured regimes, and what each one's bytes/token buys it:")
    print("regime | ms/token | MB/token | misses | tier by measured time | disk ms alone | fits tier A?")
    print("-------+----------+----------+--------+----------------------+--------------+--------------")
    for label, ms, mb, miss, _floor, over in REGIMES:
        dm = disk_ms(mb, gbs, over)
        best = "below %d" % min(TIERS)
        for t in TIERS:
            if ms <= 1000.0 / t:
                best = str(int(t))
        fits = "yes" if mb <= budget_mb_a else "no"
        print("%s | %8.1f | %8.1f | %6.0f | %20s | %11.0f | %s"
              % (label, ms, mb, miss, best, dm, fits))
    print("\n(tier A needs <= %.0f MB/token at the measured 1.1x overlap; the disk column"
          % budget_mb_a)
    print("is the service time for that regime's bytes alone, so a regime whose disk ms")
    print("exceeds its time budget is disk-bound, and no policy can reach its tier")
    print("without fetching fewer bytes per token - which is what the replay measures.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
