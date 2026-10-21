"""Compare the *last* capture per (key, ne) between two runs.

The diag eval callback is asked (`ask=true`) for every node before it computes
the range, and asked again to dump the node it needs
(`ggml-backend.cpp:1836/1843/1856`). The pre-compute ask leaves a capture whose
bytes are whatever the buffer held before, so `occ 0` of a multi-capture key is
not a value: it must not be compared. This pairs the final capture per
(key, ne) instead.

    python tools/cap_last.py results/phase0/A results/phase0/B [--filter str]
"""
from __future__ import annotations

import argparse
import json
import os


def recs(run):
    out = {}
    order = []
    for line in open(os.path.join(run, "main", "cap", "index.jsonl")):
        d = json.loads(line)
        k = (d["key"], tuple(d["ne"]))
        out.setdefault(k, []).append(d)
        if d["key"] not in order:
            order.append(d["key"])
    return out, order


def read(run, rec):
    with open(os.path.join(run, "main", "cap", rec["key"] + ".bin"), "rb") as f:
        f.seek(rec["offset"])
        return f.read(rec["nbytes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--filter", default="")
    args = ap.parse_args()

    ia, order = recs(args.a)
    ib, _ = recs(args.b)
    print("key (reference capture order) | last-capture comparison")
    print("-------------------------------+------------------------")
    seen = set()
    for k in order:
        if k in seen or (args.filter and args.filter not in k):
            continue
        seen.add(k)
        va = ia.get((k, *() ), [])
        # group by ne
        nes = [n for n in ia if n[0] == k]
        for (key, ne) in sorted(nes, key=lambda n: ia[n][0]["offset"]):
            ra, rb = ia[(key, ne)], ib.get((key, ne))
            if not rb:
                print(f"{k:31s} | only in A ({len(ra)} captures)")
                continue
            if len(ra) != len(rb):
                print(f"{k:31s} | capture count {len(ra)} vs {len(rb)}")
                continue
            a = read(args.a, ra[-1])
            b = read(args.b, rb[-1])
            if a == b:
                print(f"{k:31s} | identical (n={len(ra)})")
            else:
                nf = len(a) // 4
                fi = next((i for i in range(nf) if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]), None)
                print(f"{k:31s} | DIFF ne {ne} first float {fi} of {nf} "
                      f"(occ {ra[-1]['occ']}, n captures {len(ra)})")


if __name__ == "__main__":
    raise SystemExit(main())
