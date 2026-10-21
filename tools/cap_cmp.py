"""Compare two capture sets per (key, occ), showing which capture differs.

    python tools/cap_cmp.py results/phase0/A results/phase0/B [--filter substring]

The diag writes one record per capture into <run>/main/cap/index.jsonl, and the
capture bytes are appended into <run>/main/cap/<key>.bin at the recorded offset.
A graph tensor is captured twice per ubatch: once at build time (stale data) and
once when the node executes, so occ parity is reported.
"""
from __future__ import annotations

import argparse
import json
import os


def load_index(run):
    idx = {}
    for line in open(os.path.join(run, "main", "cap", "index.jsonl")):
        d = json.loads(line)
        idx.setdefault(d["key"], []).append(d)
    return idx


def read(run, rec):
    with open(os.path.join(run, "main", "cap", rec["key"] + ".bin"), "rb") as f:
        f.seek(rec["offset"])
        return f.read(rec["nbytes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--filter", default="")
    ap.add_argument("--max-row", type=int, default=0)
    args = ap.parse_args()

    ia = load_index(args.a)
    ib = load_index(args.b)

    keys = sorted(set(ia) | set(ib))
    n_diff = 0
    for key in keys:
        if args.filter and args.filter not in key:
            continue
        ra = ia.get(key, [])
        rb = ib.get(key, [])
        if len(ra) != len(rb):
            print(f"{key}: capture count {len(ra)} vs {len(rb)}")
            n_diff += 1
            continue
        for j, (x, y) in enumerate(zip(ra, rb)):
            if x["nbytes"] != y["nbytes"]:
                print(f"{key} occ {x['occ']}: size {x['nbytes']} vs {y['nbytes']}")
                n_diff += 1
                continue
            ba = read(args.a, x)
            bb = read(args.b, y)
            if ba == bb:
                continue
            n_diff += 1
            nf = x["nbytes"] // 4
            first = next(i for i in range(nf) if ba[4 * i:4 * i + 4] != bb[4 * i:4 * i + 4])
            row = first // x["ne"][0] if x["ne"][0] else first
            print(f"{key} occ {x['occ']} ({'build' if x['occ'] % 2 == 0 else 'compute'}) step {x['step']}: "
                  f"DIFF {x['nbytes']} bytes, ne {x['ne']}, first diff float {first} (row {row}), "
                  f"{nf} floats")
    print("mismatching captures:", n_diff)


if __name__ == "__main__":
    raise SystemExit(main())
