"""Compare two runs' outputs with a row offset, for restored-state exactness.

A run that starts from a saved state has no logits row for the prompt's last
token (the state holds memory, not logits), so its row k is the continuing run's
row k+1. This aligns the rows and reports byte-identity over the overlap.

    python tools/cmp_rows.py refdir grpdir [--offset 1] [--rows N]

Offsets are in units of rows (one row = one full logit vector). The tool refuses
silent partial matches: if the overlap is empty it says so.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys


def load(path):
    b = open(path, "rb").read()
    ty = "i" if path.endswith((".i32", ".i64")) else "f"
    fmt = "<%d%s" % (len(b) // struct.calcsize(ty), ty)
    return list(struct.unpack(fmt, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--offset", type=int, default=1,
                    help="b's row i is a's row i+offset")
    ap.add_argument("--rows", type=int, default=0, help="limit the row count")
    args = ap.parse_args()

    ok = True
    for name in ("logits.f32", "argmax.i32", "fed.i32"):
        pa = os.path.join(args.a, name)
        pb = os.path.join(args.b, name)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print("%-12s MISSING on one side" % name)
            ok = False
            continue
        a = load(pa)
        b = load(pb)
        # rows: logits are n_vocab wide, the others one scalar per step
        width = 248320 if name == "logits.f32" else 1
        ra = len(a) // width
        rb = len(b) // width
        lo = max(0, 0 + args.offset)
        hi = min(ra, rb + args.offset)
        n = hi - lo
        if args.rows:
            n = min(n, args.rows)
        if n <= 0:
            print("%-12s EMPTY overlap (a=%d rows, b=%d rows, offset %d)" % (name, ra, rb, args.offset))
            ok = False
            continue
        sa = a[(lo) * width:(lo + n) * width]
        sb = b[(lo - args.offset) * width:(lo - args.offset + n) * width]
        same = sa == sb
        print("%-12s %s  (%d rows compared: a[%d:%d] vs b[%d:%d])" % (
            name, "BYTE-IDENTICAL" if same else "DIFF", n,
            lo, lo + n, lo - args.offset, lo - args.offset + n))
        if not same:
            ok = False
            for i in range(n):
                x = sa[i * width:(i + 1) * width]
                y = sb[i * width:(i + 1) * width]
                if x != y:
                    print("   first differing row %d (a step %d, b step %d)" % (i, lo + i, lo - args.offset + i))
                    if width == 1:
                        print("   a=%d b=%d" % (x[0], y[0]))
                    else:
                        for j in range(width):
                            if x[j] != y[j]:
                                print("   first differing float %d: %r vs %r" % (j, x[j], y[j]))
                                break
                    break
    print("VERDICT:", "BIT-EXACT" if ok else "DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
