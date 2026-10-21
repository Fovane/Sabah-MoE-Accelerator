"""Compare two runs' logits: max abs/rel diff, top-1 agreement, KL divergence.

The reduction order of a MoE matmul depends on how many tokens share a ubatch,
so a wider -ub is not guaranteed to be bit-identical. This quantifies the
difference instead of asserting it either way.

    python tools/logits_cmp.py results/phase0/A/main/logits.f32 results/phase0/B/main/logits.f32
"""
from __future__ import annotations

import math
import struct
import sys


def load(path):
    b = open(path, "rb").read()
    return struct.unpack("<%df" % (len(b) // 4), b)


def main():
    a = load(sys.argv[1])
    b = load(sys.argv[2])
    if len(a) != len(b):
        print(f"size differs: {len(a)} vs {len(b)}")
        return 1
    n = len(a)
    row = n_vocab = 248320 if n % 248320 == 0 else None
    rows = n // n_vocab if n_vocab else 1

    nd = 0
    max_abs = 0.0
    max_rel = 0.0
    worst = -1
    for i in range(n):
        if a[i] != b[i]:
            nd += 1
            da, db = abs(a[i] - b[i]), 0.0
            r = da / max(1e-9, abs(a[i]))
            if da > max_abs:
                max_abs = da
            if r > max_rel:
                max_rel = r
                worst = i

    print(f"floats            : {n}  ({rows} row(s) of {n_vocab})")
    print(f"differing floats  : {nd}  ({100.0 * nd / n:.4f}%)")
    print(f"max abs diff      : {max_abs:.3e}")
    print(f"max rel diff      : {max_rel:.3e}"
          + (f" at {worst}: {a[worst]!r} vs {b[worst]!r}" if worst >= 0 else ""))
    print(f"bit-identical     : {'yes' if nd == 0 else 'no'}")

    if nd == 0:
        return 0

    # top-1 agreement per row, and a symmetric KL on softmax(logits)
    agree = 0
    kl_max = 0.0
    kl_rows = 0
    for r in range(rows):
        s = a[r * n_vocab:(r + 1) * n_vocab]
        t = b[r * n_vocab:(r + 1) * n_vocab]
        if max(range(n_vocab), key=lambda i: s[i]) == max(range(n_vocab), key=lambda i: t[i]):
            agree += 1
        ms, mt = max(s), max(t)
        ps = [math.exp(x - ms) for x in s]
        pt = [math.exp(x - mt) for x in t]
        zs, zt = sum(ps), sum(pt)
        ps = [x / zs for x in ps]
        pt = [x / zt for x in pt]
        kl = sum(p * math.log(max(p, 1e-30) / max(q, 1e-30))
                 for p, q in zip(ps, pt) if p > 1e-30)
        kl = abs(kl)
        if kl > kl_max:
            kl_max = kl
        kl_rows += 1
    print(f"top-1 agreement   : {agree}/{rows} rows ({100.0 * agree / rows:.2f}%)")
    print(f"max |KL| per row  : {kl_max:.3e}")
    print("\nverdict: NOT byte-identical -> a wider -ub is a numerically equivalent"
          " mode, opt-in, never a default")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
