"""Recover the steady-state window from a run's per-step times.

At `-c 131072` a 6 GB slab needs ~380 decode steps before its working set is
resident; the first window is a fill (300 ms/step) and only the tail is steady
(44 ms/step).  `measurement.json:steady_ms_median` averages over a window that
starts too early and reports the fill, so item 3's slab comparison has to be
recomputed from `step_ms.f64`.

    .venv\\Scripts\\python.exe tools\\steady_tail.py results\\phase0\\sweep-ctx128k-B-*
"""
import glob
import json
import os
import sys

import numpy as np


def plateau(a, window=32, tol=1.10):
    """First index from which a 32-step window stays within tol of the final
    window's median: the end of the fill and the start of steady state."""
    if len(a) < 2 * window:
        return 0
    ref = np.median(a[-window:])
    for i in range(len(a) - 2 * window, -1, -1):
        if np.median(a[i:i + window]) > ref * tol:
            return min(i + window, len(a))
    return 0


def main(paths):
    print(f"{'run':34s} {'steps':>5s} {'fill_from':>9s} {'steady_ms':>9s} "
          f"{'tok/s':>7s} {'min':>7s} {'max':>7s}")
    for pat in paths:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            p = os.path.join(d, "step_ms.f64")
            if not os.path.exists(p):
                continue
            a = np.fromfile(p, dtype=np.float64)
            if not len(a):
                continue
            s = plateau(a)
            seg = a[s:]
            m = float(np.median(seg))
            print(f"{os.path.basename(d):34s} {len(a):5d} {s:9d} {m:9.2f} "
                  f"{1000.0/m:7.2f} {seg.min():7.1f} {seg.max():7.1f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/phase0/sweep-ctx128k-B-*"])
