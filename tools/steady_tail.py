"""Recover the steady-state window from a run's per-step times.

At `-c 131072` a 6 GB slab needs hundreds of decode steps before its working set
is resident; the first window is a fill (300 ms/step) and only the tail is steady
(44 ms/step).  `measurement.json:steady_ms_median` averages over a window that
starts too early and reports the fill, so the slab comparison has to be
recomputed from `step_ms.f64`.

The tail window found here is also used for the I/O columns: the same steps are
read from `step_bytes.i64` and `step_misses.i64`, so bytes and misses are tail
only, never whole run.

    .venv\\Scripts\\python.exe tools\\steady_tail.py results\\phase0\\item6b-*
"""
import glob
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


def _rows(label, a, nbytes=None, nmiss=None):
    if not len(a):
        return
    s = plateau(a)
    seg = a[s:]
    m = float(np.median(seg))
    extra = ""
    if nbytes is not None and len(nbytes) == len(a):
        extra += f" {np.median(nbytes[s:]) / (1 << 20):8.1f}"
    else:
        extra += f" {'-':>8s}"
    if nmiss is not None and len(nmiss) == len(a):
        extra += f" {np.median(nmiss[s:]):8.0f}"
    else:
        extra += f" {'-':>8s}"
    print(f"{label:32s} {len(a):5d} {s:9d} {m:9.2f} {1000.0/m:7.2f} "
          f"{seg.min():7.1f} {seg.max():7.1f}{extra}")


def _load(path, dtype):
    return np.fromfile(path, dtype=dtype) if os.path.exists(path) else None


def main(paths):
    print(f"{'run':32s} {'steps':>5s} {'fill_from':>9s} {'tail_ms':>9s} "
          f"{'tok/s':>7s} {'min':>7s} {'max':>7s} {'MiB/tok':>8s} {'miss/tok':>8s}")
    for pat in paths:
        for d in sorted(glob.glob(pat)):
            if not os.path.isdir(d):
                continue
            name = os.path.basename(d)
            # Manifest runs keep each prompt in its own <out>/<id>/ directory.
            dirs = [d] + [x for x in sorted(glob.glob(os.path.join(d, "*")))
                          if os.path.isdir(x)]
            for sub in dirs:
                p = os.path.join(sub, "step_ms.f64")
                if os.path.exists(p):
                    tag = name if sub == d else f"{name}/{os.path.basename(sub)}"
                    _rows(tag, _load(p, np.float64),
                          _load(os.path.join(sub, "step_bytes.i64"), np.int64),
                          _load(os.path.join(sub, "step_misses.i64"), np.int64))
                q = os.path.join(sub, "prefill_ms.f64")
                if os.path.exists(q):
                    tag = "prefill " + (name if sub == d else f"{name}/{os.path.basename(sub)}")
                    _rows(tag, _load(q, np.float64),
                          _load(os.path.join(sub, "prefill_bytes.i64"), np.int64))


if __name__ == "__main__":
    main(sys.argv[1:] or ["results/phase0/sweep-ctx128k-B-*"])
