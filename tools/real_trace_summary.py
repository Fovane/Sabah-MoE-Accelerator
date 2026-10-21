"""Summarize a real-trace run: per-regime ms, and the runtime's own counters.

Reads ``<run>/<item>/step_ms.f64`` (step 0 is prefill plus one decode) and the
``SABAH_DIAG`` line from ``<run>/run.log``.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import statistics
import struct


def read_f64(path: str) -> list[float]:
    raw = open(path, "rb").read()
    return list(struct.unpack("<%dd" % (len(raw) // 8), raw))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--skip", type=int, default=1,
                    help="leading steps per item to exclude (prefill)")
    args = ap.parse_args()

    items = sorted(d for d in glob.glob(os.path.join(args.run, "*"))
                   if os.path.isdir(d) and os.path.exists(os.path.join(d, "step_ms.f64")))
    all_ms: list[float] = []
    fill_ms = 0.0
    print("%-14s %6s %10s %10s %10s" % ("item", "steps", "prefill_ms", "med_ms", "p90_ms"))
    for d in items:
        ms = read_f64(os.path.join(d, "step_ms.f64"))
        if not ms:
            continue
        fill_ms += ms[0]
        steady = ms[args.skip:]
        if not steady:
            continue
        all_ms.extend(steady)
        s = sorted(steady)
        print("%-14s %6d %10.1f %10.1f %10.1f"
              % (os.path.basename(d), len(steady), ms[0],
                 statistics.median(steady), s[int(0.9 * (len(s) - 1))]))

    if all_ms:
        s = sorted(all_ms)
        med = statistics.median(all_ms)
        print("\n%d steady positions over %d items" % (len(all_ms), len(items)))
        print("median %.2f ms = %.2f tok/s   p10 %.1f  p90 %.1f  p99 %.1f"
              % (med, 1000.0 / med, s[int(0.1 * (len(s) - 1))],
                 s[int(0.9 * (len(s) - 1))], s[int(0.99 * (len(s) - 1))]))
        print("total prefill %.1f s" % (fill_ms / 1000.0))

    log = os.path.join(args.run, "run.log")
    if os.path.exists(log):
        txt = open(log, "r", errors="replace").read()
        for pat, label in ((r"SABAH_DIAG (\{.*?\})", "SABAH_DIAG"),
                           (r"SABAH_DISK (.*)", "SABAH_DISK")):
            m = re.search(pat, txt)
            if m:
                print("%s %s" % (label, m.group(1)[:600]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
