"""Where does the GPU idle in a decode token?

    python tools/nsys_gaps.py <report.sqlite> [--tokens N] [--min-us 1]

Builds the union of all kernel intervals in the trailing N tokens and reports
every uncovered interval (a true GPU idle gap), with a duration histogram and
the kernel names on both sides. Submission-bound gaps are a few microseconds and
numbered in the thousands per token; I/O-bound gaps are hundreds of microseconds
and numbered in the tens.
"""
from __future__ import annotations

import argparse
import bisect
import sqlite3

NS = 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--min-us", type=float, default=1.0)
    args = ap.parse_args()

    c = sqlite3.connect(args.db)
    c.text_factory = bytes
    rows = c.execute(
        "select k.start, k.end, s.value "
        "from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.demangledName "
        "order by k.start").fetchall()
    if not rows:
        print("no kernels")
        return 1
    rows = [(int(a), int(b), d.decode("utf-8", "replace") if d else "?")
            for a, b, d in rows]

    # a decode token is roughly 1/median-token-duration wide; find the last token
    # boundary from the gaps instead: use the trailing window by kernel count.
    last = rows[-1][1]
    # estimate kernels per token from the total span and the known 3.6k/token
    n_kern = len(rows)
    # take a window whose idle+busy is about args.tokens tokens: 58.6 ms each
    win_start = last - int(args.tokens * 58.6e6)
    win = [r for r in rows if r[0] >= win_start]
    if not win:
        win = rows
    busy = sum(b - a for a, b, _ in win)
    span = win[-1][1] - win[0][0]

    # union coverage and gaps
    gaps = []
    cur_end = win[0][0]
    cur_name = win[0][2]
    for a, b, name in win:
        if a > cur_end:
            gaps.append((cur_end, a, cur_name, name))
        cur_end = max(cur_end, b)
        cur_name = name

    idle = sum(g[1] - g[0] for g in gaps)
    min_ns = int(args.min_us * 1000)
    big = [g for g in gaps if g[1] - g[0] >= min_ns]
    print("window: %d kernels, %.2f ms, busy %.2f ms (%.1f%%), idle %.2f ms"
          % (len(win), span / 1e6, busy / 1e6, 100.0 * busy / span, idle / 1e6))
    print("gaps >= %.1f us: %d, total %.2f ms" % (args.min_us, len(big),
                                                   sum(g[1] - g[0] for g in big) / 1e6))

    buckets = [(0, 5), (5, 20), (20, 100), (100, 500), (500, 2000), (2000, 10**9)]
    print("\n%12s %8s %12s %12s" % ("duration", "count", "total_ms", "mean_us"))
    for lo, hi in buckets:
        sel = [g for g in big if lo * 1000 <= g[1] - g[0] < hi * 1000]
        if not sel:
            continue
        tot = sum(g[1] - g[0] for g in sel) / 1e6
        print("%7d-%-5d %8d %12.2f %12.1f"
              % (lo, hi, len(sel), tot, tot * 1000.0 / len(sel)))

    print("\ntop gaps:")
    for a, b, before, after in sorted(big, key=lambda g: -(g[1] - g[0]))[:12]:
        print("  %8.1f us  after %-40s before %s"
              % ((b - a) / 1000.0, before[:40], after[:40]))

    # runtime API: how much of the idle sits inside cudaStreamSynchronize
    try:
        syncs = c.execute(
            "select r.start, r.end, s.value from CUPTI_ACTIVITY_KIND_RUNTIME r "
            "join StringIds s on s.id = r.nameId "
            "where s.value like '%Synchronize%'").fetchall()
        syncs = [(int(a), int(b)) for a, b, _ in syncs if int(b) > win[0][0]]
        sync_ns = sum(min(b, last) - max(a, win[0][0]) for a, b in syncs)
        print("\n%d synchronizing runtime calls in window, %.2f ms inside them"
              % (len(syncs), sync_ns / 1e6))
    except Exception as exc:  # pragma: no cover - schema drift
        print("runtime table unavailable: %s" % exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
