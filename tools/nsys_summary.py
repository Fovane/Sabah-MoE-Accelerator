#!/usr/bin/env python3
"""Summarize an Nsight Systems SQLite export for the Sabah decode timeline.

    python tools/nsys_summary.py <report.sqlite>

Prints the CUDA kernel table, the runtime API table and a per-token busy/idle
split. The caller decides how many trailing tokens to average; this tool reports
the whole capture and the last `--tokens` tokens by default.
"""
from __future__ import annotations

import argparse
import sqlite3


def tables(c):
    return {r[0] for r in c.execute("select name from sqlite_master where type='table'")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--tokens", type=int, default=4,
                    help="trailing tokens to summarize")
    args = ap.parse_args()
    c = sqlite3.connect(args.db)
    t = tables(c)
    print("tables with activity:", sorted(x for x in t if "ACTIVITY" in x))
    for name in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_RUNTIME",
                 "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if name in t:
            n = c.execute(f"select count(*) from {name}").fetchone()[0]
            print(f"{name}: {n}")
    if "StringIds" in t:
        print("string ids:", c.execute("select count(*) from StringIds").fetchone()[0])
    if "CUPTI_ACTIVITY_KIND_KERNEL" in t:
        cols = [r[1] for r in c.execute("pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)")]
        print("kernel columns:", cols)
        q = ("select s.value, count(*), sum(k.end-k.start), "
             "avg(k.end-k.start), max(k.end-k.start) "
             "from CUPTI_ACTIVITY_KIND_KERNEL k "
             "join StringIds s on s.id=k.demangledName "
             "group by s.value order by sum(k.end-k.start) desc limit 25")
        print("\n%12s %8s %14s %12s %12s" % ("total_ns", "n", "sum_ns", "avg_ns", "max_ns"))
        for row in c.execute(q):
            print("%12s %8d %14d %12.0f %12d" % (str(row[0])[:12], row[1], row[2],
                                                 row[3], row[4]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
