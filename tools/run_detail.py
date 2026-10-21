"""Per-regime detail for one finished run, from the harness's own records.

    python tools/run_detail.py results/phase0/<label>

Reads step_ms.f64 / step_bytes.i64 / step_misses.i64 and measurement.json and
prints the three regimes with min/median/max, so a number is never quoted as one
value.
"""
from __future__ import annotations

import json
import os
import statistics
import struct
import sys


def f64(path):
    b = open(path, "rb").read()
    return list(struct.unpack("<%dd" % (len(b) // 8), b))


def i64(path):
    b = open(path, "rb").read()
    return list(struct.unpack("<%dq" % (len(b) // 8), b))


def stats(v):
    if not v:
        return "n/a"
    return (f"n={len(v)} min={min(v):.2f} med={statistics.median(v):.2f} "
            f"max={max(v):.2f} mean={statistics.fmean(v):.2f}")


def main():
    label = sys.argv[1].rstrip("/\\")
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "results/phase0", label)
    d = os.path.join(root, "main")
    mp = os.path.join(root, "measurement.json")
    if not os.path.exists(mp):
        print("no measurement.json for", label)
        return 1
    rec = json.load(open(mp, encoding="utf-8"))

    ms = f64(os.path.join(d, "step_ms.f64")) if os.path.exists(os.path.join(d, "step_ms.f64")) else []
    bts = i64(os.path.join(d, "step_bytes.i64")) if os.path.exists(os.path.join(d, "step_bytes.i64")) else []
    mis = i64(os.path.join(d, "step_misses.i64")) if os.path.exists(os.path.join(d, "step_misses.i64")) else []

    print("label      :", label)
    print("exit       :", rec.get("exit_code"), " wall_s:", round(rec.get("wall_s", 0), 1))
    print("prompt tok :", rec.get("prompt_tokens_count"))
    pre = rec.get("prefill")
    if pre:
        print("prefill    : %.1f s, %d chunks, %.1f GB, %.2f ms/token"
              % (pre["ms"] / 1000, pre["chunks"], pre["bytes"] / 1e9,
                 pre["ms"] / max(1, rec.get("prompt_tokens_count") or 1)))
    else:
        print("prefill    : none (this run started from a saved state)")

    warm = slice(0, min(128, len(ms)))
    tail = slice(max(0, len(ms) - 256), len(ms))
    for name, sl in (("warm-up (first 128)", warm), ("steady (last 256)", tail)):
        print(f"\n{name}")
        print("  ms/token  :", stats(ms[sl]))
        print("  MB/token  :", stats([b / 1e6 for b in bts[sl]]))
        print("  misses/tok:", stats(mis[sl]))

    if ms:
        print("\nall decode: ms/token", stats(ms))
        print("all decode: tok/s   %.2f (1/median)" % (1000.0 / statistics.median(ms)))

    cl = rec.get("clocks") or {}
    st = cl.get("steady", {}) or {}
    sm = (st.get("sm_mhz") or {})
    print("\nclocks     : steady SM min/med/max %s/%s/%s, mem %s, %s -> %s"
          % (sm.get("min"), sm.get("median"), sm.get("max"),
             (st.get("mem_mhz") or {}).get("median"), st.get("dominant_pstate"),
             "VALID" if (sm.get("median") or 0) >= 2400 else "INVALID"))
    m = rec.get("memory", {})
    print("memory     : VRAM peak %s MiB, host WS peak %.0f MiB, host avail min %.0f MiB"
          % (m.get("vram_peak_mib"), m.get("process_peak_ws_mib") or 0,
             m.get("avail_phys_min_mib") or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
