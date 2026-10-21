"""Collect the Phase 2 -ub sweep rows from the finished run directories.

    python tools/p2_rows.py [labels...]

For each label it prints the per-chunk bytes and ms, the clock verdict the
harness gave the run, and the byte-identity of the decode outputs against the
smallest -ub. No GPU work: this only reads what the runs wrote.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chunks(label):
    dl = os.path.join(ROOT, "results/phase0", label, "driver.log")
    out = []
    if not os.path.exists(dl):
        return out
    for line in open(dl, encoding="utf-8", errors="ignore"):
        m = re.search(r"prefill chunk \d+ off=\d+ n=(\d+)\s+([0-9.]+) ms bytes=(\d+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2)) / 1000.0, int(m.group(3))))
    return out


def row(label):
    mp = os.path.join(ROOT, "results/phase0", label, "measurement.json")
    d = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(mp) or not os.path.exists(os.path.join(d, "prefill_ms.f64")):
        return None
    rec = json.load(open(mp, encoding="utf-8"))
    cl = rec.get("clocks") or {}
    st = cl.get("steady", {}) or {}
    blob = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        blob += open(os.path.join(d, f), "rb").read()
    return {
        "label": label,
        "ub": int(label.split("ub")[-1]),
        "prompt": rec.get("prompt_tokens_count"),
        "chunks": chunks(label),
        "prefill_s": rec["prefill"]["ms"] / 1000.0,
        "prefill_gb": rec["prefill"]["bytes"] / 1e9,
        "vram": max(rec.get("memory", {}).get("vram_mib_series") or [0]),
        "ws": rec.get("memory", {}).get("process_peak_ws_mib"),
        "avail": rec.get("memory", {}).get("avail_phys_min_mib"),
        "clock_valid": cl.get("clock_valid"),
        "sm": st.get("sm_mhz", {}).get("median"),
        "mem_mhz": st.get("mem_mhz", {}).get("median"),
        "pstate": st.get("dominant_pstate"),
        "hash": hashlib.sha256(blob).hexdigest()[:16],
        "artifacts": blob,
        "ms_per_tok": rec["prefill"]["ms"] / max(1, rec.get("prompt_tokens_count") or 1),
    }


def main():
    labels = sys.argv[1:] or [f"p2-ub{u}" for u in (512, 1024, 2048, 4096)]
    rows = [r for r in (row(l) for l in labels) if r]
    if not rows:
        print("no finished runs")
        return 1
    base = min(rows, key=lambda r: r["ub"])

    print("per-chunk disk bytes (each chunk is 2048 tokens unless it is the tail):")
    for r in rows:
        cs = "  ".join(f"{n}tok {s:.1f}s {b/1e9:.1f}GB" for n, s, b in r["chunks"])
        print(f"  -ub {r['ub']:>5}: {cs}")

    print("\nsummary:")
    print("  -ub | prompt tok | prefill s | disk GB | GB/token | ms/token | VRAM peak | WS peak | clocks")
    for r in rows:
        gbt = r["prefill_gb"] * 1000 / max(1, r["prompt"])
        print(f"  {r['ub']:>5} | {r['prompt']:>10} | {r['prefill_s']:>9.1f} | {r['prefill_gb']:>7.1f} | "
              f"{gbt:>8.1f} | {r['ms_per_tok']:>8.1f} | {r['vram']:>9} | {r['ws']:>7.0f} | "
              f"{'VALID' if r['clock_valid'] else ('INVALID' if r['clock_valid'] is False else 'n/a')} "
              f"SM {r['sm']} mem {r['mem_mhz']} {r['pstate']}")

    print("\ndecode outputs vs -ub %d:" % base["ub"])
    for r in rows:
        if r is base:
            continue
        print(f"  -ub {r['ub']} vs -ub {base['ub']}: "
              f"{'IDENTICAL' if r['artifacts'] == base['artifacts'] else 'DIFF'} "
              f"(hash {r['hash']} vs {base['hash']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
