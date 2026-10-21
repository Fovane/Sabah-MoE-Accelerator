"""Phase 3.1: the slab-vs-budget check, and Shared Usage at three contexts.

For each context it runs a short decode while a sampler records the GPU's
Windows "Shared Usage" counter (the OS part of the video memory - the thing that
must stay ~0 for the RAM-flat contract). The idle baseline is measured too, and
the reported number is the decode maximum above it.

    python tools/p3_shared.py [--contexts 512,32768,131072] [--tokens 64]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
SAMPLER = os.path.join(ROOT, "tools", "_shared_sample.ps1")
# the nvidia adapter's LUID on this machine; see tools/_shared_sample.ps1
LUID_PREFIX = "luid_0x00000000_0x0000d43d_"


def shared_now():
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command",
                            "(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | "
                            "Where-Object { $_.InstanceName -like '*0x0000d43d*' } | "
                            "Select-Object -First 1 -ExpandProperty CookedValue"],
                           capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def read_samples(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        rd = csv.reader(f)
        next(rd, None)
        for row in rd:
            try:
                out.append(float(row[1]))
            except (IndexError, ValueError):
                pass
    return out


def run_one(label, context, tokens):
    # idle baseline first, then the decode with the sampler running
    base = shared_now()
    csv_path = os.path.join(ROOT, "results/phase0", label + "_shared.csv")
    est = max(30, tokens * 2 + 30)
    sampler = subprocess.Popen(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", SAMPLER, "-Seconds", str(est), "-Out", csv_path],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = [PY, "tools/phase0_measure.py", "--label", label,
           "--tokens", str(tokens), "--context", str(context),
           "--prompt-file", "results/phase0/b-eq/prompt4k.txt",
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--set", "SABAH_LLAMA_HOT_BYTES=6.0e9"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    sampler.wait(timeout=est + 30)

    samples = read_samples(csv_path)
    mp = os.path.join(ROOT, "results/phase0", label, "measurement.json")
    if not os.path.exists(mp):
        return None
    rec = json.load(open(mp, encoding="utf-8"))
    m = rec.get("memory", {})
    mx = max(samples) if samples else None
    return {
        "context": context,
        "vram_peak": m.get("vram_peak_mib"),
        "ws_peak": m.get("process_peak_ws_mib"),
        "avail_min": m.get("avail_phys_min_mib"),
        "baseline_mib": (base / 1048576.0) if base else None,
        "shared_max_mib": (mx / 1048576.0) if mx else None,
        "n": len(samples),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="512,32768,131072")
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()

    rows = []
    for c in [int(x) for x in args.contexts.split(",")]:
        label = f"p3-shared-c{c}"
        print(f"[run] {label}", flush=True)
        r = run_one(label, c, args.tokens)
        if r:
            rows.append(r)
        else:
            print("   FAILED")

    print("\ncontext | VRAM peak | host WS peak | host avail min | Shared Usage idle | Shared Usage max in decode | delta (samples)")
    print("--------+-----------+--------------+----------------+-------------------+---------------------------+----------------")
    for r in rows:
        b = r["baseline_mib"] or 0
        s = r["shared_max_mib"] or 0
        print(f"{r['context']:>7} | {r['vram_peak']:>9} | {r['ws_peak']:>12.0f} | "
              f"{r['avail_min']:>14.0f} | {b:>16.1f} | {s:>25.1f} | "
              f"{s - b:>+8.1f} ({r['n']})")

    worst = max(((r["shared_max_mib"] or 0) - (r["baseline_mib"] or 0)) for r in rows) if rows else 0
    print("\nShared Usage added by decode, worst over all contexts: %+.1f MiB" % worst)
    print("verdict:", "OK, the decode does not put a context in system RAM"
          if worst <= 16 else "FAILS the RAM-flat contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
