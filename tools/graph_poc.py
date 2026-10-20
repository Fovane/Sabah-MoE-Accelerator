"""CUDA-graph split PoC: measure the segmented path against the eager path.

Runs the same decode three times with pinned clocks:

  eager   - SABAH_CUDA_GRAPH_SEG unset (every node submitted one by one)
  layer1  - SABAH_CUDA_GRAPH_SEG=1  (the first layer's 3 segments captured)
  all     - SABAH_CUDA_GRAPH_SEG=48 (all 48 layers, 144 Sabah calls)

and byte-compares the logits of the captured runs against the eager run. The
SABAH_SEG lines the backend prints at exit give the host cost split (per-launch,
per-eager-call, capture).

    python tools/graph_poc.py --tokens 256 --context 2048 --hot-bytes 9.8e9
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
MEASURE = os.path.join(HERE, "phase0_measure.py")


def run(label, seg, args):
    argv = [PY, MEASURE, "--label", label, "--tokens", str(args.tokens),
            "--context", str(args.context), "--hot-bytes", args.hot_bytes,
            "--prompt", args.prompt, "--allow-low-clock", "--gpu-mem"]
    for kv in args.set:
        argv += ["--set", kv]
    if seg:
        argv += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}",
                 "--set", "SABAH_CUDA_GRAPH_SEG_STATS=1"]
    print("$", " ".join(argv[1:] + ([] if seg else ["(eager)"])))
    subprocess.run(argv, cwd=ROOT)
    path = os.path.join(ROOT, "results", "phase0", label, "measurement.json")
    rec = json.load(open(path, "r", encoding="utf-8"))
    segstats = None
    log = os.path.join(ROOT, "results", "phase0", label, "driver.log")
    if os.path.exists(log):
        for line in open(log, "r", encoding="utf-8", errors="replace"):
            if line.startswith("SABAH_SEG"):
                segstats = line.strip()
    reg = rec.get("regimes") or {}
    return {
        "label": label,
        "seg": seg,
        "steady_ms": (reg.get("steady") or {}).get("median_ms"),
        "steady_mean_ms": (reg.get("steady") or {}).get("mean_ms"),
        "warm_ms": (reg.get("warmup") or {}).get("median_ms"),
        "fill_s": (reg.get("fill_ms") or 0) / 1000.0,
        "tok_s": rec.get("steady_tok_per_s"),
        "misses": rec.get("misses"),
        "decodes": rec.get("n_decodes"),
        "bytes_per_token": (rec.get("bytes_fetched") or 0) / max(rec.get("n_decodes") or 1, 1),
        "segstats": segstats,
        "clock_valid": rec.get("clock_valid"),
        "dedicated_mib": (rec.get("gpu_memory") or {}).get("dedicated_peak_mib"),
        "shared_mib": (rec.get("gpu_memory") or {}).get("shared_peak_mib"),
    }


def compare(a, b):
    argv = [PY, os.path.join(HERE, "compare_runs.py"),
            os.path.join(ROOT, "results", "phase0", a),
            os.path.join(ROOT, "results", "phase0", b)]
    return subprocess.run(argv, cwd=ROOT).returncode


def parse_seg(line):
    if not line:
        return {}
    out = {}
    for m in re.finditer(r"(\w+)=([\d.]+)", line):
        out[m.group(1)] = float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--context", type=int, default=2048)
    ap.add_argument("--hot-bytes", default="9.8e9")
    ap.add_argument("--prompt", default=("Write a Python function that merges "
                                         "two sorted lists and explain its complexity."))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", default="poc")
    args = ap.parse_args()

    rows = []
    rows.append(run(f"graph-{args.tag}-eager", "", args))
    rows.append(run(f"graph-{args.tag}-layer1", "1", args))
    rows.append(run(f"graph-{args.tag}-all", "48", args))

    ok_a = compare(f"graph-{args.tag}-eager", f"graph-{args.tag}-layer1")
    ok_b = compare(f"graph-{args.tag}-eager", f"graph-{args.tag}-all")

    print("\n" + "=" * 100)
    print(f"graph PoC: -c {args.context}, {args.tokens} tokens, {args.hot_bytes} slab")
    print(f"{'config':>10} {'steady ms':>10} {'tok/s':>7} {'warm ms':>8} "
          f"{'MB/tok':>7} {'ded MiB':>8} {'shared':>7}")
    print("-" * 100)
    for r in rows:
        print(f"{r['label'].split('-')[-1]:>10} {r['steady_ms'] or 0:>10.2f} "
              f"{r['tok_s'] or 0:>7.2f} {r['warm_ms'] or 0:>8.1f} "
              f"{r['bytes_per_token']/1e6:>7.1f} {r['dedicated_mib'] or 0:>8.0f} "
              f"{r['shared_mib'] or 0:>7.0f}")
    print("=" * 100)

    eager = rows[0]
    for r in rows[1:]:
        if r["steady_ms"] and eager["steady_ms"]:
            saved = eager["steady_ms"] - r["steady_ms"]
            print(f"{r['label'].split('-')[-1]}: {saved:+.2f} ms/token "
                  f"({100.0 * saved / eager['steady_ms']:+.1f}%)")
        if r["segstats"]:
            print("  " + r["segstats"])
    print("\nlogits: layer1 vs eager:",
          "BIT-EXACT" if ok_a == 0 else "DRIFT",
          "| all vs eager:", "BIT-EXACT" if ok_b == 0 else "DRIFT")

    out = os.path.join(ROOT, "results", "phase0", f"graph-{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "rows": rows,
                   "bitexact_layer1": ok_a == 0, "bitexact_all": ok_b == 0},
                  f, indent=2)
        f.write("\n")
    print("wrote", out)
    return 0 if ok_a == 0 and ok_b == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
