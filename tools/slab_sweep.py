"""Pinned-clock slab sweep with per-run clock and GPU-memory records.

    python tools/slab_sweep.py --context 131072 --tokens 1024 \
        --slabs 6.0e9,7.0e9,8.0e9,9.8e9 --n 3 --tag ctx128

Refuses to run unless a short burn shows the card at boost, because a sweep
taken in the low-clock state measures the clock, not the slab. Every run is a
full `tools/phase0_measure.py` record (clocks, dedicated/shared GPU memory,
misses, bytes per token), and the summary prints min/median/max per slab.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sabah.tools import clocks  # noqa: E402

PY = sys.executable
MEASURE = os.path.join(HERE, "phase0_measure.py")
BURN = os.path.join(HERE, "gpu_burn.exe")


def clock_guard(seconds=6):
    """Burn briefly and require boost clocks before the sweep."""
    if not os.path.exists(BURN):
        print("clock guard: gpu_burn.exe missing (run tools\\_build_burn.bat)")
        return False
    p = subprocess.Popen([BURN, str(seconds)], stdout=subprocess.DEVNULL)
    sampler = clocks.ClockSampler(interval=0.5)
    sampler.start()
    p.wait()
    rec = sampler.stop().record()
    s = rec.get("steady") or {}
    sm = (s.get("sm_mhz") or {}).get("median")
    mem = (s.get("mem_mhz") or {}).get("median")
    print(f"clock guard: SM {sm} MHz, mem {mem} MHz, {s.get('dominant_pstate')}")
    if not sm or sm < clocks.MIN_STEADY_SM_MHZ or not mem or mem < clocks.MIN_STEADY_MEM_MHZ:
        print("clock guard FAILED: pin the clocks first:\n"
              "  nvidia-smi -lgc 2800,3090\n"
              "  nvidia-smi -lmc 13000,14001")
        return False
    return True


def one_run(label, args, slab, rep):
    argv = [PY, MEASURE, "--label", label, "--tokens", str(args.tokens),
            "--context", str(args.context), "--hot-bytes", slab,
            "--gpu-mem", "--allow-low-clock"]
    if args.prompt:
        argv += ["--prompt", args.prompt]
    for kv in args.set:
        argv += ["--set", kv]
    t0 = time.time()
    r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    wall = time.time() - t0
    path = os.path.join(ROOT, "results", "phase0", label, "measurement.json")
    try:
        rec = json.load(open(path, "r", encoding="utf-8"))
    except (OSError, ValueError):
        print(f"  {label}: no record (exit {r.returncode})\n{r.stdout[-800:]}")
        return None
    reg = rec.get("regimes") or {}
    steady = (reg.get("steady") or {}).get("median_ms")
    warm = (reg.get("warmup") or {}).get("median_ms")
    calls = rec.get("calls") or 0
    decodes = rec.get("n_decodes") or 0
    out = {
        "label": label, "slab": slab, "rep": rep, "wall_s": round(wall, 1),
        "exit": rec.get("exit_code"),
        "fill_s": round((reg.get("fill_ms") or 0) / 1000.0, 2),
        "warm_ms": warm, "steady_ms": steady,
        "steady_tok_s": (1000.0 / steady) if steady else None,
        "bytes_per_token": (rec.get("bytes_fetched") or 0) / decodes if decodes else None,
        "misses_per_token": (rec.get("misses") or 0) / decodes if decodes else None,
        "hits": rec.get("hits"), "misses": rec.get("misses"),
        "calls_per_token": (calls / decodes) if decodes else None,
        "vram_peak_mib": (rec.get("memory") or {}).get("vram_peak_mib"),
        "dedicated_peak_mib": (rec.get("gpu_memory") or {}).get("dedicated_peak_mib"),
        "shared_peak_mib": (rec.get("gpu_memory") or {}).get("shared_peak_mib"),
        "resident_bytes": rec.get("resident_bytes"),
        "clock_valid": rec.get("clock_valid"),
        "clock_steady": (rec.get("clocks") or {}).get("steady"),
        "machine": rec.get("machine"),
    }
    print(f"  {label}: steady {steady} ms, {out['bytes_per_token']/1e6:.1f} MB/tok, "
          f"{out['misses_per_token']:.0f} misses/tok, "
          f"clocks {'ok' if out['clock_valid'] else 'INVALID'}")
    return out


def agg(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {"n": n, "min": vals[0], "median": med, "max": vals[-1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--slabs", default="6.0e9,7.0e9,8.0e9,9.8e9")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--tag", default="ctx128")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default="")
    ap.add_argument("--skip-guard", action="store_true")
    args = ap.parse_args()

    if not args.skip_guard and not clock_guard():
        return 2

    slabs = [s.strip() for s in args.slabs.split(",") if s.strip()]
    rows = []
    for slab in slabs:
        for rep in range(1, args.n + 1):
            label = f"sweep-{args.tag}-{slab.replace('.', 'p')}-r{rep}"
            r = one_run(label, args, slab, rep)
            if r:
                rows.append(r)

    print("\n" + "=" * 96)
    print(f"slab sweep: -c {args.context}, {args.tokens} tokens, N={args.n}, "
          f"pinned clocks")
    print(f"{'slab':>8} {'n':>2} {'steady ms (min/med/max)':>28} "
          f"{'tok/s med':>10} {'MB/tok':>8} {'miss/tok':>9} {'VRAM peak':>10} {'ded':>7} {'shared':>7}")
    print("-" * 96)
    summary = []
    for slab in slabs:
        sub = [r for r in rows if r["slab"] == slab]
        if not sub:
            continue
        ms = agg(sub, "steady_ms")
        bt = agg(sub, "bytes_per_token")
        mt = agg(sub, "misses_per_token")
        vr = agg(sub, "vram_peak_mib")
        dd = agg(sub, "dedicated_peak_mib")
        sh = agg(sub, "shared_peak_mib")
        tok = agg(sub, "steady_tok_s")
        s = {
            "slab": slab, "n": ms["n"] if ms else 0,
            "steady_ms": ms, "steady_tok_s": tok,
            "bytes_per_token": bt, "misses_per_token": mt,
            "vram_peak_mib": vr, "dedicated_peak_mib": dd,
            "shared_peak_mib": sh,
            "clock_valid_all": all(r.get("clock_valid") for r in sub),
            "rows": sub,
        }
        summary.append(s)
        if ms:
            print(f"{slab:>8} {ms['n']:>2} "
                  f"{ms['min']:>8.1f}/{ms['median']:>8.1f}/{ms['max']:>8.1f}     "
                  f"{tok['median'] if tok else 0:>10.2f} "
                  f"{(bt['median'] if bt else 0)/1e6:>8.1f} "
                  f"{mt['median'] if mt else 0:>9.0f} "
                  f"{vr['median'] if vr else 0:>10.0f} "
                  f"{dd['median'] if dd else 0:>7.0f} "
                  f"{sh['median'] if sh else 0:>7.0f}")
    print("=" * 96)
    if not all(s["clock_valid_all"] for s in summary):
        print("WARNING: at least one run had an invalid clock record")

    out = args.out or os.path.join(ROOT, "results", "phase0",
                                   f"sweep-{args.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"context": args.context, "tokens": args.tokens,
                   "n": args.n, "slabs": slabs, "summary": summary,
                   "generated": time.strftime("%Y-%m-%dT%H:%M:%S")}, f, indent=2)
        f.write("\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
