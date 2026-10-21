"""Phase 2: long prompts through the ubatch size, no new algorithm.

One ~8K-token prompt is prefilled with -ub 512, 1024, 2048 and 4096. Per -ub it
reports the compute buffer, peak VRAM, host Shared Usage, disk bytes per chunk,
ms/token, and whether the decode after the prefill is byte-identical to the
-ub 512 run (reduction order can depend on batch size, so a non-identical run
is an opt-in "numerically equivalent" mode, never a default).

    python tools/p2_ub.py [--prompt PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(label, prompt, ub, tokens, context, slab, batch=2048):
    cmd = [PY, "tools/phase0_measure.py", "--label", label,
           "--tokens", str(tokens), "--context", str(context),
           "--prompt-file", prompt,
           "--flag=-b", "--flag=" + str(batch),
           "--flag=-ub", "--flag=" + str(ub),
           "--allow-low-clock",
           "--set", f"SABAH_LLAMA_HOT_BYTES={slab}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    d = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(os.path.join(d, "prefill_ms.f64")):
        return None, (r.stdout or "")[-500:]

    rec = json.load(open(os.path.join(ROOT, "results/phase0", label, "measurement.json")))
    dl = open(os.path.join(ROOT, "results/phase0", label, "driver.log"),
              errors="replace").read()

    cbuf = None
    for m in re.finditer(r"compute buffer size is\s+([0-9.]+)\s*MiB", dl):
        cbuf = float(m.group(1))
    shared = None
    for m in re.finditer(r"CUDA0 model buffer size\s*=\s*([0-9.]+)\s*MiB", dl):
        cbuf = cbuf or float(m.group(1))

    blob = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        blob += open(os.path.join(d, f), "rb").read()

    cl = rec.get("clocks") or {}
    st = cl.get("steady", {}) or {}
    return {
        "ub": ub, "slab": slab,
        "compute_buffer_mib": cbuf,
        "vram_peak_mib": max(rec.get("memory", {}).get("vram_mib_series") or [0]),
        "ws_peak_mib": rec.get("memory", {}).get("process_peak_ws_mib"),
        "avail_min_mib": rec.get("memory", {}).get("avail_phys_min_mib"),
        "prefill_s": rec["prefill"]["ms"] / 1000.0,
        "prefill_gb": rec["prefill"]["bytes"] / 1e9,
        "chunks": rec["prefill"]["chunks"],
        "chunk_gb": [round(b / 1e9, 1) for b in rec["prefill"]["chunk_bytes"]],
        "chunk_ms": [round(m / 1000.0, 1) for m in rec["prefill"]["chunk_ms"]],
        "prompt_tok": rec.get("prompt_tokens_count"),
        "ms_per_tok": rec["prefill"]["ms"] / max(1, rec.get("prompt_tokens_count") or 1),
        "hash": hashlib.sha256(blob).hexdigest()[:16],
        "logits": blob,
        "clock_valid": cl.get("clock_valid"),
        "clock_reason": cl.get("reason"),
        "sm_median": ((st.get("sm_mhz") or {}).get("median")),
        "mem_median": ((st.get("mem_mhz") or {}).get("median")),
        "pstate": st.get("dominant_pstate"),
        "exit": rec.get("exit_code"),
    }, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="results/phase0/b-eq/prompt8k.txt")
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--context", type=int, default=20000)
    ap.add_argument("--slab", default="9.8e9")
    ap.add_argument("--ubs", default="512,1024,2048,4096")
    args = ap.parse_args()

    out = {}
    for ub in [int(x) for x in args.ubs.split(",")]:
        label = f"p2-ub{ub}"
        d, err = run(label, args.prompt, ub, args.tokens, args.context, args.slab)
        print(f"[run] {label}: {'ok' if d else 'FAILED'}")
        if d:
            out[ub] = d
        else:
            print(err)

    if not out:
        return 1
    base = min(out)

    print("\n-ub | compute buf MiB | VRAM peak | WS peak | avail min | prefill s | disk GB | "
          "chunks (s, GB) | decode vs -ub %d | ms/tok | clocks" % base)
    print("----+-----------------+-----------+---------+-----------+-----------+---------+"
          "---------------+-------------------+--------+--------")
    for ub in sorted(out):
        d = out[ub]
        same = "IDENTICAL" if d["logits"] == out[base]["logits"] else "DIFF"
        chunks = ",".join(f"{m}s/{g}GB" for m, g in zip(d["chunk_ms"], d["chunk_gb"]))
        print(f"{ub:4d}| {str(d['compute_buffer_mib']):>15} | {d['vram_peak_mib']:>9} | "
              f"{d['ws_peak_mib']:>7.0f} | {d['avail_min_mib']:>9.0f} | {d['prefill_s']:>9.1f} | "
              f"{d['prefill_gb']:>7.1f} | {chunks:>13} | {same:>17} | {d['ms_per_tok']:>6.1f} | "
              f"{'VALID' if d['clock_valid'] else 'INVALID'} SM {d['sm_median']} mem {d['mem_median']} {d['pstate']}")
    print("\nclock reason where INVALID:")
    for ub in sorted(out):
        d = out[ub]
        if not d["clock_valid"]:
            print(f"  -ub {ub}: {d['clock_reason']}")

    print("\nprompt tokens:", out[base]["prompt_tok"], " (same for all runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
