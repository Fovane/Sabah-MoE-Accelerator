"""Step 2 and 3 of the group-prefill equivalence investigation.

T0: the diverging case (1888-token prompt, -ub 512, G=4) with SABAH_CUDA_GRAPH_SEG=0
    on both sides.
T3: two prompt lengths (777 tok = 2 ubatches, 1888 tok = 4 ubatches) x G in
    {48, 24, 12, 4}, SEG=0 on both sides, comparing final logits byte-exactly.

    python tools/b_step23.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
BASE = [PY, "tools/phase0_measure.py"]

CFG = [
    # (label, prompt file, tokens, G)
    ("b23-refA", "results/phase0/b-eq/prompt650.txt", 1, None),
    ("b23-refB", "results/phase0/b-eq/prompt1630.txt", 1, None),
    ("b23-A-g48", "results/phase0/b-eq/prompt650.txt", 1, 48),
    ("b23-A-g24", "results/phase0/b-eq/prompt650.txt", 1, 24),
    ("b23-A-g12", "results/phase0/b-eq/prompt650.txt", 1, 12),
    ("b23-A-g4", "results/phase0/b-eq/prompt650.txt", 1, 4),
    ("b23-B-g48", "results/phase0/b-eq/prompt1630.txt", 1, 48),
    ("b23-B-g24", "results/phase0/b-eq/prompt1630.txt", 1, 24),
    ("b23-B-g12", "results/phase0/b-eq/prompt1630.txt", 1, 12),
    ("b23-B-g4", "results/phase0/b-eq/prompt1630.txt", 1, 4),
]


def run(label, prompt, tokens, g):
    extra = []
    if g is not None:
        extra += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    else:
        extra += ["--set", "SABAH_PREFILL_GROUP=0"]

    out = os.path.join("results/phase0", label)
    cmd = BASE + ["--label", label, "--tokens", str(tokens), "--context", "4096",
                  "--prompt-file", prompt, "--flag=-b", "--flag=2048",
                  "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock"] + extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    tail = (r.stdout or "")[-400:]
    ok = os.path.exists(os.path.join(out, "main", "argmax.i32"))
    return ok, tail


def final(name):
    p = os.path.join("results/phase0", name, "main")
    blob = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        blob += open(os.path.join(p, f), "rb").read()
    return blob


def main():
    results = {}
    for label, prompt, tokens, g in CFG:
        ok, tail = run(label, prompt, tokens, g)
        print(f"[run] {label} g={g} ok={ok}")
        if not ok:
            print(tail)
            continue
        results[label] = final(label)

    print()
    print("label          | n_bytes | vs reference")
    print("---------------+---------+-------------")
    for label in results:
        ref = "b23-refA" if label.startswith("b23-A") else "b23-refB"
        if label.startswith("b23-ref"):
            print(f"{label:14s} | {len(results[label]):7d} | (reference)")
        else:
            same = results[label] == results[ref]
            print(f"{label:14s} | {len(results[label]):7d} | {'IDENTICAL' if same else 'DIFF'}")


if __name__ == "__main__":
    raise SystemExit(main())
