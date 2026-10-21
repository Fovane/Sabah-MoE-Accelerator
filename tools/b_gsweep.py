"""G sweep for the single-ubatch case, with and without the segmented CUDA graph.

    python tools/b_gsweep.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P = "results/phase0/b-eq/prompt1630.txt"

# label, ub, G, seg (0 or None for default 48)
RUNS = [
    ("bg-ref-seg0", 2048, None, 0),
    ("bg-ref-seg48", 2048, None, None),
    ("bg-g48-seg0", 2048, 48, 0),
    ("bg-g24-seg0", 2048, 24, 0),
    ("bg-g12-seg0", 2048, 12, 0),
    ("bg-g8-seg0", 2048, 8, 0),
    ("bg-g6-seg0", 2048, 6, 0),
    ("bg-g4-seg0", 2048, 4, 0),
    ("bg-g2-seg0", 2048, 2, 0),
    ("bg-g4-seg48", 2048, 4, None),
    ("bg-g24-seg48", 2048, 24, None),
]


def run(label, ub, g, seg):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", P,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=" + str(ub),
           "--allow-low-clock"]
    if seg is not None:
        cmd += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}"]
    if g is None:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return os.path.exists(os.path.join(ROOT, "results/phase0", label, "main", "logits.f32"))


def blob(label):
    p = os.path.join(ROOT, "results/phase0", label, "main")
    out = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        out += open(os.path.join(p, f), "rb").read()
    return out


def passes(label):
    dl = os.path.join(ROOT, "results/phase0", label, "driver.log")
    if not os.path.exists(dl):
        return "-"
    for line in open(dl, errors="replace"):
        if "prefill group" in line:
            return line.split("passes=")[1].split()[0]
    return "-"


def main():
    for label, ub, g, seg in RUNS:
        print(f"[run] {label}", flush=True)
        if not run(label, ub, g, seg):
            print(f"  {label} FAILED")

    print("\nlabel          | G  | SEG | passes | artifacts vs SEG-matched reference")
    print("---------------+----+-----+--------+-----------------------------------")
    for label, ub, g, seg in RUNS:
        if not os.path.exists(os.path.join(ROOT, "results/phase0", label, "main", "logits.f32")):
            continue
        ref = "bg-ref-seg48" if seg is None else "bg-ref-seg0"
        if label.startswith("bg-ref"):
            print(f"{label:14s} | {'-' if g is None else g:>2} | {'48' if seg is None else seg} | "
                  f"{passes(label):>6} | (reference)")
            continue
        same = blob(label) == blob(ref)
        print(f"{label:14s} | {g:>2} | {'48' if seg is None else seg} | {passes(label):>6} | "
              f"{'IDENTICAL' if same else 'DIFF'}")


if __name__ == "__main__":
    raise SystemExit(main())
