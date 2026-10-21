"""Phase 4, H1b: what changes when a 24-layer group graph is built?

One ubatch, G=24, SEG=0. The reference is the plain 48-layer graph. Each row
turns one knob and reports whether the final logits become byte-identical.

    python tools/b_h1b.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P = "results/phase0/b-eq/prompt1630.txt"

RUNS = [
    ("h1b-ref", None, []),
    ("h1b-base", 24, []),
    ("h1b-nooffload", 24, ["--set", "GGML_OP_OFFLOAD_MIN_BATCH=1000000"]),
    ("h1b-t1", 24, ["--flag", "-t", "1"]),
    ("h1b-noreuse", 24, ["--set", "LLAMA_GRAPH_REUSE_DISABLE=1"]),
    ("h1b-gpuonly", 24, ["--set", "GGML_CUDA_DISABLE_GRAPHS=1"]),
]


def run(label, g, extra):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", P,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock"]
    if g is None:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    cmd += extra
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(os.path.join(p, "logits.f32")):
        return None
    out = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        out += open(os.path.join(p, f), "rb").read()
    return out


def main():
    blobs = {}
    for label, g, extra in RUNS:
        print(f"[run] {label}", flush=True)
        blobs[label] = run(label, g, extra)
    print("\nconfig                 | logits vs reference")
    print("-----------------------+--------------------")
    for label, g, extra in RUNS:
        b = blobs[label]
        if b is None:
            print(f"{label:21s} | FAILED")
        elif label == "h1b-ref":
            print(f"{label:21s} | (reference)")
        else:
            print(f"{label:21s} | {'IDENTICAL' if b == blobs['h1b-ref'] else 'DIFF'}")


if __name__ == "__main__":
    raise SystemExit(main())
