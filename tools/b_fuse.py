"""Phase 4, H4: is the G=24 divergence a different fusion decision?

`ggml_cuda_try_fuse` refuses any pattern within 8 nodes of a Sabah node
(ggml-cuda.cu:3610-3614), so the node neighbourhood - which the layer-group
split changes - decides whether a pattern fuses. Run the G=24 case with fusion
on and off on both sides.

    python tools/b_fuse.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P = "results/phase0/b-eq/prompt1630.txt"


def run(label, g, fuse_off):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", P,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock"]
    if fuse_off:
        cmd += ["--set", "GGML_CUDA_DISABLE_FUSION=1"]
    if g is None:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(os.path.join(p, "logits.f32")):
        return None
    out = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        out += open(os.path.join(p, f), "rb").read()
    return out


def main():
    cfgs = [
        ("fu-ref-on", None, False),
        ("fu-grp24-on", 24, False),
        ("fu-ref-off", None, True),
        ("fu-grp24-off", 24, True),
        ("fu-grp12-off", 12, True),
        ("fu-grp4-off", 4, True),
    ]
    blobs = {}
    for label, g, off in cfgs:
        print(f"[run] {label}", flush=True)
        blobs[label] = run(label, g, off)

    print("\nconfig            | fusion | logits vs the matching reference")
    print("------------------+--------+---------------------------------")
    for label, g, off in cfgs:
        b = blobs[label]
        ref = "fu-ref-off" if off else "fu-ref-on"
        if b is None:
            print(f"{label:17s} | {'off' if off else 'on':>6} | FAILED")
        elif label.startswith("fu-ref"):
            print(f"{label:17s} | {'off' if off else 'on':>6} | (reference)")
        else:
            print(f"{label:17s} | {'off' if off else 'on':>6} | "
                  f"{'IDENTICAL' if b == blobs[ref] else 'DIFF'}")


if __name__ == "__main__":
    raise SystemExit(main())
