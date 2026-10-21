"""Phase 1.3: the two missing acceptance cells, on the current build.

  kv640 : short prompt + 640 decodes at -c 2048 (the -c 512 variant cannot fit:
          any prompt plus 640 decodes exceeds n_ctx 512 and the hybrid memory
          does not shift). Byte-identical fed/argmax/logits, SEG 0 vs 48.
  c128k-B: a second prompt at -c 131072, 1024 decodes, SEG 0 vs 48.

    python tools/p1_cells.py
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
HELLO = None
PB = "results/phase0/b-eq/promptB.txt"


def run(label, prompt_file, tokens, context, seg, use_hello, slab=None):
    cmd = [PY, "tools/phase0_measure.py", "--label", label,
           "--tokens", str(tokens), "--context", str(context),
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--allow-low-clock"]
    if use_hello:
        cmd += ["--prompt", "Hello"]
        # without --prompt-file the harness writes the artifacts into the label dir
        data = os.path.join(ROOT, "results/phase0", label, "")
    else:
        cmd += ["--prompt-file", prompt_file]
        data = os.path.join(ROOT, "results/phase0", label, "main")
    if seg is not None:
        cmd += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}"]
    if slab:
        cmd += ["--set", f"SABAH_LLAMA_HOT_BYTES={slab}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = {}
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        q = os.path.join(data, f)
        out[f] = hashlib.sha256(open(q, "rb").read()).hexdigest()[:16] if os.path.exists(q) else "MISSING"
    exitc = ""
    for line in (r.stdout or "").splitlines():
        if line.startswith(label):
            exitc = line
    return out, exitc


def show(a, b):
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        print(f"  {f:10s} {a[f]}  {b[f]}  {'IDENTICAL' if a[f] == b[f] and a[f] != 'MISSING' else 'DIFF'}")


def main():
    print("[kv640] short prompt + 640 decodes, -c 2048, SEG 0 vs 48")
    a, ea = run("p1kv-ref", None, 640, 2048, 0, True)
    b, eb = run("p1kv-seg48", None, 640, 2048, 48, True)
    print("  ", ea)
    print("  ", eb)
    show(a, b)

    print("\n[c128k-B] second prompt, -c 131072, 1024 decodes, SEG 0 vs 48")
    a, ea = run("p1cb-ref", PB, 1024, 131072, 0, False)
    b, eb = run("p1cb-seg48", PB, 1024, 131072, 48, False)
    print("  ", ea)
    print("  ", eb)
    show(a, b)


if __name__ == "__main__":
    raise SystemExit(main())
