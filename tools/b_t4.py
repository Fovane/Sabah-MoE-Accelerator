"""T4: op-offload discriminator.

The harness base env sets GGML_OP_OFFLOAD_MIN_BATCH=1, which offloads ops by batch
size. If the group-path divergence disappears when that is disabled, the cause is a
kernel/backend assignment difference rather than a data difference.

    python tools/b_t4.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PROMPT = "results/phase0/b-eq/prompt1630.txt"

RUNS = [
    ("b24-ref", 0, None),
    ("b24-grp", 1, 4),
    ("b24-ref-nomoo", 0, None),   # second reference with offload disabled
    ("b24-grp-nomoo", 1, 4),
]

EXTRA = {
    "b24-ref": [],
    "b24-grp": [],
    "b24-ref-nomoo": ["--set", "GGML_OP_OFFLOAD_MIN_BATCH=1000000"],
    "b24-grp-nomoo": ["--set", "GGML_OP_OFFLOAD_MIN_BATCH=1000000"],
}


def run(label, group, g):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", PROMPT,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=512",
           "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock"]
    if group:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    cmd += EXTRA[label]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return os.path.exists(os.path.join(ROOT, "results/phase0", label, "main", "logits.f32"))


def blob(label):
    p = os.path.join(ROOT, "results/phase0", label, "main")
    out = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        out += open(os.path.join(p, f), "rb").read()
    return out


def main():
    for label, group, g in RUNS:
        print(f"[run] {label}")
        if not run(label, group, g):
            print(f"  {label} FAILED to produce artifacts")
    print()
    print("reference vs group (harness env, offload min batch = 1):")
    print("  ", "IDENTICAL" if blob("b24-ref") == blob("b24-grp") else "DIFF")
    print("reference vs group (op offload disabled):")
    print("  ", "IDENTICAL" if blob("b24-ref-nomoo") == blob("b24-grp-nomoo") else "DIFF")


if __name__ == "__main__":
    raise SystemExit(main())
