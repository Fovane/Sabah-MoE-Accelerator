"""Equivalence verification of the layer-group prefill fallback.

The fallback only groups a decode call whose whole batch is one ubatch
(n_tokens_all <= n_ubatch). Everything else runs the plain path, so it must be
byte-identical to the reference, and the single-ubatch case must stay identical
while the grouping is active (passes > 1 in the driver log).

    python tools/b_verify.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P650 = "results/phase0/b-eq/prompt650.txt"
P1888 = "results/phase0/b-eq/prompt1630.txt"

# label, prompt, ub, G, expect_group
RUNS = [
    ("bv-ref650-2ub", P650, 512, None, None),
    ("bv-grp650-2ub", P650, 512, 4, False),
    ("bv-grp650-2ub-g24", P650, 512, 24, False),
    ("bv-ref1888-4ub", P1888, 512, None, None),
    ("bv-grp1888-4ub", P1888, 512, 4, False),
    ("bv-ref1888-1ub", P1888, 2048, None, None),
    ("bv-grp1888-1ub", P1888, 2048, 4, True),
    ("bv-grp1888-1ub-g24", P1888, 2048, 24, True),
]


def run(label, prompt, ub, g, seg48):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", prompt,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=" + str(ub),
           "--allow-low-clock"]
    if g is None:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    if not seg48:
        cmd += ["--set", "SABAH_CUDA_GRAPH_SEG=0"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(os.path.join(p, "logits.f32")):
        return None, (r.stdout or "")[-300:]
    blob = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        blob += open(os.path.join(p, f), "rb").read()
    log = ""
    dl = os.path.join(ROOT, "results/phase0", label, "driver.log")
    if os.path.exists(dl):
        for line in open(dl, errors="replace"):
            if "prefill group" in line:
                log += line.strip().split("prefill group")[-1].strip()
                break
    return blob, log


def main():
    out = {}
    for label, prompt, ub, g, expect in RUNS:
        blob, log = run(label, prompt, ub, g, seg48=False)
        out[label] = (blob, log, expect)
        print(f"[run] {label}: {'ok' if blob else 'FAILED'}")
        if log:
            print(f"       {log}")

    print("\nlabel                  | reference      | logits | grouping active")
    print("-----------------------+----------------+--------+-----------------")
    REF = {
        "bv-grp650-2ub": ("bv-ref650-2ub", False),
        "bv-grp650-2ub-g24": ("bv-ref650-2ub", False),
        "bv-grp1888-4ub": ("bv-ref1888-4ub", False),
        "bv-grp1888-1ub": ("bv-ref1888-1ub", True),
        "bv-grp1888-1ub-g24": ("bv-ref1888-1ub", True),
    }
    for label, prompt, ub, g, expect in RUNS:
        blob, log, _ = out[label]
        if blob is None:
            print(f"{label:22s} | FAILED")
            continue
        active = log.split("passes=")[1].split()[0] != "1" if "passes=" in log else False
        if label.startswith("bv-ref"):
            print(f"{label:22s} | (reference)    |        | off")
            continue
        rlabel, expect_active = REF[label]
        same = "IDENTICAL" if out[rlabel][0] == blob else "DIFF"
        print(f"{label:22s} | {rlabel:14s} | {same} | "
              f"{'yes' if active else 'no'} (expected {'yes' if expect_active else 'no'})")


if __name__ == "__main__":
    raise SystemExit(main())
