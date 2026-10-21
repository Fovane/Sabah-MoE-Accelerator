"""Exercise the TILE3 guard fallback and prove it is bit-exact.

The model's expert tensors all fit the staged tree (k = 2560 and 640, i.e. 80
and 20 sub-blocks, against a 96-sub-block limit), so the fallback branch would
never run on real weights. `SABAH_LLAMA_TILE3_MAXNSUB` lowers the threshold so
the same graph takes `k_mul_mat_id` instead; the outputs must be byte-identical
because the fallback *is* the reference reduction.

    python tools/tile3_fallback_test.py

Runs three 128-token passes at -c 512 with pinned clocks:
  tile3     - default (TILE3 on, guard 96)          -> staged tree
  guard16   - TILE3 on, guard 16                    -> fallback for 80 and 20
  off       - TILE3=0                               -> reference kernel
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
MEASURE = os.path.join(HERE, "phase0_measure.py")


def run(label, extra):
    argv = [PY, MEASURE, "--label", label, "--tokens", "128",
            "--context", "1024", "--hot-bytes", "6.0e9", "--prompt", "Hello",
            "--set", "SABAH_LLAMA_EVENTS=1"]
    for kv in extra:
        argv += ["--set", kv]
    print("$", " ".join(argv))
    r = subprocess.run(argv, cwd=ROOT)
    if r.returncode != 0:
        print(f"{label}: measurement exit {r.returncode}")
    return r.returncode


def compare(a, b):
    argv = [PY, os.path.join(HERE, "compare_runs.py"),
            os.path.join(ROOT, "results", "phase0", a),
            os.path.join(ROOT, "results", "phase0", b)]
    return subprocess.run(argv, cwd=ROOT).returncode


def main():
    rc = 0
    rc |= run("fb-tile3", [])
    rc |= run("fb-guard16", ["SABAH_LLAMA_TILE3_MAXNSUB=16"])
    rc |= run("fb-off", ["SABAH_LLAMA_TILE3=0"])

    print("\n--- tile3 vs guard16 fallback ---")
    rc_a = compare("fb-tile3", "fb-guard16")
    print("\n--- tile3 vs reference kernel ---")
    rc_b = compare("fb-tile3", "fb-off")
    print("\n--- guard16 fallback vs reference kernel ---")
    rc_c = compare("fb-guard16", "fb-off")

    ev = os.path.join(ROOT, "results", "phase0", "fb-guard16", "driver.log")
    if os.path.exists(ev):
        for line in open(ev, "r", encoding="utf-8", errors="replace"):
            if line.startswith("SABAH_EV"):
                print("guard16 runtime:", line.strip())
                break

    ok = rc_a == 0 and rc_b == 0 and rc_c == 0
    print("\nFALLBACK TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
