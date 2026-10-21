"""Phase 1.1: fusion decisions must be identical with SEG=0 and SEG=48.

Runs one short decode with SABAH_FUSE_TRACE=1 for both SEG settings and diffs
the per-node decisions with tools/fuse_diff.py.

    python tools/p1_fuse.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def run(label, seg, extra):
    env = dict(os.environ)
    env["SABAH_FUSE_TRACE"] = "1"
    out = os.path.join(ROOT, "results/phase0", label)
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "3",
           "--context", "2048", "--prompt", "Hello",
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--allow-low-clock"]
    cmd += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}"] if seg is not None else []
    cmd += extra
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    dl = os.path.join(out, "driver.log")
    return dl if os.path.exists(dl) else None


def main():
    logs = {}
    for label, seg, extra in (("f1-seg0", 0, []), ("f1-seg48", 48, [])):
        p = run(label, seg, extra)
        print(f"[run] {label}: {'log written' if p else 'NO LOG'}")
        if p:
            logs[label] = p

    if len(logs) < 2:
        return 1

    a = logs["f1-seg0"]
    b = logs["f1-seg48"]
    # keep only the trace lines so fuse_diff sees the same input shape
    print()
    r = subprocess.run([PY, "tools/fuse_diff.py", a, b], cwd=ROOT,
                       capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.stderr:
        print(r.stderr[-1000:])


if __name__ == "__main__":
    raise SystemExit(main())
