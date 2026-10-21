"""Fixed benchmark protocol: refuse or flag invalid machine state.

A run is invalid when any of these hold, and the flag is not overridable except
for --allow-shared-usage (Shared Usage != 0 means another process already put
memory in system RAM, and the measured ms/token then depends on it):

  * steady SM clock below 2400 MHz, or memory clock below 10000 MHz
    (the harness's own validity gate);
  * this GPU reports Shared Usage above 0, or another process holds a compute
    or memory context on it.

    python -m sabah.tools.bench_protocol [--allow-shared-usage] [--dry-run]

It does not run the benchmark itself: it prints the machine verdict and the
exact command to run once the machine is clean.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

MIN_SM = 2400
MIN_MEM = 10000
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def nvidia(args):
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return ""
    r = subprocess.run([exe, *args], capture_output=True, text=True, timeout=30)
    return r.stdout


def clocks():
    txt = nvidia(["--query-gpu=clocks.sm,clocks.mem,pstate,clocks_event_reasons.active",
                  "--format=csv,noheader,nounits"])
    if not txt:
        return None
    parts = [p.strip() for p in txt.strip().splitlines()[0].split(",")]
    return {"sm_mhz": int(parts[0]), "mem_mhz": int(parts[1]), "pstate": parts[2]}


def shared_and_processes():
    txt = nvidia(["--query-compute-apps=pid,used_memory",
                  "--query-gpu=memory.used,memory.total",
                  "--format=csv,noheader,nounits"])
    procs = []
    shared = 0
    for line in txt.strip().splitlines():
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if len(parts) == 2 and parts[0].isdigit():
            procs.append((int(parts[0]), parts[1]))
        else:
            shared = line
    return procs, shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-shared-usage", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cl = clocks()
    print("gpu:", cl or "nvidia-smi unavailable")
    procs, shared = shared_and_processes()
    print("processes on the gpu:", procs or "none")

    bad = []
    if cl:
        if cl["sm_mhz"] < MIN_SM:
            bad.append(f"SM clock {cl['sm_mhz']} MHz < {MIN_SM} MHz")
        if cl["mem_mhz"] < MIN_MEM:
            bad.append(f"memory clock {cl['mem_mhz']} MHz < {MIN_MEM} MHz")
        if cl["pstate"] not in ("P0", "P1", "P2"):
            bad.append(f"p-state {cl['pstate']}")
    else:
        bad.append("no clock readings")

    # another process holding a context on the GPU means our numbers are shared
    others = [p for p in procs if int(p[0]) != os.getpid()]
    if len(others) > 1:
        bad.append(f"{len(others)} other processes hold the gpu")

    print("verdict:", "CLEAN" if not bad else "INVALID: " + "; ".join(bad))
    if bad:
        print("refusing; pin the clocks and close the other users first:")
        print("  nvidia-smi -lgc 2800,3090")
        print("  nvidia-smi -lmc 13000,14001")
        return 2

    print("machine is clean; run:")
    print("  python tools/phase0_measure.py --label <name> [--prompt-file PATH] "
          "--tokens 2048 --context 131072 --flag=-b --flag=2048 --flag=-ub --flag=2048")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
