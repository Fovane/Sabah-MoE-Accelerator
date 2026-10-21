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
    """(context holders, shared-usage info).

    `nvidia-smi --query-compute-apps` lists every process holding a compute
    context, not only the ones computing, so on a Windows desktop that list is
    routinely in the double digits. It is a FLAG, not a refusal: the measured
    number then carries a caveat, and a refusal only applies when this process
    is not the one driving the load.
    """
    txt = nvidia(["--query-compute-apps=pid", "--format=csv,noheader,nounits"])
    pids = []
    for line in txt.splitlines():
        line = line.strip()
        if line.isdigit():
            pids.append(int(line))

    # Shared Usage is not in the query string (it is a per-process field), so
    # read the GPU's own memory split instead: system-shared > 0 means the OS
    # put a context in system RAM, which is exactly the case reported in
    # phase0b.
    shared = "unknown"
    mem = nvidia(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
    try:
        used, total = [int(x.strip()) for x in mem.strip().splitlines()[0].split(",")]
        shared = f"{used} MiB used of {total} MiB"
    except (ValueError, IndexError, AttributeError):
        pass
    return pids, shared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-shared-usage", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cl = clocks()
    print("gpu:", cl or "nvidia-smi unavailable")
    pids, shared = shared_and_processes()
    print("processes holding a compute context:", pids or "none")
    print("gpu memory:", shared)

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

    print("verdict:", "CLEAN" if not bad else "INVALID: " + "; ".join(bad))

    # flags, not refusals
    others = [p for p in pids if p != os.getpid()]
    if others:
        print(f"FLAG: {len(others)} other process(es) hold a compute context "
              f"({', '.join(str(p) for p in others[:8])}{' ...' if len(others) > 8 else ''}). "
              "WDDM may time-slice: the same decode has measured 88.9 vs 58 ms on this "
              "machine, so a timed number taken now carries that caveat.")
    if "unknown" not in shared and args.allow_shared_usage is False:
        print(f"FLAG: gpu memory split - {shared}. If the OS has put a context in "
              "system RAM, host RAM is not flat for that process, and the "
              "low-clock / shared caveat applies.")

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
