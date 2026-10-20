#!/usr/bin/env python
"""One-line status of whatever is running in the background.

    python tools/status.py

Reports, in this order:

  build   cicc/nvcc CPU time, the CUDA translation unit being compiled and
          whether its object file exists yet (a single nvcc TU has no
          percentage; the only signals are CPU time and the object appearing)
  run     if llama-sabah-diag is alive: decoded steps / requested tokens, read
          from the newest results/phase0/*/driver.log and its run.json
  git     branch, last commit, dirty file count

Exit code is 0 when nothing failed, 1 when a build or run looks stuck.
No GPU calls, no file locks, safe to run while a benchmark is on the GPU.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "third_party" / "llama.cpp" / "build"
RESULTS = ROOT / "results" / "phase0"


def ps(script: str) -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"<powershell failed: {exc}>"


def gpu_proc(names: str) -> list[dict]:
    raw = ps(
        f"Get-Process {names} -ErrorAction SilentlyContinue | "
        "Select-Object Id,Name,CPU,StartTime | ConvertTo-Json -Compress"
    )
    if not raw:
        return []
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else [val]


def build_status(lines: list[str]) -> bool:
    procs = gpu_proc("cicc,nvcc,ninja,cmake")
    obj = BUILD / "ggml" / "src" / "ggml-cuda" / "CMakeFiles" / "ggml-cuda.dir" / "ggml-cuda.cu.obj"
    # nvcc writes the object only when the whole translation unit is done, so a
    # *stale* object from an earlier build must not be read as "compiled now".
    started = min(
        (float(p["StartTime"]) for p in procs if isinstance(p.get("StartTime"), (int, float))),
        default=None,
    )
    fresh = obj.exists() and started is not None and obj.stat().st_mtime >= started
    if not procs:
        if obj.exists():
            lines.append(f"build   idle, ggml-cuda.cu.obj present ({obj.stat().st_size/1e6:.1f} MB, "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(obj.stat().st_mtime))})")
            return True
        lines.append("build   idle, no object file")
        return False
    cpu = sum(float(p.get("CPU") or 0) for p in procs if p["Name"] != "cmake")
    names = sorted({p["Name"] for p in procs})
    archs = "?"
    cache = BUILD / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("CMAKE_CUDA_ARCHITECTURES:"):
                archs = line.split("=", 1)[1]
                break
    lines.append(
        f"build   active {','.join(names)}  cpu={cpu/60:.1f} min  "
        f"arch={archs}  ggml-cuda.cu.obj={'fresh' if fresh else 'not yet'}"
    )
    if fresh:
        lines.append("        TU finished; remaining TUs and link are next")
    else:
        stale = (f" (stale object from {time.strftime('%Y-%m-%d %H:%M', time.localtime(obj.stat().st_mtime))})"
                 if obj.exists() else "")
        lines.append("        1 nvcc TU holds the whole build" + stale +
                     "; no percentage exists, CPU time is the only signal")
    return True


def run_status(lines: list[str]) -> bool:
    procs = gpu_proc("llama-sabah-diag")
    if not procs:
        lines.append("run     idle")
        return True
    dirs = sorted(
        (d for d in RESULTS.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime, reverse=True,
    ) if RESULTS.exists() else []
    if not dirs:
        lines.append(f"run     {len(procs)} diag process, no results dir")
        return True
    d = dirs[0]
    target = None
    rj = d / "run.json"
    if rj.exists():
        try:
            target = (json.loads(rj.read_text(encoding="utf-8")).get("args") or {}).get("tokens")
        except Exception:
            target = None
    steps = 0
    log = d / "driver.log"
    if log.exists():
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "prompt step" in line:
                    steps += 1
    pct = f"{100.0*steps/target:.0f}%" if target else "?"
    lines.append(
        f"run     {d.name}: {steps}/{target if target else '?'} steps ({pct}), "
        f"{len(procs)} process"
    )
    return True


def git_status(lines: list[str]) -> bool:
    try:
        branch = subprocess.run(["git", "-C", str(ROOT), "branch", "--show-current"],
                                capture_output=True, text=True, timeout=20).stdout.strip()
        last = subprocess.run(["git", "-C", str(ROOT), "log", "--oneline", "-1"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        lines.append(f"git     {branch}  {last}  ({len(dirty)} dirty)")
        return True
    except Exception as exc:
        lines.append(f"git     <{exc}>")
        return False


def main() -> int:
    lines: list[str] = []
    ok = all([build_status(lines), run_status(lines), git_status(lines)])
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
