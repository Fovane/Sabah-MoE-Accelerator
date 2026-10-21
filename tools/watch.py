#!/usr/bin/env python
"""Live progress for the two things that take time here: a CUDA build and a run.

    python tools/watch.py                 # refresh every 2 s, Ctrl-C to stop
    python tools/watch.py --once          # one snapshot
    python tools/watch.py --snapshots 3 --interval 5

A single nvcc translation unit reports nothing, so the build panel shows the
honest signals: the stage files nvcc is writing (the front end writes .ii, the
device back end writes .gpu/.ptx), their growth rate, and the CPU minutes the
compiler has burned. The ETA is an estimate against a measured baseline, and it
is labelled as one.

The run panel counts decoded steps in the active result directory's driver.log
and extrapolates from the last 50 step times.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "third_party" / "llama.cpp" / "build"
RESULTS = ROOT / "results" / "phase0"
NINJA = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ninja.exe")

# Measured: the same TU needed ~50 CPU minutes while it was built for two
# architectures (compute_120a + compute_61). With one it should land near half.
TU_BASELINE_S = 25 * 60.0


def ps_json(script: str):
    try:
        # The process command lines contain non-ASCII user names; force UTF-8 on
        # both sides so a decode error cannot silently return "nothing running".
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "[Console]::OutputEncoding=[Text.Encoding]::UTF8; " + script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=25)
        raw = out.stdout.strip()
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


def procs(names: str):
    wanted = [n if "." in n else n + ".exe" for n in names.split(",")]
    val = ps_json(
        "Get-CimInstance Win32_Process | Where-Object { $_.Name -in @(" +
        ",".join(f"'{n}'" for n in wanted) + ") } | "
        "Select-Object ProcessId,Name,CommandLine,KernelModeTime,UserModeTime,"
        "CreationDate,WorkingSetSize | ConvertTo-Json -Compress")
    if not val:
        return []
    return val if isinstance(val, list) else [val]


def cpu_s(p) -> float:
    # Win32_Process reports 100 ns units
    return ((p.get("KernelModeTime") or 0) + (p.get("UserModeTime") or 0)) / 1e7


def started(p) -> float | None:
    # /Date(1697812345678)/
    m = re.search(r"(\d{10,})", str(p.get("CreationDate") or ""))
    return float(m.group(1)) / 1000.0 if m else None


def human_s(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 60}:{s % 60:02d}"


def bar(frac: float, width: int = 24) -> str:
    frac = max(0.0, min(1.0, frac))
    n = int(round(frac * width))
    return "[" + "#" * n + "." * (width - n) + "]"


def build_panel(state: dict) -> list[str] | None:
    cs = [p for p in procs("cicc,nvcc,ninja,cmake,cl,link")]
    if not cs:
        obj = BUILD / "ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/ggml-cuda.cu.obj"
        exe = BUILD / "bin/llama-sabah-diag.exe"
        if exe.exists():
            return [f"build   idle, {exe.name} {time.strftime('%H:%M:%S', time.localtime(exe.stat().st_mtime))}"]
        return None

    cicc = next((p for p in cs if p["Name"].lower() == "cicc.exe"), None)
    if cicc is None:
        return [f"build   active: {', '.join(sorted({p['Name'] for p in cs}))}"]

    cmd = cicc.get("CommandLine") or ""
    stage = {}
    for key, pat in (("ii", r'"([^"]*cpp1\.ii)"'),
                     ("gpu", r'"([^"]*cudafe1\.gpu)"'),
                     ("c", r'"([^"]*cudafe1\.c)"'),
                     ("ptx", r'"(C:[^"]*\.ptx)"')):
        m = re.search(pat, cmd)
        if m:
            for path in (m.group(1), m.group(1).replace("/", "\\")):
                try:
                    stage[key] = os.path.getsize(path)
                    break
                except OSError:
                    stage[key] = 0

    t0 = started(cicc)
    elapsed = time.time() - t0 if t0 else 0.0
    cpu = cpu_s(cicc)

    hist = state.setdefault("tu_hist", deque(maxlen=30))
    gpu_bytes = stage.get("gpu", 0)
    now = time.time()
    hist.append((now, gpu_bytes))
    rate = 0.0
    if len(hist) > 1 and now > hist[0][0]:
        rate = (gpu_bytes - hist[0][1]) / (now - hist[0][0])
    frac = min(0.999, cpu / TU_BASELINE_S)
    if cpu < TU_BASELINE_S:
        head = f"build   {bar(frac)} {frac*100:4.1f}%  nvcc TU (estimate against a 25 min baseline)"
    else:
        head = (f"build   nvcc TU  cpu {cpu/60:.1f} min  "
                f"(25 min baseline exceeded, no reliable percentage)")
    lines = [
        head,
        f"        cicc {cicc['ProcessId']}  elapsed {human_s(elapsed)}  "
        f"rss {(cicc.get('WorkingSetSize') or 0)/2**20:,.0f} MiB  "
        f"{'busy' if state.get('busy') else ''}",
        "        stages: " + "  ".join(
            f"{k}={v/2**20:.1f} MB" for k, v in stage.items()) +
        (f"   gpu growth {rate/2**20:.2f} MB/s" if rate else ""),
    ]
    return lines


def ninja_steps(state: dict) -> list[str] | None:
    now = time.time()
    if now - state.get("steps_at", 0) < 15 and "steps" in state:
        return state["steps"]
    steps = None
    try:
        out = subprocess.run([NINJA, "-n", "llama-sabah-diag"], cwd=str(BUILD),
                             capture_output=True, text=True, timeout=90)
        lines = [l for l in (out.stdout or "").splitlines() if l.startswith("[")]
        steps = lines
    except Exception:
        steps = None
    state["steps_at"] = now
    state["steps"] = steps
    return steps


def run_panel(state: dict) -> list[str] | None:
    dg = [p for p in procs("llama-sabah-diag") if p["Name"].lower().startswith("llama-sabah-diag")]
    if not dg:
        return None
    d = dg[0]
    cmd = d.get("CommandLine") or ""
    m = re.search(r"-n\s+(\d+)", cmd)
    target = int(m.group(1)) if m else None

    dirs = [p for p in RESULTS.iterdir() if p.is_dir()] if RESULTS.exists() else []
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    log = None
    for cand in dirs:
        for lg in (cand / "driver.log",):
            if lg.exists():
                log = lg
                break
        if log:
            break
    if log is None:
        return [f"run     llama-sabah-diag {d['ProcessId']} (no log yet)"]

    steps = 0
    times: deque[float] = state.setdefault("run_times", deque(maxlen=50))
    try:
        with log.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                mm = re.search(r"[\w.-]+ step (\d+) .*?([\d.]+) ms", line)
                if mm and int(mm.group(1)) >= 1:
                    steps += 1
                    times.append(float(mm.group(2)))
    except OSError:
        pass
    mean = (sum(times) / len(times) / 1000.0) if times else None
    frac = (steps / target) if target else 0.0
    eta = (target - steps) * mean if (target and mean) else None
    head = f"run     {log.parent.name}: {steps}"
    if target:
        head += f"/{target} steps  {bar(frac, 20)} {frac*100:3.0f}%"
    if mean:
        head += f"  last50 {mean*1000:.1f} ms  {1/mean:.2f} tok/s"
    if eta:
        head += f"  eta {human_s(eta)}"
    return [head, f"        log {log}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--snapshots", type=int, default=0,
                    help="print N snapshots then exit (for pasting into a report)")
    args = ap.parse_args()

    state: dict = {}
    n = 0
    while True:
        parts: list[str] = []
        b = build_panel(state)
        if b:
            parts += b
            st = ninja_steps(state)
            if st:
                parts.append(f"        {len(st)} step(s) left in llama-sabah-diag")
                for line in st[:2]:
                    parts.append(f"        {line}")
        r = run_panel(state)
        if r:
            parts += r
        if not parts:
            parts = ["idle: no build, no run"]
        stamp = time.strftime("%H:%M:%S")
        if not args.once and not args.snapshots:
            sys.stdout.write("\x1b[2J\x1b[H")
        print(f"{stamp}")
        print("\n".join(parts))
        sys.stdout.flush()

        n += 1
        if args.once:
            return 0
        if args.snapshots and n >= args.snapshots:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
