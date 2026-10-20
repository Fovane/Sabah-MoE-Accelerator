#!/usr/bin/env python3
"""Phase 0 measurement wrapper.

Runs the shipped diag driver with exactly the flags/env that `sabah run` uses,
samples VRAM and host memory while it runs, and writes one JSON record that
contains the numbers the phase report needs.

This is instrumentation only: it does not touch the runtime or the bridge.

    python tools/phase0_measure.py --label baseline-n256 --tokens 256
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sabah.tools import clocks, stream  # noqa: E402

DEFAULT_MODEL = (r"C:\Strata-data\models\IQ3_S"
                 r"\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
DEFAULT_DIAG = os.path.join(ROOT, "third_party", "llama.cpp", "build", "bin",
                            "llama-sabah-diag.exe")


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", wt.DWORD),
        ("dwMemoryLoad", wt.DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wt.DWORD),
        ("PageFaultCount", wt.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def avail_phys():
    m = MEMORYSTATUSEX()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys


def process_mem(pid):
    h = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0400, False, pid)  # QUERY_LIMITED
    if not h:
        return None
    try:
        c = PROCESS_MEMORY_COUNTERS_EX()
        c.cb = ctypes.sizeof(c)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
                h, ctypes.byref(c), ctypes.sizeof(c)):
            return None
        return {"working_set": int(c.WorkingSetSize),
                "peak_working_set": int(c.PeakWorkingSetSize),
                "private": int(c.PrivateUsage)}
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def file_mb(path):
    try:
        return os.path.getsize(path) / (1 << 20)
    except OSError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--diag", default=os.environ.get("SABAH_DIAG_EXE", DEFAULT_DIAG))
    ap.add_argument("--rt", default=os.environ.get("SABAH_RT_LIB",
                    os.path.join(ROOT, "sabah", "runtime", "cuda", "sabah_rt.dll")))
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--hot-bytes", default="9.8e9")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--prompt-file", default="",
                    help="read the prompt from a file and pass it through the "
                         "diag's manifest (a 128K-token prompt does not fit on "
                         "the Windows command line). Results land in "
                         "<out>/<id>/ and are read from there.")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results", "phase0"))
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--flag", action="append", default=[], metavar="ARG")
    ap.add_argument("--no-monitor", action="store_true")
    ap.add_argument("--allow-low-clock", action="store_true",
                    help="keep exit code 0 even if the steady SM clock was below "
                         "2400 MHz (the run is still recorded as invalid)")
    ap.add_argument("--gpu-mem", action="store_true",
                    help="sample this process's Dedicated/Shared GPU memory from "
                         "the Windows GPU performance counters")
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.label)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "driver.log")
    json_path = os.path.join(out_dir, "measurement.json")
    data_dir = out_dir
    manifest = ""
    prompt_text = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8", errors="replace") as fh:
            prompt_text = fh.read()
        manifest = os.path.join(out_dir, "manifest.jsonl")
        with open(manifest, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"id": "main", "text": prompt_text, "raw": True}) + "\n")
        data_dir = os.path.join(out_dir, "main")

    env = dict(os.environ)
    env.update(stream.BASE_ENV)
    env["SABAH_RT_LIB"] = os.path.abspath(args.rt)
    env["SABAH_LLAMA_HOT_BYTES"] = str(args.hot_bytes)
    env["SABAH_DIAG_OUT"] = out_dir
    env["SABAH_LLAMA_PREFETCH"] = "0"
    env.pop("SABAH_LLAMA_STATUS_FILE", None)
    env.pop("SABAH_LLAMA_LAYER_MERGE", None)
    for item in args.set:
        k, v = item.split("=", 1)
        env[k] = v
    if manifest:
        env["SABAH_DIAG_MANIFEST"] = manifest
    else:
        env.pop("SABAH_DIAG_MANIFEST", None)

    argv = [os.path.abspath(args.diag), "-m", os.path.abspath(args.model)]
    argv += stream.BASE_FLAGS
    argv += ["-t", str(args.threads), "-c", str(args.context),
             "-n", str(args.tokens)]
    if not manifest:
        argv += ["-p", prompt_text]
    argv += list(args.flag)

    stream.clean_run_dir(out_dir)
    if manifest and os.path.isdir(data_dir):
        shutil.rmtree(data_dir, ignore_errors=True)
    samples = {"vram_mib": [], "avail_phys": [], "ws": []}
    stop = threading.Event()

    nv = None
    if not args.no_monitor:
        try:
            nv = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=memory.used",
                 "--format=csv,noheader,nounits", "-lms", "500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            nv = None

    def sample(pid):
        if nv is not None:
            line = nv.stdout.readline()
            try:
                samples["vram_mib"].append(int(line.strip()))
            except (TypeError, ValueError):
                pass
        samples["avail_phys"].append(avail_phys())
        m = process_mem(pid)
        if m:
            samples["ws"].append(m["working_set"])
            samples["peak_ws"] = max(samples.get("peak_ws", 0), m["peak_working_set"])

    t0 = time.time()
    clock_sampler = None
    if not args.no_monitor:
        clock_sampler = clocks.ClockSampler(interval=0.5)
        clock_sampler.start()
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        mem_probe = None
        gpu_mem = {}
        if args.gpu_mem:
            mem_csv = os.path.join(out_dir, "gpu_mem.csv")
            if os.path.exists(mem_csv):
                os.remove(mem_csv)
            mem_probe = subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
                 "-File", os.path.join(ROOT, "tools", "gpu_mem_sampler.ps1"),
                 "-ProcessId", str(proc.pid), "-Out", mem_csv],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            while proc.poll() is None:
                sample(proc.pid)
                stop.wait(0.25)
        finally:
            proc.wait()
            sample(proc.pid)
            if mem_probe is not None:
                mem_probe.terminate()
    if mem_probe is not None:
        rows = []
        try:
            for line in open(os.path.join(out_dir, "gpu_mem.csv"),
                             "r", encoding="ascii", errors="replace").read().splitlines()[1:]:
                t, d, s = line.split(",")
                rows.append((float(t), int(d), int(s)))
        except (OSError, ValueError):
            rows = []
        if rows:
            gpu_mem = {
                "samples": len(rows),
                "dedicated_peak_mib": max(r[1] for r in rows) / (1 << 20),
                "dedicated_last_mib": rows[-1][1] / (1 << 20),
                "shared_peak_mib": max(r[2] for r in rows) / (1 << 20),
                "shared_last_mib": rows[-1][2] / (1 << 20),
            }
    wall = time.time() - t0
    clock_rec = None
    if clock_sampler is not None:
        clock_sampler.stop()
    if nv is not None:
        nv.terminate()

    try:
        text = open(log_path, "r", encoding="utf-8", errors="replace").read()
    except OSError:
        text = ""
    rec = stream.parse_driver(text)
    rec["label"] = args.label
    rec["exit_code"] = proc.returncode
    rec["wall_s"] = wall
    rec["argv"] = argv
    if args.prompt_file:
        rec["prompt_file"] = os.path.abspath(args.prompt_file)
        rec["prompt_chars"] = len(prompt_text)
    rec["env"] = {k: env.get(k) for k in (
        "SABAH_LLAMA", "SABAH_LLAMA_HOT_BYTES", "SABAH_LLAMA_DIRECT_IO",
        "SABAH_LLAMA_IO_DEPTH", "SABAH_LLAMA_TRACE", "SABAH_LLAMA_PREFETCH",
        "GGML_OP_OFFLOAD_MIN_BATCH")}
    rec["env"].update({i.split("=", 1)[0]: i.split("=", 1)[1] for i in args.set})
    if clock_sampler is not None:
        clock_rec = clock_sampler.record(decode_ms=rec.get("decode_ms_sum"))
        rec["clocks"] = clock_rec
        rec["clock_valid"] = clock_rec.get("valid")
        rec["clock_invalid_reason"] = clock_rec.get("reason")
        clocks.clock_series_csv(clock_rec, os.path.join(out_dir, "clocks.csv"))
    rec["machine"] = clocks.machine_state()
    rec["memory"] = {        "vram_peak_mib": max(samples["vram_mib"]) if samples["vram_mib"] else None,
        "vram_samples": len(samples["vram_mib"]),
        "vram_mib_series": samples["vram_mib"],
        "avail_phys_min_mib": (min(samples["avail_phys"]) / (1 << 20)
                               if samples["avail_phys"] else None),
        "avail_phys_first_mib": (samples["avail_phys"][0] / (1 << 20)
                                 if samples["avail_phys"] else None),
        "avail_phys_last_mib": (samples["avail_phys"][-1] / (1 << 20)
                                if samples["avail_phys"] else None),
        "process_peak_ws_mib": (samples.get("peak_ws", 0) / (1 << 20)
                                if samples.get("peak_ws") else None),
        "process_ws_first_mib": (samples["ws"][0] / (1 << 20) if samples["ws"] else None),
        "process_ws_last_mib": (samples["ws"][-1] / (1 << 20) if samples["ws"] else None),
    }
    if gpu_mem:
        rec["gpu_memory"] = gpu_mem
    reg = stream.regimes(data_dir, fill_ms=rec.get("prefill_ms"))
    if reg:
        rec["regimes"] = reg
        rec["fill_ms"] = reg.get("fill_ms")
        for k in ("decode_only_tok_per_s", "end_to_end_tok_per_s"):
            if reg.get(k) is not None:
                rec[k] = reg[k]
        if reg.get("steady"):
            rec["steady_ms_median"] = reg["steady"]["median_ms"]
            rec["steady_tok_per_s"] = reg["steady"]["tok_per_s"]
    pt = stream.prompt_tokens(data_dir)
    if pt:
        rec["prompt_tokens_count"] = len(pt)
    if rec.get("read_bytes") and rec.get("n_decodes"):
        rec["read_bytes_per_decoded_token"] = (rec["read_bytes"] /
                                               rec["n_decodes"])
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, sort_keys=True)
        f.write("\n")

    reg = rec.get("regimes") or {}
    print(f"{args.label}: exit={rec['exit_code']} wall={wall:.1f}s")
    if reg.get("fill_ms") is not None:
        print(f"  fill {reg['fill_ms']/1000.0:.2f}s  "
              f"warm {reg.get('warmup',{}).get('median_ms')}ms  "
              f"steady {reg.get('steady',{}).get('median_ms')}ms = "
              f"{rec.get('steady_tok_per_s')} tok/s")
    print(f"  VRAM peak {rec['memory']['vram_peak_mib']} MiB  "
          f"WS peak {rec['memory']['process_peak_ws_mib']:.0f} MiB  "
          f"avail min {rec['memory']['avail_phys_min_mib']:.0f} MiB")
    print(f"  calls={rec.get('calls')} lookups={rec.get('lookups')} "
          f"reads={rec.get('unbuffered_reads')} bytes={rec.get('read_bytes')}")
    if clock_rec is not None:
        print(f"  {clocks.format_steady(clock_rec)}")
        if clock_rec.get("valid") is False:
            print(f"  CLOCK INVALID: {clock_rec.get('reason')}")
    if gpu_mem:
        print(f"  GPU mem: dedicated peak {gpu_mem['dedicated_peak_mib']:.0f} MiB "
              f"(last {gpu_mem['dedicated_last_mib']:.0f}), shared peak "
              f"{gpu_mem['shared_peak_mib']:.0f} MiB (last {gpu_mem['shared_last_mib']:.0f})")
    print(f"  wrote {json_path}")
    if clock_rec is not None and clock_rec.get("valid") is False and not args.allow_low_clock:
        return 4
    return 0 if rec["exit_code"] == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
