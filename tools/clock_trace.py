"""1 s-resolution idle -> decode -> idle clock trace.

Catches the transition into the low-clock state (if it happens) and records the
machine settings that allow it: Windows power plan, HAGS, NVIDIA power
management registry values, display state.

    python tools/clock_trace.py --idle-before 20 --idle-after 40 --tokens 64

The trace is a CSV of every sample plus a JSON summary that names the first
sample below 2400 MHz and what preceded it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sabah.tools import clocks, stream  # noqa: E402

DEFAULT_MODEL = (r"D:\Strata-data\models\IQ3_S"
                 r"\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
DEFAULT_DIAG = os.path.join(ROOT, "third_party", "llama.cpp", "build", "bin",
                            "llama-sabah-diag.exe")
LOW_MHZ = clocks.MIN_STEADY_SM_MHZ


def run_decode(args, out_dir, sampler):
    env = dict(os.environ)
    env.update(stream.BASE_ENV)
    env["SABAH_RT_LIB"] = os.path.abspath(args.rt)
    env["SABAH_LLAMA_HOT_BYTES"] = str(args.hot_bytes)
    env["SABAH_DIAG_OUT"] = out_dir
    env["SABAH_LLAMA_PREFETCH"] = "0"
    env.pop("SABAH_LLAMA_STATUS_FILE", None)
    argv = [os.path.abspath(args.diag), "-m", os.path.abspath(args.model)]
    argv += stream.BASE_FLAGS
    argv += ["-t", "8", "-c", str(args.context), "-n", str(args.tokens),
             "-p", args.prompt]
    stream.clean_run_dir(out_dir)
    log = os.path.join(out_dir, "driver.log")
    t0 = time.monotonic()
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(argv, env=env, stdout=f, stderr=subprocess.STDOUT)
    t1 = time.monotonic()
    text = open(log, "r", encoding="utf-8", errors="replace").read()
    rec = stream.parse_driver(text)
    return {"exit_code": proc.returncode, "t0": t0, "t1": t1,
            "decode_ms_sum": rec.get("decode_ms_sum"),
            "decode_ms_median": rec.get("decode_ms_median"),
            "n_decodes": rec.get("n_decodes"),
            "read_bytes": rec.get("read_bytes")}


def analyse(samples, decode):
    out = {"low_threshold_mhz": LOW_MHZ, "first_low": None,
           "low_samples": 0, "min_sm": None, "min_at_s": None,
           "transitions": []}
    prev = None
    for t, s in samples:
        sm = s.get("sm_mhz")
        if sm is not None and (out["min_sm"] is None or sm < out["min_sm"]):
            out["min_sm"] = sm
            out["min_at_s"] = round(t, 2)
        state = "low" if (sm is not None and sm < LOW_MHZ) else "boost"
        if prev is not None and state != prev:
            out["transitions"].append({"t_s": round(t, 2), "to": state,
                                       "sm_mhz": sm, "pstate": s.get("pstate")})
        prev = state
        if state == "low":
            out["low_samples"] += 1
            if out["first_low"] is None:
                out["first_low"] = {"t_s": round(t, 2), "sm_mhz": sm,
                                    "pstate": s.get("pstate"),
                                    "power_w": s.get("power_w")}
    if decode:
        out["decode"] = {"start_s": round(decode["t0"], 2),
                         "end_s": round(decode["t1"], 2),
                         "duration_s": round(decode["t1"] - decode["t0"], 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--diag", default=DEFAULT_DIAG)
    ap.add_argument("--rt", default=os.path.join(ROOT, "sabah", "runtime",
                                                 "cuda", "sabah_rt.dll"))
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--hot-bytes", default="6.0e9")
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--idle-before", type=float, default=20.0)
    ap.add_argument("--idle-after", type=float, default=40.0)
    ap.add_argument("--label", default="idle-decode-idle")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results",
                                                       "phase0", "clock-trace"))
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.label)
    os.makedirs(out_dir, exist_ok=True)

    sampler = clocks.ClockSampler(interval=1.0)
    sampler.start()
    started = time.monotonic()

    print(f"machine: {json.dumps(clocks.machine_state())}")
    print(f"idle {args.idle_before:.0f}s ...")
    time.sleep(args.idle_before)
    print(f"decode {args.tokens} tokens ...")
    dec = run_decode(args, out_dir, sampler)
    print(f"  exit={dec['exit_code']} decodes={dec['n_decodes']} "
          f"median={dec['decode_ms_median']} ms  "
          f"({dec['t1'] - dec['t0']:.1f}s wall)")
    print(f"idle {args.idle_after:.0f}s ...")
    time.sleep(args.idle_after)
    sampler.stop()

    samples = [(round(t - started, 2), s) for t, s in sampler.samples]
    dec["t0"] -= started
    dec["t1"] -= started
    summary = analyse(samples, dec)
    summary["machine"] = clocks.machine_state()
    summary["decode"] = dec
    summary["label"] = args.label
    summary["idle_before_s"] = args.idle_before
    summary["idle_after_s"] = args.idle_after

    csv_path = os.path.join(out_dir, "clocks.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("t_s,sm_mhz,mem_mhz,pstate,power_w,temp_c,util_pct,throttle\n")
        for t, s in samples:
            f.write(f"{t},{s['sm_mhz']},{s['mem_mhz']},{s['pstate']},"
                    f"{s['power_w']},{s['temp_c']},{s['util_pct']},"
                    f"{s['throttle']}\n")
    json_path = os.path.join(out_dir, "trace.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    same = [s for _, s in samples]
    print(f"\n{'t':>6} {'SM':>6} {'mem':>6} {'P':>3} {'W':>6} {'util':>5}")
    for t, s in samples:
        mark = ""
        if dec["t0"] <= t <= dec["t1"]:
            mark = " decode"
        print(f"{t:6.1f} {s['sm_mhz']:6.0f} {s['mem_mhz']:6.0f} "
              f"{s['pstate']:>3} {s['power_w']:6.1f} {s['util_pct']:5.0f}{mark}")
    print(f"\nmin SM {summary['min_sm']:.0f} MHz at {summary['min_at_s']}s, "
          f"low samples {summary['low_samples']}/{len(same)}")
    print(f"first low: {summary['first_low']}")
    print(f"transitions: {summary['transitions']}")
    print(f"wrote {csv_path}\n      {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
