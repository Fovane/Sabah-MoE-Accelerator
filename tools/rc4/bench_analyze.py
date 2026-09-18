"""Sustained-generation benchmark from uninstrumented sabah-diag runs."""
import json
import os
import sys

import numpy as np

RUNS = "D:/sabah_rc4/runs/"
rows = {}
for name in sorted(os.listdir(RUNS)):
    if not name.startswith("bench_"):
        continue
    p = RUNS + name
    if not os.path.exists(p + "/step_ms.f64"):
        continue
    ms = np.fromfile(p + "/step_ms.f64", np.float64)
    arg = np.fromfile(p + "/argmax.i32", np.int32)
    mode = json.load(open(p + "/mode.json"))
    dec = ms[1:]
    rows[name] = dict(mode=mode["mode"], tokens=int(len(ms)), prefill_ms=float(ms[0]),
                      decode_tok_s=float(len(dec) / (dec.sum() / 1000.0)),
                      decode_ms_median=float(np.median(dec)), decode_ms_p95=float(np.percentile(dec, 95)),
                      decode_tok_s_last32=float(32 / (dec[-32:].sum() / 1000.0)),
                      wall_s=mode["wall_s"], sabah_metrics=mode.get("sabah_metrics"),
                      argmax_sha=__import__("hashlib").sha256(arg.tobytes()).hexdigest()[:16])


def best(mode):
    v = [r for r in rows.values() if r["mode"] == mode]
    return max(v, key=lambda r: r["decode_tok_s"]) if v else None


sab, stock, cuda, cpu = best("sabah"), best("stock"), best("cuda"), best("cpu")
out = dict(runs=rows)
if sab and stock:
    out["MEASURED_SPEEDUP_vs_stock_llama_cpp"] = sab["decode_tok_s"] / stock["decode_tok_s"]
if sab and cuda:
    out["MEASURED_SPEEDUP_vs_native_cuda_same_placement"] = sab["decode_tok_s"] / cuda["decode_tok_s"]
for k, r in rows.items():
    print("%-16s %-6s decode %.3f tok/s  (median %.0f ms, p95 %.0f ms, last32 %.3f tok/s)  prefill %.1f s"
          % (k, r["mode"], r["decode_tok_s"], r["decode_ms_median"], r["decode_ms_p95"],
             r["decode_tok_s_last32"], r["prefill_ms"] / 1000))
for k in ("MEASURED_SPEEDUP_vs_stock_llama_cpp", "MEASURED_SPEEDUP_vs_native_cuda_same_placement"):
    if k in out:
        print("%s = %.3f" % (k, out[k]))
json.dump(out, open(sys.argv[1], "w"), indent=1)
