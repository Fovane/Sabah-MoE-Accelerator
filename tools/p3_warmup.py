"""Phase 3: warm-up and slab, measured on the chat and coding traces.

For each (trace, slab, SEG) it runs one full decode and reports, per regime:

  * fill: ms, GB read
  * warm-up: the first 128 decodes - ms/token, MB/token, misses/token
  * steady: the last 256 decodes (tail window) - ms/token, MB/token, misses
  * hit rate over the whole run, and over the tail window alone

That is the evidence for two claims in the README: the decode headline belongs
to one regime, and the coding trace's warm-up is a different thing from the
chat trace's.

    python tools/p3_warmup.py [--traces chat,code] [--slabs 6.0e9,7.0e9]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

TRACES = {
    # the two traces item 6 measured, with the prompts they used
    "chat": "results/phase0/item6-chat/main/prompt.txt",
    "code": "results/phase0/item6-code/main/prompt.txt",
}


def run(label, prompt, tokens, context, slab, seg, batch=2048, ub=512):
    cmd = [PY, "tools/phase0_measure.py", "--label", label,
           "--tokens", str(tokens), "--context", str(context),
           "--prompt-file", prompt,
           "--flag=-b", "--flag=" + str(batch),
           "--flag=-ub", "--flag=" + str(ub),
           "--allow-low-clock",
           "--set", f"SABAH_LLAMA_HOT_BYTES={slab}"]
    if seg is not None:
        cmd += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}"]
    subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    d = os.path.join(ROOT, "results/phase0", label, "main")
    mp = os.path.join(ROOT, "results/phase0", label, "measurement.json")
    if not os.path.exists(os.path.join(d, "prefill_ms.f64")) or not os.path.exists(mp):
        return None

    rec = json.load(open(mp, encoding="utf-8"))

    def f64(name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return []
        return list(struct.unpack("<%dd" % (os.path.getsize(p) // 8),
                                  open(p, "rb").read()))

    def i64(name):
        p = os.path.join(d, name)
        if not os.path.exists(p):
            return []
        return list(struct.unpack("<%dq" % (os.path.getsize(p) // 8),
                                  open(p, "rb").read()))

    ms = f64("step_ms.f64")
    bts = i64("step_bytes.i64")
    mis = i64("step_misses.i64")
    blob = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        blob += open(os.path.join(d, f), "rb").read()
    return {
        "label": label, "trace": None, "slab": slab, "seg": seg,
        "ms": ms, "bytes": bts, "misses": mis,
        "fill_s": rec["prefill"]["ms"] / 1000.0,
        "fill_gb": rec["prefill"]["bytes"] / 1e9,
        "prompt_tok": rec.get("prompt_tokens_count"),
        "hash": hashlib.sha256(blob).hexdigest()[:16],
        "artifacts": blob,
        "clock_valid": (rec.get("clocks") or {}).get("clock_valid"),
        "sm": ((rec.get("clocks") or {}).get("steady", {}) or {}).get("sm_mhz", {}).get("median"),
    }


def regime(d, lo, hi):
    """mean ms/token, MB/token, misses/token over steps [lo, hi)."""
    sl = slice(lo, hi)
    n = max(1, len(d["ms"][sl]))
    return (sum(d["ms"][sl]) / n,
            sum(d["bytes"][sl]) / n / 1e6,
            sum(d["misses"][sl]) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="chat")
    ap.add_argument("--slabs", default="6.0e9")
    ap.add_argument("--segs", default="48")
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ub", type=int, default=512)
    args = ap.parse_args()

    rows = []
    for trace in args.traces.split(","):
        for slab in args.slabs.split(","):
            for seg in args.segs.split(","):
                seg = None if seg in ("-", "none") else seg
                label = f"p3-{trace}-{slab.replace('.', '_').replace('e9', 'e9')}-seg{seg}"
                print(f"[run] {label}", flush=True)
                d = run(label, TRACES[trace], args.tokens, args.context, slab, seg, args.batch, args.ub)
                if d is None:
                    print("   FAILED")
                    continue
                d["trace"] = trace
                rows.append(d)

    print("\ntrace | slab | SEG | fill s | fill GB | prompt tok | clocks (steady SM/mem, pstate) | hash")
    print("------+------+-----+--------+---------+------------+----------------------------------+-----")
    for d in rows:
        # validity comes from the steady medians against the floors, not from
        # the harness's clock_valid field: it is unset in these records, and
        # treating "unset" as invalid reported valid runs as invalid
        ok = bool(d["sm"] and d["sm"] >= 2400)
        print(f"{d['trace']:5s} | {d['slab']:>4} | {str(d['seg']):>3} | {d['fill_s']:>6.1f} | "
              f"{d['fill_gb']:>7.1f} | {d['prompt_tok']:>10} | "
              f"{'VALID' if ok else 'INVALID'} {d['sm']}/{d['sm']} {d['pstate']} | {d['hash']}")

    print("\nregime table (mean per decode token):")
    print("trace | slab | SEG | warm-up ms | warm-up MB | warm-up misses | tail ms | tail MB | tail misses")
    print("------+------+-----+------------+------------+----------------+---------+---------+-------------")
    for d in rows:
        n = len(d["ms"])
        w_ms, w_mb, w_mis = regime(d, 0, min(128, n))
        t_ms, t_mb, t_mis = regime(d, max(0, n - 256), n)
        print(f"{d['trace']:5s} | {d['slab']:>4} | {str(d['seg']):>3} | {w_ms:>10.1f} | {w_mb:>10.1f} | "
              f"{w_mis:>14.1f} | {t_ms:>7.1f} | {t_mb:>7.2f} | {t_mis:>12.2f}")

    # the byte-identical gate between traces/slabs that used the same prompt
    by_trace = {}
    for d in rows:
        by_trace.setdefault(d["trace"], []).append(d)
    print("\noutput byte-identity within a trace (same prompt, different slab/SEG):")
    for trace, ds in by_trace.items():
        if len(ds) < 2:
            continue
        a = ds[0]
        for b in ds[1:]:
            same = a["artifacts"] == b["artifacts"]
            print(f"  {trace}: {a['label']} vs {b['label']} -> "
                  f"{'IDENTICAL' if same else 'DIFF'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
