"""Phase 1.3/1.4: the missing item-4 acceptance cells.

  c128k-B : a second prompt at -c 131072, >=1024 tokens, SEG=0 vs SEG=48,
            byte-identical fed/argmax/logits.
  kv640   : 640 tokens at -c 2048 (the -c 512 variant is not possible: a prompt
            plus 640 decodes exceeds n_ctx and the hybrid memory does not shift).
  tail N=2: 128K chat, 2048 decode steps, slab 6.0e9, SEG=48, two runs of the
            6 ms / 12.38 ms saving, tail window = last 256 steps.

    python tools/p1_accept.py [--only c128k-B,kv640,tail]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MODEL = r"C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
P1 = "results/phase0/b-eq/prompt1630.txt"          # README text, 1888 tokens
P2 = "results/phase0/item6-prefill64k/main/prompt.txt"  # code text, second prompt
COMMON = ["--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
          "--allow-low-clock", "--set", "SABAH_LLAMA_HOT_BYTES=6.0e9"]


def run(label, prompt, tokens, context, seg, extra):
    cmd = [PY, "tools/phase0_measure.py", "--label", label,
           "--tokens", str(tokens), "--context", str(context),
           "--prompt-file", prompt] + COMMON
    if seg is not None:
        cmd += ["--set", f"SABAH_CUDA_GRAPH_SEG={seg}"]
    if extra:
        cmd += extra
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "results/phase0", label, "main")
    if not os.path.exists(os.path.join(p, "argmax.i32")):
        return None, (r.stdout or "")[-400:]
    d = {}
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        d[f] = hashlib.sha256(open(os.path.join(p, f), "rb").read()).hexdigest()[:16]
    d["dir"] = os.path.join(ROOT, "results/phase0", label)
    d["n_prompt"] = json.load(open(os.path.join(p, "run.json")))["n_prompt"]
    return d, ""


def brief(d):
    if d is None:
        return "FAILED"
    return "logits %s argmax/fed %s (prompt %d tok)" % (d["logits.f32"], d["argmax.i32"], d["n_prompt"])


def tail_ms(label):
    steps = None
    p = os.path.join(ROOT, "results/phase0", label, "main", "step_ms.f64")
    if not os.path.exists(p):
        return None
    import struct
    ms = struct.unpack("<%dd" % (os.path.getsize(p) // 8), open(p, "rb").read())
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="c128k-B,kv640,tail")
    args = ap.parse_args()
    want = set(args.only.split(","))

    if "c128k-B" in want:
        print("[c128k-B] second prompt at -c 131072, 1024 tokens, SEG 0 vs 48")
        a, ta = run("p1-c128k-B-ref", P2, 1024, 131072, 0, [])
        b, tb = run("p1-c128k-B-seg48", P2, 1024, 131072, 48, [])
        print("  ref  :", brief(a) or ta)
        print("  SEG48:", brief(b) or tb)
        if a and b:
            print("  verdict:", "BYTE-IDENTICAL" if a == b else "DIFF")

    if "kv640" in want:
        print("[kv640] 640 tokens at -c 2048 (the -c 512 case does not fit)")
        a, ta = run("p1-kv640-ref", P1, 640, 2048, 0, [])
        b, tb = run("p1-kv640-seg48", P1, 640, 2048, 48, [])
        print("  ref  :", brief(a) or ta)
        print("  SEG48:", brief(b) or tb)
        if a and b:
            print("  verdict:", "BYTE-IDENTICAL" if a == b else "DIFF")

    if "tail" in want:
        print("[tail N=2] 128K chat, 2048 steps, slab 6.0e9, tail = last 256")
        runs = [("p1-tail1-seg0", 0), ("p1-tail1-seg48", 48),
                ("p1-tail2-seg0", 0), ("p1-tail2-seg48", 48)]
        got = {}
        for label, seg in runs:
            d, t = run(label, P1, 2048, 131072, seg, [])
            print(f"  {label}: {'ok' if d else 'FAILED'}")
            if d:
                got[label] = d
        for k in (1, 2):
            a = got.get(f"p1-tail{k}-seg0")
            b = got.get(f"p1-tail{k}-seg48")
            if not (a and b):
                continue
            ma = tail_ms(f"p1-tail{k}-seg0")
            mb = tail_ms(f"p1-tail{k}-seg48")
            print(f"  run {k}: logits identical={a['logits.f32'] == b['logits.f32']}"
                  f"  tail seg0={min(ma[-256:]) if ma else '?'}"
                  f"  tail seg48={min(mb[-256:]) if mb else '?'}")


if __name__ == "__main__":
    raise SystemExit(main())
