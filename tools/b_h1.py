"""Phase 4, H1: bisect the first diverging node at the group-1 boundary for G=24.

One ubatch (-ub 2048, a single 1888-token prompt), G=24, SEG=0, so the only
difference from the reference is that group 1 (layers 24..47) replays the ubatch
through seek() and reads the staged hc.

    python tools/b_h1.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
P = "results/phase0/b-eq/prompt1630.txt"

NAMES = "hc_norm,hc_gate,hc_mixed,hc_combine,state_predelta,conv_state_at,q_conv,k_conv,v_conv,beta,alpha,gate,conv_output_raw,conv_output_silu,attn_output,new_state,final_output,l_last"


def run(label, g, names, layers):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", P,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=2048",
           "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock",
           "--set", f"SABAH_DIAG_NAMES={names}", "--set", f"SABAH_DIAG_LAYERS={layers}"]
    if g is None:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    idx = os.path.join(ROOT, "results/phase0", label, "main", "cap", "index.jsonl")
    return os.path.exists(idx), (r.stdout or "")[-300:]


def recs(label):
    out = []
    for line in open(os.path.join(ROOT, "results/phase0", label, "main", "cap", "index.jsonl")):
        out.append(json.loads(line))
    return out


def read(label, rec):
    with open(os.path.join(ROOT, "results/phase0", label, "main", "cap", rec["key"] + ".bin"), "rb") as f:
        f.seek(rec["offset"])
        return f.read(rec["nbytes"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="23,24,25,44,45,46,47")
    ap.add_argument("--g", default="24")
    args = ap.parse_args()

    ok, tail = run("h1-ref", None, NAMES, args.layers)
    if not ok:
        print(tail)
        return 1
    ok, tail = run("h1-grp", int(args.g), NAMES, args.layers)
    if not ok:
        print(tail)
        return 1

    ra, rb = recs("h1-ref"), recs("h1-grp")
    print(f"captures: ref {len(ra)}  grp {len(rb)}")
    ia, ib = {}, {}
    for r in ra:
        ia.setdefault(r["key"], []).append(r)
    for r in rb:
        ib.setdefault(r["key"], []).append(r)

    order = []
    for r in ra:
        if r["key"] not in order:
            order.append(r["key"])
    print("\nkey (reference capture order) | count ref/grp | first differing capture")
    print("------------------------------+---------------+-------------------------")
    for k in order:
        va, vb = ia.get(k, []), ib.get(k, [])
        n = min(len(va), len(vb))
        first = None
        for j in range(n):
            if va[j]["nbytes"] != vb[j]["nbytes"]:
                first = f"size {va[j]['nbytes']} vs {vb[j]['nbytes']}"
                break
            if read("h1-ref", va[j]) != read("h1-grp", vb[j]):
                a = read("h1-ref", va[j])
                b = read("h1-grp", vb[j])
                nf = len(a) // 4
                fi = next((i for i in range(nf) if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]), None)
                first = f"occ {va[j]['occ']} ne {va[j]['ne']} first float {fi}"
                break
        print(f"{k:29s} | {len(va):5d}/{len(vb):5d} | {first if first else 'identical'}")

    print("\nfinal artifacts:")
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        a = open(os.path.join(ROOT, "results/phase0/h1-ref/main", f), "rb").read()
        b = open(os.path.join(ROOT, "results/phase0/h1-grp/main", f), "rb").read()
        same = a == b
        extra = ""
        if not same and f == "logits.f32":
            import struct
            fa = struct.unpack("<%df" % (len(a) // 4), a)
            fb = struct.unpack("<%df" % (len(b) // 4), b)
            d = [i for i in range(len(fa)) if fa[i] != fb[i]]
            extra = f" (ndiff {len(d)}, max_abs {max(abs(fa[i]-fb[i]) for i in d):.3e})"
        print(f"  {f}: {'IDENTICAL' if same else 'DIFF'}{extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
