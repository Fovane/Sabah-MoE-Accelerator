"""Step 4: node bisect inside the first diverging layer, with SEG=0.

    python tools/b_step4.py [--layers 3,4,5,6] [--names a,b,c]

Captures the layer-4 graph nodes in build order for the failing config
(1888-token prompt, -ub 512 -> 4 ubatches, G=4, SEG=0) on both sides and reports,
per (key, occ), the first capture whose bytes differ.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PROMPT = "results/phase0/b-eq/prompt1630.txt"


def run(label, group, g, names, layers):
    cmd = [PY, "tools/phase0_measure.py", "--label", label, "--tokens", "1",
           "--context", "4096", "--prompt-file", PROMPT,
           "--flag=-b", "--flag=2048", "--flag=-ub", "--flag=512",
           "--set", "SABAH_CUDA_GRAPH_SEG=0", "--allow-low-clock",
           "--set", f"SABAH_DIAG_NAMES={names}",
           "--set", f"SABAH_DIAG_LAYERS={layers}"]
    if group:
        cmd += ["--set", "SABAH_PREFILL_GROUP=1", "--set", f"SABAH_PREFILL_GROUP_G={g}"]
    else:
        cmd += ["--set", "SABAH_PREFILL_GROUP=0"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    p = os.path.join(ROOT, "results/phase0", label, "main")
    return os.path.exists(os.path.join(p, "cap", "index.jsonl")), (r.stdout or "")[-300:]


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
    ap.add_argument("--names", default="hc_norm,hc_gate,hc_mixed,hc_combine,beta,alpha,gate,conv_state_at,state_predelta,conv_output_raw,q_conv,k_conv,v_conv,attn_output,final_output,l_last")
    ap.add_argument("--layers", default="3,4,5,6,7")
    args = ap.parse_args()

    ok, tail = run("b4-ref", False, None, args.names, args.layers)
    if not ok:
        print(tail)
        return 1
    ok, tail = run("b4-grp", True, 4, args.names, args.layers)
    if not ok:
        print(tail)
        return 1

    ra, rb = recs("b4-ref"), recs("b4-grp")
    print(f"\ncaptures per key: ref {len(ra)}  grp {len(rb)}")

    # report the number of captures per key and their sizes first
    seen = {}
    for r in ra + rb:
        seen.setdefault(r["key"], []).append((r["step"], r["occ"], r["ne"], r["nbytes"]))
    print("\nkey            | count | ne/size sequence")
    for k, v in sorted(seen.items()):
        print(f"{k:28s} | {len(v):5d} | {[(n, s) for _, _, n, s in v]}")

    # pair by (key, occ) and find the first difference, in capture order
    print("\nfirst differing capture per key (paired by occ):")
    ia = {}
    for r in ra:
        ia.setdefault(r["key"], {})[r["occ"]] = r
    ib = {}
    for r in rb:
        ib.setdefault(r["key"], {})[r["occ"]] = r
    for k in sorted(set(ia) & set(ib)):
        for occ in sorted(set(ia[k]) & set(ib[k])):
            x, y = ia[k][occ], ib[k][occ]
            if x["nbytes"] != y["nbytes"]:
                print(f"  {k} occ {occ}: size {x['nbytes']} vs {y['nbytes']}")
                continue
            if read("b4-ref", x) != read("b4-grp", y):
                nf = x["nbytes"] // 4
                first = None
                a = read("b4-ref", x)
                b = read("b4-grp", y)
                for i in range(nf):
                    if a[4 * i:4 * i + 4] != b[4 * i:4 * i + 4]:
                        first = i
                        break
                row = first // x["ne"][0] if first is not None and x["ne"][0] else None
                print(f"  {k} occ {occ}: DIFF  ne {x['ne']}  first diff float {first} (row {row})")
                break
        else:
            print(f"  {k}: all captures identical")

    # also the final artifacts
    print("\nfinal artifacts:")
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        a = open(os.path.join(ROOT, "results/phase0/b4-ref/main", f), "rb").read()
        b = open(os.path.join(ROOT, "results/phase0/b4-grp/main", f), "rb").read()
        print(f"  {f}: {'IDENTICAL' if a == b else 'DIFF'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
