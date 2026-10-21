"""Convert the diag's expert-id captures into a compact replay trace.

The diag captures, per layer and role, the MUL_MAT_ID ids tensor (top-k per
token). This reads those captures and writes one trace file of fixed-width
records so the offline slab simulator can replay it:

    uint32 step, uint32 role, uint32 layer, uint32 expert

    python tools/route_trace.py results/phase0/<label>/main/cap out.bin [--topk 10]

Roles are the three expert projections in the order they execute:
0 gate, 1 up, 2 down.

The captures are decode steps (one token per step), so each step contributes
`topk` ids of 4 bytes = `topk`*4 bytes per (role, layer) file.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

ROLES = (("ffn_moe_gate_ids", 0), ("ffn_moe_up_ids", 1), ("ffn_moe_down_ids", 2))
REC = struct.Struct("<IIII")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cap")
    ap.add_argument("out")
    ap.add_argument("--topk", type=int, default=10)
    args = ap.parse_args()
    cap = args.cap
    row = args.topk * 4

    layers = set()
    for name in os.listdir(cap):
        for base, _r in ROLES:
            if name.startswith(base + "_L") and name.endswith(".bin"):
                layers.add(int(name[len(base) + 2:-4]))
    if not layers:
        print("no expert id captures in", cap)
        print("run with SABAH_DIAG_MMID=1 SABAH_DIAG_NAMES=ffn_moe_gate,ffn_moe_up,ffn_moe_down")
        return 1
    layers = sorted(layers)

    counts = {}
    for base, r in ROLES:
        sizes = []
        for il in layers:
            p = os.path.join(cap, "%s_L%d.bin" % (base, il))
            if not os.path.exists(p):
                print("missing", p)
                return 1
            sizes.append(os.path.getsize(p))
        if len(set(sizes)) != 1:
            print("inconsistent capture sizes for %s: %s" % (base, sorted(set(sizes))))
            return 1
        if sizes[0] % row:
            print("capture size %d for %s is not a multiple of %d bytes/step"
                  % (sizes[0], base, row))
            return 1
        counts[r] = sizes[0] // row

    steps = min(counts.values())
    print("layers: %d (%d..%d), topk %d" % (len(layers), layers[0], layers[-1], args.topk))
    print("steps per role: %s (using %d)" % (counts, steps))

    n_written = 0
    by_step = {}
    with open(args.out, "wb") as f:
        for s in range(steps):
            per_step = 0
            for base, r in ROLES:
                for il in layers:
                    p = os.path.join(cap, "%s_L%d.bin" % (base, il))
                    with open(p, "rb") as g:
                        g.seek(s * row)
                        ids = struct.unpack("<%di" % args.topk, g.read(row))
                    for e in ids:
                        if e < 0:
                            continue
                        f.write(REC.pack(s, r, il, e))
                        n_written += 1
                        per_step += 1
            by_step[s] = per_step
    ks = sorted(by_step.values())
    print("wrote %d records (%d bytes); per step: min %d, median %d, max %d"
          % (n_written, os.path.getsize(args.out), ks[0], ks[len(ks) // 2], ks[-1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
