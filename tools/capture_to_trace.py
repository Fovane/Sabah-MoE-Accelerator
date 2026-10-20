"""Convert sabah-diag capture blobs into the trace_locality i32 layout.

The diag writes ``<item>/cap/ffn_moe_topk_L<block>.bin``: i32, token-major rows
of ``top_k`` expert ids, appended in execution order (prefill positions first,
then one row per decode step). trace_locality wants one
``ffn_moe_topk_L<block>.i32`` per block with every item concatenated in manifest
order, so this is a concatenation, not a conversion.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", required=True, help="SABAH_DIAG_OUT directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--tail-decode", action="store_true",
                    help="keep only the last n_predict rows of each item (decode)")
    args = ap.parse_args()

    order = []
    n_predict = {}
    with open(args.manifest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                e = json.loads(line)
                order.append(e["id"])
                n_predict[e["id"]] = int(e.get("n_predict", 64))

    os.makedirs(args.out, exist_ok=True)
    pat = re.compile(r"^ffn_moe_topk_L(\d+)\.bin$")
    totals: dict[int, int] = {}
    for item in order:
        cap = os.path.join(args.capture, item, "cap")
        if not os.path.isdir(cap):
            raise SystemExit("missing capture dir: %s" % cap)
        for name in sorted(os.listdir(cap)):
            m = pat.match(name)
            if not m:
                continue
            block = int(m.group(1))
            with open(os.path.join(cap, name), "rb") as src:
                data = src.read()
            if len(data) % (4 * args.top_k):
                raise SystemExit("%s: size %d is not a whole number of rows"
                                 % (name, len(data)))
            rows = len(data) // (4 * args.top_k)
            if args.tail_decode:
                keep = min(n_predict[item], rows)
                data = data[(rows - keep) * 4 * args.top_k:]
            dst = os.path.join(args.out, "ffn_moe_topk_L%d.i32" % block)
            with open(dst, "ab") as out:
                out.write(data)
            totals[block] = totals.get(block, 0) + len(data) // (4 * args.top_k)

    if not totals:
        raise SystemExit("no ffn_moe_topk blobs found under %s" % args.capture)
    if len(set(totals.values())) != 1:
        print("warning: blocks differ in steps: %s"
              % sorted(set(totals.values())))
    print("%d blocks, %d steps each, %d items -> %s"
          % (len(totals), next(iter(totals.values())), len(order), args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
