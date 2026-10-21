"""Show the backend (buf) and op of the captures for selected keys in two runs."""
import json
import os
import sys

KEYS = sys.argv[3:] or ["hc_norm_L19", "hc_norm_L20", "hc_mixed_L20",
                        "hc_combine_L19", "q_conv_L20", "conv_output_raw_L20",
                        "alpha_L20", "beta_L20", "state_predelta_L20"]

for run in sys.argv[1:3]:
    print("===", run)
    for line in open(os.path.join(run, "main", "cap", "index.jsonl")):
        d = json.loads(line)
        if d["key"] in KEYS:
            print(f"  {d['key']:22s} occ {d['occ']} buf {d['buf']:8s} op {d['op']:10s} ne {d['ne']}")
