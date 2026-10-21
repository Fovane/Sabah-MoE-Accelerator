"""Check the item6 manifests for scrub damage."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for f in ("results/phase0/item6-prefill64k/manifest.jsonl",
          "results/phase0/item6-prefill128k/manifest.jsonl",
          "results/phase0/item6b-prefill64k-r1/manifest.jsonl"):
    p = os.path.join(ROOT, f)
    if not os.path.exists(p):
        print(f, "MISSING")
        continue
    t = open(p, encoding="utf-8", errors="ignore").read()
    n_drive = t.count("$DRIVE")
    n_ws = t.count("C:\\Projects\\quantization\\Sabah-MoE-Accelerator")
    print(f"{f}: $DRIVE x{n_drive}  C:\\Projects\\quantization\\Sabah-MoE-Accelerator x{n_ws}  len {len(t)}")
    i = t.find("$DRIVE")
    if i >= 0:
        print("   context:", repr(t[max(0, i - 50):i + 20]))
