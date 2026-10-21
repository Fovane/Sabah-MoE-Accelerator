"""Item 5, static half: bytes per dense op class per token, from the GGUF.

No GPU, no model load: reads the tensor table and reports, per op class, the
weight bytes a decode token must read, the class's share of dense traffic, and
the time it would take at the card's ~448 GB/s peak (16283 MiB, 128-bit
GDDR7 @ 14001 MHz effective -> 448.0 GB/s).  Compare those floors with the
measured 12.4-12.8 ms dense time to see which classes have slack.

    .venv\\Scripts\\python.exe tools\\dense_static.py
"""
import collections
import re
import sys

sys.path.insert(0, r"third_party\llama.cpp\gguf-py")
import gguf

GGUF = (r"D:\Strata-data\models\IQ3_S"
        r"\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
PEAK_GBS = 448.0
DENSE_MS = 12.6          # measured, REPORT.md section 2 (12.4-12.8)

r = gguf.GGUFReader(GGUF)

# Classify every tensor by role.  Expert tensors and the gather tables are
# handled separately: experts belong to Sabah, the tables are row-gathers.
def klass(name):
    if name.endswith("ffn_gate_exps.weight") or name.endswith("ffn_up_exps.weight") \
            or name.endswith("ffn_down_exps.weight"):
        return "experts (Sabah)"
    if name.startswith("token_embd.") or name.startswith("blk.*.ple_"):
        return "embedding / PLE (gather)"
    if name.startswith("output."):
        return "output matmul"
    if ".hc_" in name:
        return "hyper-connections"
    if ".attn_qkv." in name or ".ssm_" in name or ".attn_gate." in name:
        return "linear attention (36 layers)"
    if ".attn_q." in name or ".attn_k." in name or ".attn_v." in name \
            or ".attn_output." in name or ".indexer." in name:
        return "full attention (12 layers)"
    if name.startswith("output."):
        return "output matmul"
    if ".ffn_gate_inp." in name:
        return "router"
    if "_shexp." in name:
        return "shared expert"
    if name.endswith("_norm.weight") or name.endswith("_norm_conv.weight") \
            or name.endswith(".ssm_a") or name.endswith(".ssm_dt.bias"):
        return "norms"
    return "other"

rows = collections.defaultdict(lambda: [0, 0, collections.Counter()])
shapes = {}
for t in r.tensors:
    key = re.sub(r"^blk\.\d+\.", "blk.*.", t.name)
    k = klass(key)
    nb = int(t.data.nbytes)
    rows[k][0] += 1
    rows[k][1] += nb
    rows[k][2][key] += nb
    shapes[t.name] = (tuple(int(x) for x in t.shape), int(t.data.nbytes))

print(f"{'class':32s} {'tensors':>7s} {'MiB/token':>10s} {'GB/token':>9s} "
      f"{'share':>6s} {'floor ms':>8s}")
total = 0
for k, (n, nb, _) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
    total += nb if "gather" not in k and "norms" not in k and "Sabah" not in k else 0
for k, (n, nb, _) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
    counted = k not in ("embedding / PLE (gather)", "norms", "experts (Sabah)", "other")
    base = nb if counted else 0
    share = 100.0 * base / total if total else 0.0
    floor = base / (PEAK_GBS * 1e9) * 1e3
    print(f"{k:32s} {n:7d} {nb/2**20:10.1f} {nb/1e9:9.3f} {share:5.1f}% {floor:8.3f}")

print(f"\ndense matmul bytes per token : {total/1e9:.3f} GB  "
      f"({total/2**20:.0f} MiB)")
floor = total / (PEAK_GBS * 1e9) * 1e3
print(f"floor at {PEAK_GBS:.0f} GB/s       : {floor:.2f} ms")
print(f"measured dense time          : {DENSE_MS:.2f} ms")
print(f"achieved                     : {total/1e9/(DENSE_MS*1e-3):.0f} GB/s "
      f"({100.0*floor/DENSE_MS:.0f}% of peak)")
print(f"slack if peak were reached   : {DENSE_MS-floor:.2f} ms/token")

print("\ntop 15 tensor kinds by bytes/token (dense only):")
flat = []
for k, (n, nb, per) in rows.items():
    if k in ("experts (Sabah)", "embedding / PLE (gather)", "norms"):
        continue
    for name, b in per.items():
        flat.append((b, name, n))
for b, name, n in sorted(flat, reverse=True)[:15]:
    sh = shapes.get(name.replace("blk.*.", "blk.0."))
    print(f"  {b/2**20:7.1f} MiB  {name:30s} shape={sh[0] if sh else '?'}")

print("\n'unclassified' tensors:")
for name, b in sorted(rows["other"][2].items(), key=lambda kv: -kv[1])[:20]:
    sh = shapes.get(name.replace("blk.*.", "blk.0."))
    print(f"  {b/2**20:7.1f} MiB  {name:30s} shape={sh[0] if sh else '?'}")

print("\ngather tables (bytes, and the row a token actually reads):")
for name in ("token_embd.weight", "blk.0.ple_key.weight", "blk.0.ple_value.weight"):
    if name in shapes:
        sh, nb = shapes[name]
        row = nb // sh[0] if sh else 0
        print(f"  {name:28s} shape={sh} total={nb/2**20:.1f} MiB "
              f"row={row} B")
