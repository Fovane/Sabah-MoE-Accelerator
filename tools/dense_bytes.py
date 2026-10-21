import re
import sys
import collections
import os

sys.path.insert(0, r"third_party\llama.cpp\gguf-py")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf

from model_dir import model_path

p = model_path()
r = gguf.GGUFReader(p)

expert_names = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
rows = []
for t in r.tensors:
    if any(t.name.endswith(n) for n in expert_names):
        continue
    key = re.sub(r"^blk\.\d+\.", "blk.*.", t.name)
    rows.append((key, int(t.data.nbytes)))

agg = collections.defaultdict(lambda: [0, 0])
for k, nb in rows:
    agg[k][0] += 1
    agg[k][1] += nb

# tensors that are read in full once per token by a matmul; token_embd is a
# row gather, the *_norm weights are vectors.
gather = {"token_embd.weight"}
vec = {k for k in agg if k.endswith("_norm.weight") or "norm_conv" in k}

print(f"{'count':>5} {'MiB/layer':>10} {'MiB total':>10}  tensor")
for k, (n, nb) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
    print(f"{n:5d} {nb/n/2**20:10.2f} {nb/2**20:10.1f}  {k}")

matmul = {k: v for k, v in agg.items() if k not in gather and k not in vec}
total = sum(v[1] for v in matmul.values())
print(f"\nmatmul weights read per token: {total/2**20:.1f} MiB = {total/2**30:.2f} GiB")
print(f"excluded: gather {sum(agg[k][1] for k in gather)/2**20:.1f} MiB, "
      f"norms {sum(agg[k][1] for k in vec)/2**20:.1f} MiB")
