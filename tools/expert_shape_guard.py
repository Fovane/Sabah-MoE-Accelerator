"""Check every expert tensor in the model against the TILE3 guard.

TILE3 stages the 32-element dots of one row in `sh[16][96]`, i.e. it can serve a
reduction of at most 96 sub-blocks (k <= 3072). Anything wider must fall back to
k_mul_mat_id. This prints the actual k of every expert tensor in every shard.
"""
import collections
import glob
import sys

sys.path.insert(0, r"third_party\llama.cpp\gguf-py")
import gguf

PAT = r"$DRIVE\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000*-of-00002.gguf"
ROLES = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
GUARD_NSUB = 96

per_role = collections.defaultdict(lambda: collections.Counter())
layers = collections.Counter()
mismatch = []

for path in sorted(glob.glob(PAT)):
    r = gguf.GGUFReader(path)
    for t in r.tensors:
        role = next((x for x in ROLES if t.name.endswith(x)), None)
        if role is None:
            continue
        shape = [int(d) for d in t.shape]
        k = shape[0]
        per_role[role.rsplit(".", 2)[0]][k] += 1
        layers[t.name.split(".")[1]] += 1

ok = True
print(f"guard: nsub = k/32 must be <= {GUARD_NSUB} (k <= {GUARD_NSUB * 32})\n")
print(f"{'tensor':<16} {'k (=ne0)':>10} {'nsub':>6} {'count':>6}  guard")
for role in sorted(per_role):
    for k, n in sorted(per_role[role].items()):
        nsub = k / 32.0
        good = nsub <= GUARD_NSUB and k % 32 == 0
        ok = ok and good
        print(f"{role:<16} {k:>10} {nsub:>6.1f} {n:>6}  {'ok' if good else 'FALLBACK'}")
        if not good:
            mismatch.append((role, k, n))

print(f"\nlayers with an expert tensor: {len(layers)}")
print(f"experts per layer (from the tensor rank): ", end="")
r = gguf.GGUFReader(sorted(glob.glob(PAT))[0])
for t in r.tensors:
    if t.name.endswith("ffn_gate_exps.weight"):
        print(int(t.shape[2]), "(layer", t.name.split(".")[1] + ")")
        break

print("\nguard verdict:", "ALL EXPERT TENSORS INSIDE TILE3" if ok else f"FALLBACK NEEDED: {mismatch}")
sys.exit(0 if ok else 1)
