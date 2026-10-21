import json, struct
a = open(r"results\phase0\baseline-n256\argmax.i32", "rb").read()
b = open(r"results\phase0\phase2-tile1\argmax.i32", "rb").read()
print("len", len(a), len(b), "equal", a == b)
if a != b:
    A = struct.unpack("<%di" % (len(a) // 4), a)
    B = struct.unpack("<%di" % (len(b) // 4), b)
    diffs = [(i, A[i], B[i]) for i in range(min(len(A), len(B))) if A[i] != B[i]]
    print("diffs", len(diffs), diffs[:10])
m1 = json.load(open(r"results\phase0\baseline-n256\measurement.json"))
m2 = json.load(open(r"results\phase0\phase2-tile1\measurement.json"))
for k in ("steady_ms_median", "steady_tok_per_s", "exit_code", "memory"):
    print(k, m1.get(k), "->", m2.get(k))
