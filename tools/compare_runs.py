"""Byte-exact comparison of two diagnostic runs.

    python tools/compare_runs.py results/phase0/run-a results/phase0/run-b

Checks every file the diag writes for decoding drift (fed.i32, argmax.i32,
logits.f32) and reports the first differing byte plus the surrounding values,
so a mismatch can be attributed to a real numeric difference instead of a
generic FAIL.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

FILES = ("fed.i32", "argmax.i32", "logits.f32")


def first_diff(a, b, dtype, name):
    n = min(a.size, b.size)
    if a.size != b.size:
        return f"{name}: size differs {a.size} vs {b.size}"
    idx = np.flatnonzero(a[:n] != b[:n])
    if idx.size == 0:
        return None
    i = int(idx[0])
    va, vb = a[i], b[i]
    if dtype == np.int32:
        detail = f"value {va} vs {vb}"
    else:
        import struct
        detail = (f"value {va!r} vs {vb!r}, bits "
                  f"0x{struct.unpack('<I', struct.pack('<f', float(va)))[0]:08x} vs "
                  f"0x{struct.unpack('<I', struct.pack('<f', float(vb)))[0]:08x}")
    return f"{name}: first diff at element {i} of {n} ({detail}); {idx.size} differ"


def load(path, dtype):
    if not os.path.exists(path):
        return None
    return np.fromfile(path, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--files", default=",".join(FILES))
    args = ap.parse_args()
    ok = True
    for name in [f.strip() for f in args.files.split(",") if f.strip()]:
        dtype = np.float32 if name.endswith(".f32") else np.int32
        a = load(os.path.join(args.a, name), dtype)
        b = load(os.path.join(args.b, name), dtype)
        if a is None or b is None:
            print(f"{name:>12}: MISSING ({'a' if a is None else 'b'})")
            ok = False
            continue
        same = a.size == b.size and np.array_equal(
            a.view(np.uint8), b.view(np.uint8))
        if same:
            print(f"{name:>12}: identical  {a.nbytes} bytes  {a.size} elements")
        else:
            ok = False
            print(f"{name:>12}: MISMATCH   {first_diff(a, b, dtype, name)}")
    print("VERDICT:", "BIT-EXACT" if ok else "DRIFT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
