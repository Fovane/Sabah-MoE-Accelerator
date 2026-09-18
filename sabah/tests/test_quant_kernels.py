"""
Do Sabah's CUDA kernels read the real quant formats correctly?

This is the test that has to pass before any speed number means anything. If
the dequantization is wrong, the model is silently a different model, and every
correctness claim Sabah makes is void.

Method: take REAL bytes out of the user's GGUF - not synthetic blocks - and
compare
    gguf-py's reference dequantizer          (trusted)
against
    the exact device code path the matvec kernels use  (under test)

The device side recovers element k by dotting the sub-block with the k-th unit
vector, so it exercises `sub32_*` itself. A separate dequant implementation
would prove nothing about the kernels that actually run.
"""
from __future__ import annotations

import os
import sys
import ctypes

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, r"D:/llama-glm53/gguf-py")

import numpy as np
from sabah.runtime import rt

DEFAULT_MODEL = (r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                 r"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")

# rows to check per quant type; each row is a full matvec row of the real model
N_ROWS = 64


def shard_paths(path):
    import re
    m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", path)
    if not m:
        return [path]
    n = int(m.group(2))
    base = path[:m.start()]
    return [("%s-%05d-of-%05d.gguf" % (base, i, n)) for i in range(1, n + 1)]


def collect_samples(model_path):
    """One real tensor per quant type Sabah must execute, with its raw bytes."""
    from gguf import GGUFReader
    want = dict(rt.QTYPE)              # name -> code
    found = {}
    for p in shard_paths(model_path):
        if not os.path.exists(p):
            continue
        r = GGUFReader(p, "r")
        for t in r.tensors:
            name = t.tensor_type.name
            if name not in want or name in found:
                continue
            if "_exps." not in t.name:      # only expert tensors matter here
                continue
            shape = tuple(int(x) for x in t.shape)
            row_elems = shape[0]
            rb = rt.row_bytes(want[name], row_elems)
            raw = np.asarray(t.data).reshape(-1).view(np.uint8)
            need = N_ROWS * rb
            if raw.size < need:
                continue
            found[name] = dict(tensor=t.name, qtype=want[name],
                               row_elems=row_elems, row_bytes=rb,
                               raw=np.array(raw[:need]))    # copy off the mmap
        if len(found) == len(want):
            break
    return found


def reference(raw, qtype_name, row_bytes, row_elems, n_rows):
    from gguf import quants
    from gguf.constants import GGMLQuantizationType
    blk = raw.reshape(n_rows, row_bytes)
    out = quants.dequantize(blk, getattr(GGMLQuantizationType, qtype_name))
    return np.asarray(out, dtype=np.float32).reshape(n_rows, row_elems)


def gpu_dequant(raw, qtype_code, row_elems, n_rows):
    L = rt.lib()
    d_src = rt.check_ptr(L.sabah_dev_alloc(raw.nbytes), "dev_alloc src")
    d_dst = rt.check_ptr(L.sabah_dev_alloc(n_rows * row_elems * 4), "dev_alloc dst")
    try:
        rt.check(L.sabah_memcpy_h2d(d_src, raw.ctypes.data_as(ctypes.c_void_p),
                                    raw.nbytes), "h2d")
        rt.check(L.sabah_dequant(d_src, d_dst, qtype_code, n_rows, row_elems),
                 "dequant")
        out = np.empty(n_rows * row_elems, dtype=np.float32)
        rt.check(L.sabah_memcpy_d2h(out.ctypes.data_as(ctypes.c_void_p), d_dst,
                                    out.nbytes), "d2h")
        return out.reshape(n_rows, row_elems)
    finally:
        L.sabah_dev_free(d_src)
        L.sabah_dev_free(d_dst)


def main(model=None):
    model = model or DEFAULT_MODEL
    print("=" * 78)
    print("SABAH - quant kernel correctness vs gguf-py reference")
    print("=" * 78)

    if not rt.available():
        print("CUDA runtime not built; cannot run.")
        return 2
    if rt.device_count() < 1:
        print("no CUDA device visible.")
        return 2
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")

    if not os.path.exists(model):
        print("model not found: %s" % model)
        return 2
    print("model : %s" % os.path.basename(model))
    print("rows  : %d per quant type, taken from real expert tensors\n" % N_ROWS)

    samples = collect_samples(model)
    missing = sorted(set(rt.QTYPE) - set(samples))
    if missing:
        print("note: no expert tensor found for %s in this artifact" % ", ".join(missing))

    print("%-7s %-34s %8s %12s %12s  %s"
          % ("qtype", "tensor", "row", "max|diff|", "rel", "verdict"))
    print("-" * 94)
    failures = []
    for qname in sorted(samples):
        s = samples[qname]
        ref = reference(s["raw"], qname, s["row_bytes"], s["row_elems"], N_ROWS)
        got = gpu_dequant(s["raw"], s["qtype"], s["row_elems"], N_ROWS)

        diff = np.abs(ref - got)
        scale = max(float(np.abs(ref).max()), 1e-30)
        maxd = float(diff.max())
        rel = maxd / scale
        # dequantised values are reconstructed in fp32 from the same integers,
        # so anything beyond fp32 rounding means the layout was read wrongly
        ok = rel < 1e-6
        if not ok:
            failures.append("%s: rel %.3e" % (qname, rel))
        print("%-7s %-34s %8d %12.3e %12.3e  %s"
              % (qname, s["tensor"][:34], s["row_elems"], maxd, rel,
                 "OK" if ok else "MISMATCH"))

    print()
    if failures:
        print("FAILED: %s" % "; ".join(failures))
        print("Sabah must not execute a model whose weights it decodes wrongly.")
        return 1
    if len(samples) < len(rt.QTYPE):
        print("PASSED for the %d types present in this artifact "
              "(%d untested)" % (len(samples), len(rt.QTYPE) - len(samples)))
    else:
        print("PASSED - all %d expert quant types decode identically to the "
              "reference." % len(samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
