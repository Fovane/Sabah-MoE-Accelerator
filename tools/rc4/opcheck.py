"""Operation-level exactness of MUL_MAT_ID in each execution path.

For every captured expert projection the path's OWN input (src[1]) and ids
(src[2]) are replayed in float64 against gguf-py's dequantized weights. This
separates what the operation computed from what flowed into it, so upstream
drift cannot be mistaken for an arithmetic error.

Besides exact math, two emulations of llama.cpp's activation quantization are
evaluated, because ggml's quantized matmuls do not multiply by the float
activation at all:

  q8_1  per-32 block, d = amax/127, q = round(x/d)   (CUDA mmvq/mmq, and CPU for
                                                      Q5_1 weights)
  q8_K  per-256 block, iscale = -127/max_signed,      (CPU for Q4_K/Q5_K weights)
        q = round(iscale*x) clipped to 127
  q8_0  per-32 block, as q8_1                          (CPU for Q8_0 weights)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, "D:/sabah_scaling/v5")
sys.path.insert(0, "D:/llama-glm53/gguf-py")
from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402

from sabah.core.model_inspector import inspect_model  # noqa: E402
from sabah.runtime.expert_bank import ExpertBank  # noqa: E402
from sabah.runtime import rt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Run, metrics  # noqa: E402

MODEL = ("D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
         "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")

_bank = None
_cache = {}


def bank():
    global _bank
    if _bank is None:
        prof = inspect_model(MODEL)
        _bank = ExpertBank(prof, mode="mmap")
    return _bank


def W(block, role, e, rows, cols):
    key = (block, role, e)
    if key not in _cache:
        b = bank()
        qt = b.desc[(block, role)]["qtype"]
        raw = np.asarray(b.slice(block, e, role)).reshape(rows, rt.row_bytes(rt.QTYPE[qt], cols))
        _cache[key] = np.asarray(quants.dequantize(raw, getattr(Q, qt)), np.float64).reshape(rows, cols)
        if len(_cache) > 1500:
            _cache.pop(next(iter(_cache)))
    return _cache[key]


def q8_1(x):
    x = np.asarray(x, np.float32).reshape(-1, 32)
    amax = np.abs(x).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float32)
    q = np.where(d > 0, np.round(x / np.where(d > 0, d, 1)), 0)
    return (q * d).reshape(-1).astype(np.float64)


def q8_K(x):
    x = np.asarray(x, np.float32).reshape(-1, 256)
    idx = np.abs(x).argmax(axis=1)
    mx = x[np.arange(x.shape[0]), idx][:, None]
    iscale = np.where(mx != 0, -127.0 / np.where(mx != 0, mx, 1), 0).astype(np.float32)
    q = np.minimum(127, np.round(iscale * x))
    d = np.where(iscale != 0, 1.0 / np.where(iscale != 0, iscale, 1), 0)
    return (q * d).reshape(-1).astype(np.float64)


CPU_ACT = {"Q4_K": q8_K, "Q5_K": q8_K, "Q5_1": q8_1, "Q8_0": q8_1}


def check_op(run, block, step, role, max_tokens=None):
    name = {"gate": "ffn_moe_gate", "up": "ffn_moe_up", "down": "ffn_moe_down"}[role]
    k_out, k_in, k_ids = "%s_L%d" % (name, block), "%s_in_L%d" % (name, block), "%s_ids_L%d" % (name, block)
    if not (run.has(k_out, step) and run.has(k_in, step) and run.has(k_ids, step)):
        return None
    out = run.get(k_out, step)            # [T, K, rows]
    xin = run.get(k_in, step)             # [T, ne11, cols]
    ids = run.get(k_ids, step)            # [T, K]
    ids = ids.reshape(out.shape[0], -1)
    T, K, rows = out.shape
    cols = xin.shape[-1]
    ne11 = xin.shape[1]
    qt = bank().desc[(block, role)]["qtype"]
    tsel = range(T) if max_tokens is None else range(min(T, max_tokens))
    got, exact, e81, ecpu = [], [], [], []
    for t in tsel:
        for i in range(K):
            w = W(block, role, int(ids[t, i]), rows, cols)
            x = xin[t, i % ne11].astype(np.float64)
            got.append(out[t, i])
            exact.append(w @ x)
            e81.append(w @ q8_1(x))
            ecpu.append(w @ CPU_ACT[qt](x))
    got, exact, e81, ecpu = map(np.concatenate, (got, exact, e81, ecpu))
    return dict(block=block, step=step, role=role, qtype=qt, tokens=len(tsel), slots=K,
                vs_exact=metrics(got, exact),
                vs_emul_q8_1=metrics(got, e81),
                vs_emul_cpu_act=metrics(got, ecpu),
                buf=run.buf(k_out, step))


if __name__ == "__main__":
    runs = {m: Run(p) for m, p in (a.split("=") for a in sys.argv[1].split(","))}
    blocks = [int(b) for b in sys.argv[2].split(",")]
    step = int(sys.argv[3])
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else None
    res = []
    print("%-6s %5s %-5s %-5s %-7s %11s %11s %11s" %
          ("path", "block", "op", "qtype", "buf", "vs exact", "vs q8_1em", "vs cpuActEm"))
    for b in blocks:
        for role in ("gate", "up", "down"):
            for m, r in runs.items():
                c = check_op(r, b, step, role, max_tokens)
                if c is None:
                    continue
                c["path"] = m
                res.append(c)
                print("%-6s %5d %-5s %-5s %-7s %11.3e %11.3e %11.3e" %
                      (m, b, role, c["qtype"], c["buf"][:7], c["vs_exact"]["rel_l2"],
                       c["vs_emul_q8_1"]["rel_l2"], c["vs_emul_cpu_act"]["rel_l2"]))
    if len(sys.argv) > 5:
        json.dump(res, open(sys.argv[5], "w"), indent=1)
