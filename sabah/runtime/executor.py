"""
Sabah / runtime / executor

Executes one routed-MoE block on the GPU.

The contract this file exists to keep:

    given router output (expert ids, routing weights), the arithmetic performed
    is exactly

        y = sum_k  w_k * W_down[e_k]^T ( silu(W_gate[e_k]^T x) * W_up[e_k]^T x )

    over the ids the router produced, using those experts' real quantized
    weights. Residency, ordering and transfer scheduling are free variables.
    The set {e_k} and the weights {w_k} are not.
"""
from __future__ import annotations

import ctypes
import numpy as np

from sabah.runtime import rt


class MoEExecutor:
    def __init__(self, bank, tier, max_used: int = 16):
        self.bank = bank
        self.tier = tier
        self.d_model = bank.model.d_model
        self.ff = bank.model.expert_ff
        self.max_used = max_used

        L = rt.lib()
        self.d_x   = rt.check_ptr(L.sabah_dev_alloc(self.d_model * 4), "d_x")
        self.d_out = rt.check_ptr(L.sabah_dev_alloc(self.d_model * 4), "d_out")
        self.d_h   = rt.check_ptr(L.sabah_dev_alloc(max_used * self.ff * 4), "d_h")
        self.d_w   = rt.check_ptr(L.sabah_dev_alloc(max_used * 4), "d_w")
        self.d_ptr = rt.check_ptr(L.sabah_dev_alloc(3 * max_used * 8), "d_ptr")

    # ------------------------------------------------------------------
    def upload_x(self, x: np.ndarray):
        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.size != self.d_model:
            raise ValueError("x has %d elements, expected %d" % (x.size, self.d_model))
        rt.check(rt.lib().sabah_memcpy_h2d(
            ctypes.c_void_p(self.d_x), x.ctypes.data_as(ctypes.c_void_p),
            x.nbytes), "upload x")

    def download_out(self) -> np.ndarray:
        out = np.empty(self.d_model, dtype=np.float32)
        rt.check(rt.lib().sabah_memcpy_d2h(
            out.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(self.d_out),
            out.nbytes), "download out")
        return out

    def zero_out(self):
        rt.check(rt.lib().sabah_memset_d(ctypes.c_void_p(self.d_out), 0,
                                         self.d_model * 4), "zero out")

    # ------------------------------------------------------------------
    def run_block(self, block: int, ids, weights, accumulate: bool = False):
        """Run one MoE block. `ids` and `weights` come from the router."""
        ids = [int(i) for i in ids]
        w = np.ascontiguousarray(weights, dtype=np.float32)
        n = len(ids)
        if n != w.size:
            raise ValueError("%d ids but %d weights" % (n, w.size))
        if n > self.max_used:
            raise ValueError("n_used %d exceeds executor capacity %d" % (n, self.max_used))

        slots = self.tier.ensure(block, ids)
        ptrs = self.tier.slot_ptrs(block, slots)

        flat = (ctypes.c_void_p * (3 * n))()
        for i, r in enumerate(self.bank.ROLES):          # gate, up, down
            for k in range(n):
                flat[i * n + k] = ptrs[r][k]

        L = rt.lib()
        rt.check(L.sabah_memcpy_h2d(ctypes.c_void_p(self.d_ptr),
                                    ctypes.cast(flat, ctypes.c_void_p),
                                    3 * n * 8), "upload ptrs")
        rt.check(L.sabah_memcpy_h2d(ctypes.c_void_p(self.d_w),
                                    w.ctypes.data_as(ctypes.c_void_p),
                                    w.nbytes), "upload weights")

        rt.check(L.sabah_moe_block(
            ctypes.c_void_p(self.d_x), ctypes.c_void_p(self.d_out),
            ctypes.c_void_p(self.d_h), ctypes.c_void_p(self.d_ptr),
            ctypes.c_void_p(self.d_w),
            self.d_model, self.ff, n,
            self.bank.qtype(block, "gate"), self.bank.qtype(block, "up"),
            self.bank.qtype(block, "down"), 1 if accumulate else 0),
            "moe_block")
        self.tier.tel.blocks_executed += 1

    def sync(self):
        rt.check(rt.lib().sabah_sync_compute(), "sync compute")

    def free(self):
        L = rt.lib()
        for p in (self.d_x, self.d_out, self.d_h, self.d_w, self.d_ptr):
            L.sabah_dev_free(ctypes.c_void_p(p))


# ---------------------------------------------------------------------------
# CPU reference. Deliberately written against gguf-py's dequantizer and plain
# numpy, sharing no code with the CUDA path, so agreement means something.
# ---------------------------------------------------------------------------
def reference_block(bank, block: int, x: np.ndarray, ids, weights) -> np.ndarray:
    import sys
    sys.path.insert(0, r"D:/llama-glm53/gguf-py")
    from gguf import quants
    from gguf.constants import GGMLQuantizationType

    d_model = bank.model.d_model
    ff = bank.model.expert_ff
    x = np.asarray(x, dtype=np.float32)

    def W(role, e, rows, cols):
        d = bank.desc[(block, role)]
        raw = np.asarray(bank.slice(block, e, role))
        rb = rt.row_bytes(rt.QTYPE[d["qtype"]], cols)
        blk = raw.reshape(rows, rb)
        return np.asarray(
            quants.dequantize(blk, getattr(GGMLQuantizationType, d["qtype"])),
            dtype=np.float32).reshape(rows, cols)

    out = np.zeros(d_model, dtype=np.float32)
    for w_k, e in zip(np.asarray(weights, dtype=np.float32), ids):
        g = W("gate", int(e), ff, d_model) @ x
        u = W("up",   int(e), ff, d_model) @ x
        h = (g / (1.0 + np.exp(-g))) * u
        out += w_k * (W("down", int(e), d_model, ff) @ h)
    return out
