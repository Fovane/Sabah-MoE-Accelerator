"""The ggml MUL_MAT_ID contract, exercised through Sabah's native ABI.

This is the test whose absence let RC1-RC3 ship a structural bug. The isolated
block tests drove `sabah_moe_block`, which builds its own per-expert SwiGLU
input, so they never exercised the llama.cpp-facing entry point with more than
one input row. In the real graph the down projection has one input row per
selected expert (src1->ne[1] == n_used); Sabah read row 0 for every slot.

Checked here, on real GGUF expert bytes:

  * broadcast input (ne11 == 1, gate/up) and per-slot input (ne11 == n_used,
    down) both equal exact float64 math on the SAME slot's row
  * the per-slot case is NOT satisfied by the old row-0 behaviour (the test
    would have failed on v1)
  * a wrong expert is detected by a margin of several orders of magnitude
  * a hot tier far smaller than one call's working set gives the identical
    result (no slot handed to a call is evicted before its kernel runs), and
    every fetched or hit slot is byte-identical to its GGUF range
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

MODEL = os.environ.get(
    "SABAH_TEST_MODEL",
    "D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
GGUF_PY = os.environ.get("SABAH_GGUF_PY", "D:/llama-glm53/gguf-py")
TOL = 2e-6          # fp32 accumulation bound over <=2560 terms (sqrt(K)*2^-24 ~ 3e-6); measured <=7.8e-7


def _available():
    from sabah.runtime import rt
    return rt.available() and rt.device_count() > 0 and os.path.exists(MODEL)


def _child(case: str, hot_bytes: int) -> dict:
    env = dict(os.environ, SABAH_LLAMA_HOT_BYTES=str(hot_bytes), SABAH_LLAMA_VERIFY_BYTES="1")
    out = subprocess.run([sys.executable, __file__, case], env=env, cwd=ROOT,
                         capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-3000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


needs_gpu = pytest.mark.skipif(not _available(), reason="CUDA runtime, GPU or model not available")


@needs_gpu
def test_down_projection_uses_each_slots_own_input_row():
    r = _child("down", 1 << 30)
    assert r["rel_vs_exact"] < TOL, r
    # the v1 behaviour (every slot fed row 0) must be clearly distinguishable
    assert r["exact_vs_row0"] > 1e-1, r
    assert r["verify_fail"] == 0 and r["verify_ok"] > 0, r


@needs_gpu
def test_broadcast_gate_projection():
    r = _child("gate", 1 << 30)
    assert r["rel_vs_exact"] < TOL, r
    assert r["verify_fail"] == 0, r


@needs_gpu
def test_wrong_expert_sentinel_is_far_outside_tolerance():
    r = _child("down", 1 << 30)
    assert r["rel_wrong_expert"] > 1e4 * max(r["rel_vs_exact"], 1e-9), r


@needs_gpu
def test_tiny_hot_tier_is_exact_and_never_evicts_a_live_slot():
    big = _child("down", 1 << 30)
    tiny = _child("down", 3 * 1228800)       # three Q5_1 down experts
    assert tiny["rel_vs_exact"] < TOL, tiny
    assert tiny["out_sha"] == big["out_sha"], "result depends on cache capacity"
    assert tiny["overflow"] > 0, "the case did not exceed capacity within one call"
    assert tiny["verify_fail"] == 0 and tiny["verify_ok"] > 0, tiny


# ---------------------------------------------------------------------------
# child process: one MUL_MAT_ID call through the native ABI
# ---------------------------------------------------------------------------
def _run_case(case: str) -> dict:
    import hashlib
    sys.path.insert(0, GGUF_PY)
    from gguf import quants
    from gguf.constants import GGMLQuantizationType as Q
    from sabah.core.model_inspector import inspect_model
    from sabah.runtime import rt
    from sabah.runtime.expert_bank import shard_paths

    L = rt.lib()
    fn = L.sabah_rt_mul_mat_id_v2
    c_vp, c_sz, c_i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    fn.argtypes = [c_vp, c_sz, c_sz, c_vp, c_sz, c_i, c_sz, c_vp, c_sz, c_sz, c_i, c_i,
                   c_vp, c_sz, c_sz, c_i, c_i, c_i, c_i, c_vp]
    fn.restype = c_i
    diag = L.sabah_rt_get_diag
    diag.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)] * 5
    rt.check(L.sabah_rt_init(0), "rt_init")

    prof = inspect_model(MODEL)
    block, role = 0, ("down" if case == "down" else "gate")
    t = next(x for x in prof.expert_tensors if x["block"] == block and x["role"] == role)
    k, rows, n_exp = t["shape"][0], t["shape"][1], t["shape"][2]
    qt = rt.QTYPE[t["qtype"]]
    rb = rt.row_bytes(qt, k)
    mm = np.memmap(shard_paths(prof.path)[t["shard"]], dtype=np.uint8, mode="r")
    src = mm[t["offset"]:t["offset"] + t["tensor_bytes"]]
    src_ptr = src.ctypes.data

    rng = np.random.default_rng(4)
    T, K = 6, 10
    ids = np.stack([rng.choice(n_exp, K, replace=False) for _ in range(T)]).astype(np.int32)
    x_rows = K if case == "down" else 1
    x = (rng.standard_normal((T, x_rows, k)) * 0.1).astype(np.float32)

    def dev(a):
        p = rt.check_ptr(L.sabah_dev_alloc(a.nbytes), "alloc")
        rt.check(L.sabah_memcpy_h2d(c_vp(p), a.ctypes.data_as(c_vp), a.nbytes), "h2d")
        return p

    d_x, d_ids = dev(x), dev(ids)
    d_dst = rt.check_ptr(L.sabah_dev_alloc(T * K * rows * 4), "alloc dst")
    rc = fn(c_vp(src_ptr), t["per_expert_bytes"], rb,
            c_vp(d_x), k * 4, x_rows, x_rows * k * 4,
            c_vp(d_ids), 4, K * 4, K, T,
            c_vp(d_dst), rows * 4, K * rows * 4,
            rows, k, n_exp, qt, None)
    if rc != 0:
        raise RuntimeError(rt.last_error())
    rt.check(L.sabah_sync_all(), "sync")
    out = np.empty((T, K, rows), np.float32)
    rt.check(L.sabah_memcpy_d2h(out.ctypes.data_as(c_vp), c_vp(d_dst), out.nbytes), "d2h")

    def W(e):
        raw = np.asarray(src[e * t["per_expert_bytes"]:(e + 1) * t["per_expert_bytes"]]).reshape(rows, rb)
        return np.asarray(quants.dequantize(raw, getattr(Q, t["qtype"])), np.float64).reshape(rows, k)

    exact = np.empty_like(out, dtype=np.float64)
    row0 = np.empty_like(exact)
    wrong = np.empty_like(exact)
    for ti in range(T):
        for s in range(K):
            w = W(int(ids[ti, s]))
            xr = x[ti, s % x_rows].astype(np.float64)
            exact[ti, s] = w @ xr
            row0[ti, s] = w @ x[ti, 0].astype(np.float64)
            wrong[ti, s] = W((int(ids[ti, s]) + 1) % n_exp) @ xr

    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
    v = [ctypes.c_ulonglong() for _ in range(5)]
    diag(*[ctypes.byref(z) for z in v])
    return dict(case=case, rel_vs_exact=rel(out, exact),
                exact_vs_row0=rel(row0, exact) if x_rows > 1 else 0.0,
                rel_wrong_expert=rel(wrong, exact),
                out_sha=hashlib.sha256(out.tobytes()).hexdigest(),
                calls=v[0].value, tokens=v[1].value, overflow=v[2].value,
                verify_ok=v[3].value, verify_fail=v[4].value)


if __name__ == "__main__":
    print(json.dumps(_run_case(sys.argv[1])))
