"""
Sabah / runtime / rt

ctypes binding to sabah_rt.dll. Thin on purpose: policy lives in Python, the
arithmetic lives in CUDA, and this file only moves arguments between them.

Every entry point returns an error rather than raising deep inside a kernel
launch, and `check()` turns that into an exception carrying the CUDA message.
"""
from __future__ import annotations

import os
import sys
import ctypes
import platform

HERE = os.path.dirname(os.path.abspath(__file__))

# ggml type codes, so a Sabah trace and a llama.cpp trace mean the same thing
QTYPE = {"Q5_1": 7, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13}
QTYPE_NAME = {v: k for k, v in QTYPE.items()}

# (elements per block, bytes per block) - asserted against the model on load
QBLOCK = {7: (32, 24), 8: (32, 34), 12: (256, 144), 13: (256, 176)}


class RuntimeUnavailable(Exception):
    """The CUDA runtime library is missing or could not be loaded."""


class SabahCudaError(Exception):
    pass


def _libname() -> str:
    return "sabah_rt.dll" if platform.system() == "Windows" else "libsabah_rt.so"


def library_path() -> str:
    return os.environ.get("SABAH_RT_LIB") or os.path.join(HERE, "cuda", _libname())


_lib = None


def lib():
    """Load the runtime library once, with an actionable error if it is absent."""
    global _lib
    if _lib is not None:
        return _lib
    p = library_path()
    if not os.path.exists(p):
        raise RuntimeUnavailable(
            "Sabah CUDA runtime not built: %s\n"
            "Build it with:\n"
            "    cd sabah/runtime/cuda && nvcc -O3 -shared -o %s sabah_rt.cu"
            % (p, _libname()))
    try:
        L = ctypes.CDLL(p)
    except OSError as e:
        raise RuntimeUnavailable("could not load %s: %s" % (p, e))

    c_p, c_vp, c_sz = ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t
    sig = {
        "sabah_rt_last_error":    ([], ctypes.c_char_p),
        "sabah_rt_init":          ([ctypes.c_int], ctypes.c_int),
        "sabah_rt_shutdown":      ([], ctypes.c_int),
        "sabah_rt_device_count":  ([], ctypes.c_int),
        "sabah_dev_alloc":        ([c_sz], c_vp),
        "sabah_dev_free":         ([c_vp], None),
        "sabah_host_alloc":       ([c_sz], c_vp),
        "sabah_host_free":        ([c_vp], None),
        "sabah_dev_mem":          ([ctypes.POINTER(c_sz), ctypes.POINTER(c_sz)], ctypes.c_int),
        "sabah_memcpy_h2d":       ([c_vp, c_vp, c_sz], ctypes.c_int),
        "sabah_memcpy_d2h":       ([c_vp, c_vp, c_sz], ctypes.c_int),
        "sabah_memset_d":         ([c_vp, ctypes.c_int, c_sz], ctypes.c_int),
        "sabah_h2d_async":        ([c_vp, c_vp, c_sz], ctypes.c_int),
        "sabah_event_create":     ([], c_vp),
        "sabah_event_destroy":    ([c_vp], None),
        "sabah_event_record_copy":    ([c_vp], ctypes.c_int),
        "sabah_event_record_compute": ([c_vp], ctypes.c_int),
        "sabah_compute_wait_event":   ([c_vp], ctypes.c_int),
        "sabah_event_elapsed_ms": ([c_vp, c_vp], ctypes.c_float),
        "sabah_event_sync":       ([c_vp], ctypes.c_int),
        "sabah_event_query":      ([c_vp], ctypes.c_int),
        "sabah_sync_copy":        ([], ctypes.c_int),
        "sabah_sync_compute":     ([], ctypes.c_int),
        "sabah_sync_all":         ([], ctypes.c_int),
        "sabah_moe_block":        ([c_vp, c_vp, c_vp, c_vp, c_vp,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int], ctypes.c_int),
        "sabah_dequant":          ([c_vp, c_vp, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int], ctypes.c_int),
        "sabah_row_bytes":        ([ctypes.c_int, ctypes.c_int], c_sz),
    }
    for name, (argt, rest) in sig.items():
        fn = getattr(L, name)
        fn.argtypes = argt
        fn.restype = rest
    _lib = L
    return _lib


def last_error() -> str:
    try:
        return (lib().sabah_rt_last_error() or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def check(rc, what="cuda call"):
    if rc is None or (isinstance(rc, int) and rc != 0):
        raise SabahCudaError("%s failed: %s" % (what, last_error()))
    return rc


def check_ptr(p, what="allocation"):
    if not p:
        raise SabahCudaError("%s failed: %s" % (what, last_error()))
    return p


def row_bytes(qt: int, n: int) -> int:
    """Bytes occupied by one row of n elements at quant type qt."""
    be, bb = QBLOCK[qt]
    if n % be:
        raise ValueError("row of %d elements is not a whole number of %s blocks"
                         % (n, QTYPE_NAME.get(qt, qt)))
    return n // be * bb


def available() -> bool:
    try:
        lib()
        return True
    except RuntimeUnavailable:
        return False


def device_count() -> int:
    try:
        return int(lib().sabah_rt_device_count())
    except RuntimeUnavailable:
        return 0
