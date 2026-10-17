"""
Sabah / runtime / ngram_prefetch

The n-gram lookup table is the other half of the artifact: ~28.8 GB at IQ4_NL
on the Qwen3.8-Flash-Next family, and it is a plain row GATHER, not a matmul.
Two consequences decide how it should be treated:

  * It never needs to be resident. One token reads one row (a few hundred
    bytes). Keeping 28.8 GB of table in RAM or VRAM to serve a handful of rows
    is pure waste, and that memory is worth far more as expert warm tier.
  * The rows a token will read are KNOWN BEFORE the forward pass. The n-gram
    index is a deterministic function of the input tokens, so the gather can be
    prefetched and its pages dropped again.

So the table is memory-mapped READ-ONLY, the rows for the current token are
hinted into the working set, and the previously used pages are advised away.
The table never competes with experts for RAM.

`index_fn` is the one model-specific piece: it maps token ids to n-gram row
indices. The exact scheme (n, hashing, table geometry) belongs to the model, so
it is INJECTED rather than guessed. `rolling_hash_indices` is provided for
plumbing and is labelled a placeholder.

The original GGUF is opened read-only and is not modified. This module touches
no GPU runtime, so it imports and tests without CUDA.
"""
from __future__ import annotations

import os
import ctypes
import mmap as _mmap
import platform
import threading
from typing import Callable, List, Optional, Sequence

import numpy as np

from sabah.runtime.expert_bank import shard_paths

_PAGE = _mmap.PAGESIZE


# ---------------------------------------------------------------------------
# row geometry. ggml lays an embedding/gather table out as [ne0, ne1, ...] with
# the row index on the SECOND axis, and row stride = ne0 elements. Quantised
# rows are a whole number of blocks, so row_bytes is uniform = bytes / n_rows.
# ---------------------------------------------------------------------------
def row_shape(tensor: dict) -> tuple:
    shp = list(tensor.get("shape") or [])
    n_rows = int(shp[1]) if len(shp) > 1 else 1
    row_elems = int(shp[0]) if shp else 0
    nbytes = int(tensor["bytes"])
    row_bytes = nbytes // n_rows if n_rows else nbytes
    return n_rows, row_elems, row_bytes


# ---------------------------------------------------------------------------
# OS page hints. Best-effort everywhere: a failed hint is not an error, it just
# means the OS will decide on its own.
# ---------------------------------------------------------------------------
_IS_WIN = platform.system() == "Windows"


class _MEMORY_RANGE_ENTRY(ctypes.Structure):
    _fields_ = [("VirtualAddress", ctypes.c_void_p),
                ("NumberOfBytes", ctypes.c_size_t)]


if _IS_WIN:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.GetCurrentProcess.restype = ctypes.c_void_p
    _k32.PrefetchVirtualMemory.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.POINTER(_MEMORY_RANGE_ENTRY),
                                           ctypes.c_ulong]
    _k32.PrefetchVirtualMemory.restype = ctypes.c_int
    _HAVE_PREFETCH = True
    _HAVE_DISCARD = hasattr(_k32, "DiscardVirtualMemory")
    if _HAVE_DISCARD:
        _k32.DiscardVirtualMemory.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _k32.DiscardVirtualMemory.restype = ctypes.c_ulong
    _proc = _k32.GetCurrentProcess()
else:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    _libc.madvise.restype = ctypes.c_int
    # Linux values; macOS differs and must be adjusted if this ever runs there.
    _MADV_WILLNEED = 3
    _MADV_DONTNEED = 4
    _HAVE_PREFETCH = True
    _HAVE_DISCARD = True
    _proc = None


def page_hint(addr: int, length: int) -> bool:
    """Ask the OS to bring [addr, addr+length) into the working set."""
    if length <= 0:
        return False
    if _IS_WIN:
        e = _MEMORY_RANGE_ENTRY(ctypes.c_void_p(addr), ctypes.c_size_t(length))
        return bool(_k32.PrefetchVirtualMemory(_proc, 1, ctypes.byref(e), 0))
    return _libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                         _MADV_WILLNEED) == 0


def page_release(addr: int, length: int) -> bool:
    """Best-effort: let the OS drop [addr, addr+length) from the working set.

    On Windows this uses DiscardVirtualMemory, which is documented for private
    memory and may be refused for a file-backed view. A refusal is not fatal:
    the pages simply stay reclaimable by the OS under pressure. A reader that
    needs a hard guarantee should use the `pread` path instead of mmap.
    """
    if length <= 0:
        return False
    if _IS_WIN:
        if not _HAVE_DISCARD:
            return False
        return _k32.DiscardVirtualMemory(ctypes.c_void_p(addr),
                                         ctypes.c_size_t(length)) == 0
    return _libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(length),
                         _MADV_DONTNEED) == 0


def _runs(indices: Sequence[int], row_bytes: int, base_offset: int,
          limit: Optional[int] = None) -> List[tuple]:
    """Merge sorted row indices into contiguous byte runs: many rows, few calls.

    Ranges are page-aligned outward, because the page-hint calls are
    page-granular (madvise on POSIX rejects an unaligned address; the Windows
    call rounds but aligning costs nothing and keeps the two paths identical).
    `limit` (inclusive end) clamps the outward alignment so a hint never runs
    past the mapping's last byte.
    """
    idx = sorted({int(i) for i in indices})
    out = []
    if not idx:
        return out
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        out.append((start, prev - start + 1))
        start = prev = i
    out.append((start, prev - start + 1))
    runs = []
    for s, n in out:
        lo = base_offset + s * row_bytes
        hi = base_offset + (s + n) * row_bytes
        lo -= lo % _PAGE
        hi = ((hi + _PAGE - 1) // _PAGE) * _PAGE
        if limit is not None:
            if lo >= limit:
                continue
            hi = min(hi, limit)
        runs.append((lo, hi - lo))
    return runs


# ---------------------------------------------------------------------------
# index function. THE model-specific piece.
# ---------------------------------------------------------------------------
NgramIndexFn = Callable[[Sequence[int]], Sequence[int]]


def rolling_hash_indices(tokens: Sequence[int], *, table_rows: int,
                         n: int = 4, seed: int = 0x9E3779B97F4A7C15
                         ) -> List[int]:
    """PLACEHOLDER n-gram index: a rolling polynomial hash of the last `n`
    tokens, folded into [0, table_rows).

    This is NOT the model's scheme. Replace it with the real one (read it from
    the GGUF metadata / the model source) before trusting prefetch quality.
    Prefetch is advice, so a wrong scheme costs bandwidth, never correctness.
    """
    if table_rows <= 0:
        return []
    mask = (1 << 64) - 1
    acc = 0
    out = []
    for t in tokens:
        acc = ((acc * 1000003) ^ (int(t) & 0xFFFFFFFF) ^ seed) & mask
        out.append(acc % table_rows)
    return out


class NgramTable:
    """Read-only random access to an n-gram gather table, with page hints."""

    def __init__(self, model_profile, tensor: Optional[dict] = None,
                 name_hint: str = "per_layer_token_embd"):
        cands = [t for t in getattr(model_profile, "lookup_tensors", [])
                 if name_hint in t["name"]]
        if not cands:
            cands = list(getattr(model_profile, "lookup_tensors", []))
        if tensor is not None:
            cands = [tensor]
        if not cands:
            raise ValueError("model profile exposes no lookup tensors; "
                             "re-inspect the model with this Sabah version")
        self.tensors = cands
        self.tensor = max(cands, key=lambda t: int(t["bytes"]))
        self.n_rows, self.row_elems, self.row_bytes = row_shape(self.tensor)
        self.qtype = str(self.tensor["qtype"])

        paths = shard_paths(model_profile.path)
        self._path = paths[int(self.tensor["shard"])]
        self._map = np.memmap(self._path, dtype=np.uint8, mode="r")
        self._base = int(self._map.ctypes.data)
        self._t0 = int(self.tensor["offset"])
        if self._t0 + int(self.tensor["bytes"]) > self._map.size:
            raise IOError("n-gram tensor range exceeds its shard")

        self._lock = threading.RLock()
        self.hinted_bytes = 0
        self.released_bytes = 0

    # -- addressing ---------------------------------------------------------
    def addr(self, index: int) -> int:
        return self._base + self._t0 + index * self.row_bytes

    def row(self, index: int) -> np.ndarray:
        if not (0 <= index < self.n_rows):
            raise IndexError("n-gram row %d out of range [0,%d)"
                             % (index, self.n_rows))
        o = self._t0 + index * self.row_bytes
        v = self._map[o:o + self.row_bytes]
        v.setflags(write=False)
        return v

    def rows(self, indices: Sequence[int]) -> List[np.ndarray]:
        return [self.row(int(i)) for i in indices]

    def gather(self, indices: Sequence[int], out: Optional[np.ndarray] = None
               ) -> np.ndarray:
        """Copy the rows into a contiguous buffer (a small pinned staging buffer
        in the streaming path)."""
        idx = [int(i) for i in indices]
        if out is None:
            out = np.empty((len(idx), self.row_bytes), dtype=np.uint8)
        if out.shape != (len(idx), self.row_bytes):
            raise ValueError("out has shape %s, expected (%d,%d)"
                             % (out.shape, len(idx), self.row_bytes))
        with self._lock:
            for k, i in enumerate(idx):
                out[k] = self.row(i)
        return out

    def pread_bytes(self, index: int) -> bytes:
        """Read one row straight from the file, without touching the mapping.

        Used when a caller wants the OS page cache to hold as little of the
        table as possible. Goes through the normal file API, so on Windows it
        still passes the system cache; it simply does not fault the mapping.
        """
        off = self._t0 + int(index) * self.row_bytes
        with self._lock:
            with open(self._path, "rb", buffering=0) as f:
                f.seek(off)
                return f.read(self.row_bytes)

    # -- hints --------------------------------------------------------------
    def prefetch(self, indices: Sequence[int]) -> int:
        if self._map is None:
            return 0
        n = 0
        lim = self._base + self._map.size
        for addr, ln in _runs(indices, self.row_bytes, self._base + self._t0, lim):
            if page_hint(addr, ln):
                n += ln
        self.hinted_bytes += n
        return n

    def release(self, indices: Sequence[int]) -> int:
        if self._map is None:
            return 0
        n = 0
        lim = self._base + self._map.size
        for addr, ln in _runs(indices, self.row_bytes, self._base + self._t0, lim):
            if page_release(addr, ln):
                n += ln
        self.released_bytes += n
        return n

    def close(self):
        with self._lock:
            self._map = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def describe(self) -> str:
        return ("n-gram tab : %s\n"
                "geometry   : %d rows x %d elems/row, %d B/row, %s\n"
                "size       : %.3f GB (disk-resident, read-only)\n"
                "hints      : prefetch %s, discard %s"
                % (os.path.basename(self._path), self.n_rows, self.row_elems,
                   self.row_bytes, self.qtype,
                   int(self.tensor["bytes"]) / 1e9,
                   "yes" if _HAVE_PREFETCH else "no",
                   "yes" if _HAVE_DISCARD else "best-effort"))


class NgramPrefetcher:
    """Turn tokens into n-gram rows and keep the mapping cold.

    `step(tokens)` prefetches the rows those tokens will read and releases the
    rows used by the previous step, so the working set stays a few pages wide.
    """

    def __init__(self, table: NgramTable, index_fn: Optional[NgramIndexFn] = None,
                 n: int = 4, release_previous: bool = True):
        self.table = table
        self.n = n
        self.index_fn = index_fn
        self.release_previous = release_previous
        self.enabled = True
        self._prev: List[int] = []
        self.steps = 0
        self.tokens = 0

    def indices(self, tokens: Sequence[int]) -> List[int]:
        if self.index_fn is not None:
            return [int(i) for i in self.index_fn(tokens)]
        return rolling_hash_indices(tokens, table_rows=self.table.n_rows,
                                    n=self.n)

    def step(self, tokens: Sequence[int]) -> List[int]:
        idx = self.indices(tokens)
        if not self.enabled:
            return idx
        if self.release_previous and self._prev:
            self.table.release(self._prev)
        self.table.prefetch(idx)
        self._prev = idx
        self.steps += 1
        self.tokens += len(tokens)
        return idx

    def close(self):
        if self._prev:
            self.table.release(self._prev)
            self._prev = []
        self.table.close()

    def stats(self) -> dict:
        return dict(steps=self.steps, tokens=self.tokens,
                    rows=self.table.n_rows, row_bytes=self.table.row_bytes,
                    table_gb=int(self.table.tensor["bytes"]) / 1e9,
                    hinted_bytes=self.table.hinted_bytes,
                    released_bytes=self.table.released_bytes,
                    prefetch_supported=_HAVE_PREFETCH,
                    discard_supported=_HAVE_DISCARD)
