"""Unbuffered disk probe: how fast can this machine actually read 2 MB experts?

The whole 10-15 tok/s plan hinges on one number: the sustained throughput of
scattered, aligned reads out of the GGUF's expert region. Sequential numbers
(1.56-2.7 GB/s) do not answer it, because expert reads are scattered and the
mmap page-fault path behaves like queue depth 1.

This opens the file with Windows FILE_FLAG_NO_BUFFERING, so the page cache
cannot flatter the result, and uses one handle per worker thread so a real
queue depth can be issued without OVERLAPPED lifetime hazards.

**Every case draws its own offsets.** An earlier version of this probe drew one
offset list and reused it for every queue depth, which is not what the model
does and is worth about 1.5x on this drive: re-reading a list it has just read,
the drive here does 1814-1829 MB/s at depth 4-8, and drawn fresh it does
1127-1457 MB/s (see tools/io_bench.cu, which varies one factor at a time, and
docs/DECODE_BUDGET.md). `--reuse-offsets` reproduces the old number.

    python tools/disk_probe.py --file <model>.gguf
    python tools/disk_probe.py --file <model>.gguf --size-kb 512 --count 400 \
        --qd 1,8,32,64 --json results/disk_probe.json
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import os
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x00000001
OPEN_EXISTING = 3
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_BEGIN = 0

k32 = ctypes.WinDLL("kernel32", use_last_error=True)
k32.CreateFileW.restype = wintypes.HANDLE
k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                            ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                            wintypes.HANDLE]
k32.ReadFile.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD,
                         ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
k32.ReadFile.restype = wintypes.BOOL
k32.SetFilePointerEx.argtypes = [wintypes.HANDLE, ctypes.c_longlong,
                                 ctypes.c_void_p, wintypes.DWORD]
k32.SetFilePointerEx.restype = wintypes.BOOL
k32.CloseHandle.argtypes = [wintypes.HANDLE]


def open_raw(path: str):
    h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS, None)
    if h == INVALID_HANDLE_VALUE or h is None:
        raise OSError("CreateFileW failed: %d" % ctypes.get_last_error())
    return h


def pread_raw(handle, addr: int, offset: int, size: int) -> int:
    """Unbuffered, synchronous read. `addr` and `offset`/`size` must be 512-aligned."""
    if not k32.SetFilePointerEx(handle, offset, None, FILE_BEGIN):
        raise OSError("SetFilePointerEx failed: %d" % ctypes.get_last_error())
    read = wintypes.DWORD(0)
    if not k32.ReadFile(handle, ctypes.c_void_p(addr), size,
                        ctypes.byref(read), None):
        raise OSError("ReadFile failed: %d" % ctypes.get_last_error())
    return read.value


def make_buf(size: int):
    """A 4096-aligned byte buffer; returns (keepalive, aligned_address)."""
    raw = ctypes.create_string_buffer(size + 4096)
    addr = ctypes.addressof(raw)
    if addr % 4096:
        addr += 4096 - (addr % 4096)
    return raw, addr


def run_case(path: str, offsets, size: int, qd: int, label: str):
    lat: list[float] = []
    lock = threading.Lock()
    work = list(offsets)
    n = len(work)
    totals = [0]
    errors: list[str] = []

    def worker():
        try:
            handle = open_raw(path)
        except OSError as e:
            with lock:
                errors.append(str(e))
            return
        keep, addr = make_buf(size)
        local = 0
        while True:
            with lock:
                if not work:
                    break
                off = work.pop()
            t0 = time.perf_counter()
            try:
                local += pread_raw(handle, addr, off, size)
            except OSError as e:
                with lock:
                    errors.append(str(e))
                break
            dt = time.perf_counter() - t0
            with lock:
                lat.append(dt)
        k32.CloseHandle(handle)
        with lock:
            totals[0] += local

    t0 = time.perf_counter()
    if qd <= 1:
        worker()
    else:
        with ThreadPoolExecutor(max_workers=qd) as ex:
            for _ in range(qd):
                ex.submit(worker)
    wall = time.perf_counter() - t0

    mbs = totals[0] / 1e6 / wall if wall else 0.0
    lat_ms = sorted(x * 1e3 for x in lat)
    p50 = statistics.median(lat_ms) if lat_ms else 0.0
    p95 = lat_ms[min(len(lat_ms) - 1, int(0.95 * len(lat_ms)))] if lat_ms else 0.0
    print("%-24s qd=%-3d %5d reqs  %8.1f MB/s  %7.1f IOPS  p50 %7.2f ms  p95 %8.2f ms"
          % (label, qd, n, mbs, n / wall if wall else 0, p50, p95))
    if errors:
        print("     errors: %s" % errors[:2])
    return {"case": label, "qd": qd, "requests": n, "mb_per_s": mbs,
            "iops": n / wall if wall else 0, "p50_ms": p50, "p95_ms": p95,
            "mean_ms": statistics.mean(lat_ms) if lat_ms else 0.0,
            "errors": errors[:4]}


def main() -> int:
    ap = argparse.ArgumentParser(prog="disk_probe")
    ap.add_argument("--file", required=True)
    ap.add_argument("--size-kb", type=int, default=2048)
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--qd", default="1,8,32,64")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", default="")
    ap.add_argument("--region-gb", type=float, default=0.0,
                    help="limit random offsets to the first N GB (0 = whole file)")
    ap.add_argument("--reuse-offsets", action="store_true",
                    help="draw one offset list and re-read it for every case "
                         "(the old behaviour; this drive serves it ~1.5x faster)")
    ap.add_argument("--skip-sequential", action="store_true")
    args = ap.parse_args()

    size = args.size_kb * 1024
    if size % 4096:
        raise SystemExit("size must be a multiple of 4096")
    fsize = os.path.getsize(args.file)
    hi = fsize - size
    if args.region_gb:
        hi = min(hi, int(args.region_gb * 1e9) - size)
    if hi <= 0:
        raise SystemExit("file too small")

    qds = [int(x) for x in args.qd.split(",")]
    print("file      : %s" % args.file)
    print("file size : %.2f GB" % (fsize / 1e9))
    print("read size : %d KB   requests: %d   region: %s"
          % (args.size_kb, args.count,
             "%.1f GB" % args.region_gb if args.region_gb else "whole file"))
    print("cache     : FILE_FLAG_NO_BUFFERING (page cache excluded)")

    rng = random.Random(args.seed)

    def draw() -> list[int]:
        out = []
        for _ in range(args.count):
            o = rng.randrange(0, hi)
            out.append(o - (o % 4096))
        return out

    offs = draw()
    print("offsets   : %s"
          % ("one list re-read (--reuse-offsets)" if args.reuse_offsets
             else "fresh per case"))

    results = []
    if not args.skip_sequential:
        seq = [i * (1 << 30) for i in range(4)]
        results.append(run_case(args.file, seq, 64 * 1024 * 1024, 1,
                                "sequential 64MB x4"))
    for qd in qds:
        case_offs = offs if args.reuse_offsets else draw()
        results.append(run_case(args.file, case_offs, size, qd,
                                "random %dKB" % args.size_kb))
    results.append(run_case(args.file, sorted(offs if args.reuse_offsets else draw()),
                            size, max(qds), "random %dKB SORTED" % args.size_kb))

    print("\ninterpretation (134 MB = per-token miss at a ~10.6 GB tier, 86%% hit):")
    for r in results:
        if r["case"].startswith("random"):
            fetch = 0.9823e9 * 0.136
            print("  %-22s qd=%-3d  %7.1f MB/s  -> %5.1f tok/s if unhidden"
                  % (r["case"], r["qd"], r["mb_per_s"], r["mb_per_s"] * 1e6 / fetch))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"file": args.file, "size_kb": args.size_kb,
                       "count": args.count, "results": results}, f, indent=2)
        print("\nwrote %s" % args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
