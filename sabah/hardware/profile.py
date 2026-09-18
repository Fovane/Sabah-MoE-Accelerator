"""
Sabah / hardware / profile

Characterises the machine and persists a versioned HardwareProfile.

Two rules this module exists to enforce:

1. **Never assume nominal capacity.** Reported VRAM and RAM are not usable
   VRAM and RAM. Everything the planner sees is a *usable* figure with an
   explicit safety reserve subtracted.

2. **Never sum per-device bandwidth.** `sum(solo)` is not aggregate. The
   simultaneous multi-GPU H2D figure is measured by the CUDA qualifier and is
   the number the planner must use. A machine whose three GPUs each reach
   11 GB/s alone but 18 GB/s together is an 18 GB/s machine.
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
import platform
import subprocess
from dataclasses import dataclass, field, asdict

HW_PROFILE_VERSION = 2

HERE = os.path.dirname(os.path.abspath(__file__))
QUALIFY_EXE = os.path.join(HERE, "mgpu_qualify.exe")

# Safety reserves. Deliberately conservative: a plan that OOMs is worse than a
# plan that leaves a gigabyte unused.
VRAM_RESERVE_FRAC = 0.06        # driver/fragmentation headroom
VRAM_RESERVE_MIN = 512 << 20    # never reserve less than this
RAM_RESERVE_FRAC = 0.12         # OS + page cache + runtime
RAM_RESERVE_MIN = 4 << 30


@dataclass
class GpuInfo:
    index: int = 0
    name: str = ""
    vram_total: int = 0
    vram_free: int = 0
    vram_usable: int = 0
    cc: str = ""
    async_engines: int = 0
    pcie_gen_max: int = 0
    pcie_width_max: int = 0
    pcie_gen_cur: int = 0
    pcie_width_cur: int = 0
    h2d_solo_gbps: float = 0.0


@dataclass
class HardwareProfile:
    version: int = HW_PROFILE_VERSION
    created: float = 0.0
    host: str = ""
    os: str = ""
    cuda_available: bool = False

    gpus: list = field(default_factory=list)
    h2d_sum_solo_gbps: float = 0.0
    h2d_simultaneous_gbps: float = 0.0
    h2d_contention_factor: float = 0.0
    topology: str = ""

    cpu_name: str = ""
    cpu_logical: int = 0
    ram_total: int = 0
    ram_usable: int = 0
    ram_bandwidth_gbps: float = 0.0

    storage_path: str = ""
    storage_read_gbps: float = 0.0

    warnings: list = field(default_factory=list)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @staticmethod
    def load(path: str) -> "HardwareProfile":
        with open(path) as f:
            d = json.load(f)
        if d.get("version") != HW_PROFILE_VERSION:
            raise ValueError("hardware profile version %s, expected %s; "
                             "run `sabah requalify`"
                             % (d.get("version"), HW_PROFILE_VERSION))
        gpus = [GpuInfo(**g) for g in d.pop("gpus", [])]
        p = HardwareProfile(**d)
        p.gpus = gpus
        return p

    @property
    def total_vram_usable(self) -> int:
        return sum(g.vram_usable for g in self.gpus)

    @property
    def effective_h2d_gbps(self) -> float:
        """The number the planner is allowed to use."""
        if self.h2d_simultaneous_gbps > 0:
            return self.h2d_simultaneous_gbps
        return self.h2d_sum_solo_gbps


# ---------------------------------------------------------------- helpers
def _run(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:
        return -1, "", str(e)


def _nvidia_smi_gpus():
    q = ("index,name,memory.total,memory.free,pcie.link.gen.max,"
         "pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current")
    rc, out, _ = _run(["nvidia-smi", "--query-gpu=" + q,
                       "--format=csv,noheader,nounits"])
    rows = []
    if rc != 0:
        return rows
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 8:
            continue
        try:
            rows.append(dict(index=int(p[0]), name=p[1],
                             vram_total=int(float(p[2])) * (1 << 20),
                             vram_free=int(float(p[3])) * (1 << 20),
                             pcie_gen_max=int(p[4]), pcie_width_max=int(p[5]),
                             pcie_gen_cur=int(p[6]), pcie_width_cur=int(p[7])))
        except ValueError:
            continue
    return rows


def _topology():
    rc, out, _ = _run(["nvidia-smi", "topo", "-m"], timeout=60)
    return out.strip() if rc == 0 else ""


def _ram_bandwidth_gbps(mb=768):
    """Coarse streaming-read estimate. Not a STREAM benchmark, but enough for
    the planner to tell a fast machine from a slow one."""
    try:
        import numpy as np
    except Exception:
        return 0.0
    n = (mb << 20) // 8
    a = np.ones(n, dtype=np.float64)
    a.sum()                                  # warm
    best = 0.0
    for _ in range(3):
        t0 = time.perf_counter()
        a.sum()
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, a.nbytes / dt / 1e9)
    return best


def _storage_read_gbps(path, mb=1024):
    """Sequential read at the model's own location.

    A split GGUF's first shard is usually a small metadata shard, so timing a
    read of it measures open() overhead rather than the device. Always pick the
    LARGEST shard in the set.
    """
    try:
        target = path
        if os.path.isdir(target):
            return 0.0
        cands = _shard_like(path)
        if cands:
            target = max(cands, key=os.path.getsize)
        size = os.path.getsize(target)
        if size < (64 << 20):
            return 0.0
        want = min(size, mb << 20)
        chunk = 8 << 20
        got = 0
        with open(target, "rb", buffering=0) as f:
            t0 = time.perf_counter()
            while got < want:
                b = f.read(min(chunk, want - got))
                if not b:
                    break
                got += len(b)
            dt = time.perf_counter() - t0
        return got / dt / 1e9 if dt > 0 else 0.0
    except Exception:
        return 0.0


def _shard_like(path):
    """All shards of a split GGUF, or just the file itself."""
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", os.path.basename(path))
    if not m:
        return [path]
    base, total = m.group(1), int(m.group(3))
    d = os.path.dirname(path)
    out = []
    for i in range(1, total + 1):
        q = os.path.join(d, "%s-%05d-of-%05d.gguf" % (base, i, total))
        if os.path.exists(q):
            out.append(q)
    return out or [path]


def _cpu_name():
    if platform.system() == "Windows":
        rc, out, _ = _run(["wmic", "cpu", "get", "name", "/value"], timeout=30)
        m = re.search(r"Name=(.+)", out or "")
        if m:
            return m.group(1).strip()
    return platform.processor() or platform.machine()


def _ram_total():
    if platform.system() == "Windows":
        # wmic is deprecated and returns nothing under some shells; prefer
        # PowerShell and keep wmic only as a fallback.
        rc, out, _ = _run(["powershell", "-NoProfile", "-Command",
                           "(Get-CimInstance Win32_ComputerSystem)."
                           "TotalPhysicalMemory"], timeout=60)
        m = re.search(r"(\d{6,})", out or "")
        if m:
            return int(m.group(1))
        rc, out, _ = _run(["wmic", "ComputerSystem", "get",
                           "TotalPhysicalMemory", "/value"], timeout=30)
        m = re.search(r"TotalPhysicalMemory=(\d+)", out or "")
        if m:
            return int(m.group(1))
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except Exception:
        return 0


def qualify(model_path: str = "", gb_per_dev: float = 1.5,
            bank_mb: int = 384, verbose: bool = True) -> HardwareProfile:
    p = HardwareProfile(created=time.time(), host=platform.node(),
                        os="%s %s" % (platform.system(), platform.release()))

    def say(*a):
        if verbose:
            print(*a)

    # ---- GPUs -----------------------------------------------------------
    say("  detecting GPUs ...")
    smi = {g["index"]: g for g in _nvidia_smi_gpus()}
    cu = {}
    if os.path.exists(QUALIFY_EXE):
        say("  measuring H2D bandwidth (solo and simultaneous) ...")
        tmp = os.path.join(HERE, "_qualify.json")
        rc, out, err = _run([QUALIFY_EXE, "--gb", str(gb_per_dev),
                             "--bank-mb", str(bank_mb), "--json-out", tmp],
                            timeout=900)
        if rc == 0 and os.path.exists(tmp):
            try:
                cu = json.load(open(tmp))
            except Exception as e:
                p.warnings.append("qualifier JSON unreadable: %s" % e)
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        else:
            p.warnings.append("H2D qualifier failed (rc=%s): %s"
                              % (rc, (err or out)[:200]))
    else:
        p.warnings.append(
            "mgpu_qualify.exe not built; H2D bandwidth unknown. Build it with "
            "nvcc -O3 -o mgpu_qualify.exe mgpu_qualify.cu")

    cu_devs = {d["index"]: d for d in cu.get("devices", [])}
    n = max(len(smi), len(cu_devs))
    for i in range(n):
        s, c = smi.get(i, {}), cu_devs.get(i, {})
        total = c.get("vram_total") or s.get("vram_total", 0)
        free = s.get("vram_free", total)
        reserve = max(int(total * VRAM_RESERVE_FRAC), VRAM_RESERVE_MIN)
        p.gpus.append(GpuInfo(
            index=i, name=c.get("name") or s.get("name", "?"),
            vram_total=total, vram_free=free,
            vram_usable=max(0, min(free, total) - reserve),
            cc=c.get("cc", ""), async_engines=int(c.get("async_engines", 0)),
            pcie_gen_max=s.get("pcie_gen_max", 0),
            pcie_width_max=s.get("pcie_width_max", 0),
            pcie_gen_cur=s.get("pcie_gen_cur", 0),
            pcie_width_cur=s.get("pcie_width_cur", 0),
            h2d_solo_gbps=float(c.get("h2d_solo_gbps", 0.0))))
    p.cuda_available = bool(p.gpus)
    p.h2d_sum_solo_gbps = float(cu.get("h2d_sum_solo_gbps", 0.0))
    p.h2d_simultaneous_gbps = float(cu.get("h2d_simultaneous_gbps", 0.0))
    p.h2d_contention_factor = float(cu.get("h2d_contention_factor", 0.0))
    p.topology = _topology()

    if len(p.gpus) > 1 and p.h2d_contention_factor and p.h2d_contention_factor < 0.8:
        p.warnings.append(
            "PCIe root complex does not scale: aggregate %.2f GB/s vs %.2f GB/s "
            "summed. The planner will size against the aggregate."
            % (p.h2d_simultaneous_gbps, p.h2d_sum_solo_gbps))
    for g in p.gpus:
        if g.pcie_gen_cur and g.pcie_gen_max and g.pcie_gen_cur < g.pcie_gen_max:
            p.warnings.append(
                "GPU%d reports PCIe gen%d of gen%d, but link state is read at "
                "idle and ramps under load; trust the measured H2D figure "
                "(%.2f GB/s) over this." % (g.index, g.pcie_gen_cur,
                                            g.pcie_gen_max, g.h2d_solo_gbps))
        if g.pcie_width_max and g.pcie_width_cur and g.pcie_width_cur < g.pcie_width_max:
            p.warnings.append(
                "GPU%d negotiated x%d of x%d — check the slot; expert streaming "
                "is bandwidth-bound." % (g.index, g.pcie_width_cur, g.pcie_width_max))

    # ---- CPU / RAM ------------------------------------------------------
    say("  probing CPU and RAM ...")
    p.cpu_name = _cpu_name()
    p.cpu_logical = os.cpu_count() or 0
    p.ram_total = _ram_total()
    reserve = max(int(p.ram_total * RAM_RESERVE_FRAC), RAM_RESERVE_MIN)
    p.ram_usable = max(0, p.ram_total - reserve)
    p.ram_bandwidth_gbps = _ram_bandwidth_gbps()

    # ---- storage --------------------------------------------------------
    if model_path and os.path.exists(model_path):
        say("  measuring storage read at the model path ...")
        p.storage_path = model_path
        p.storage_read_gbps = _storage_read_gbps(model_path)

    return p


def summarize(p: HardwareProfile) -> str:
    L, a = [], lambda s: L.append(s)
    a("host       : %s  (%s)" % (p.host, p.os))
    a("cpu        : %s, %d logical cores" % (p.cpu_name, p.cpu_logical))
    a("ram        : %.1f GB total, %.1f GB usable, ~%.0f GB/s read"
      % (p.ram_total / 1e9, p.ram_usable / 1e9, p.ram_bandwidth_gbps))
    if p.storage_read_gbps:
        a("storage    : %.2f GB/s sequential at the model path" % p.storage_read_gbps)
    if not p.gpus:
        a("gpus       : none detected")
    else:
        a("gpus       : %d" % len(p.gpus))
        for g in p.gpus:
            link = ("gen%d x%d" % (g.pcie_gen_cur, g.pcie_width_cur)
                    if g.pcie_gen_cur else "link ?")
            a("  [%d] %-34s %5.2f GB total / %5.2f usable  %-10s H2D %.2f GB/s"
              % (g.index, g.name, g.vram_total / 1e9, g.vram_usable / 1e9,
                 link, g.h2d_solo_gbps))
        a("h2d        : sum(solo) %.2f GB/s | SIMULTANEOUS %.2f GB/s | contention %.2f"
          % (p.h2d_sum_solo_gbps, p.h2d_simultaneous_gbps, p.h2d_contention_factor))
        a("             the planner uses the SIMULTANEOUS figure: %.2f GB/s"
          % p.effective_h2d_gbps)
    for w in p.warnings:
        a("warning    : %s" % w)
    return "\n".join(L)


if __name__ == "__main__":
    mp = sys.argv[1] if len(sys.argv) > 1 else ""
    prof = qualify(mp)
    print(summarize(prof))
