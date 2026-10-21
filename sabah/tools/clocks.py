"""GPU clock hygiene for benchmark records.

A measurement is only reproducible if the card ran at boost. This module gives
every benchmark the same three things:

* ``ClockSampler`` - a 0.5 s-resolution nvidia-smi stream covering SM clock,
  memory clock, P-state, power, temperature, utilisation, display state and the
  throttle-reason bitmask;
* ``ClockSampler.record()`` - whole-run plus steady-window statistics, and a
  ``valid`` flag that is false when the steady-window median SM clock is below
  ``MIN_STEADY_SM_MHZ`` (2400 MHz);
* ``pin()`` / ``unpin()`` - the reversible ``nvidia-smi -lgc``/``-lmc``
  pinning that makes a run reproducible, and ``machine_state()`` - the
  machine-level settings that decide whether the card is allowed to downclock
  (Windows power plan, HAGS, NVIDIA power-management registry values, display
  state).

The driver is never modified. Pinning needs an elevated shell; ``pin`` reports
the nvidia-smi message when it is refused instead of raising.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time

MIN_STEADY_SM_MHZ = 2400.0
MIN_STEADY_MEM_MHZ = 10000.0

QUERY = ("clocks.sm,clocks.mem,pstate,power.draw,power.limit,temperature.gpu,"
         "utilization.gpu,fan.speed,display_active,display_attached,"
         "clocks_throttle_reasons.active")

_NUM = {
    "sm_mhz": "clocks.sm",
    "mem_mhz": "clocks.mem",
    "power_w": "power.draw",
    "power_limit_w": "power.limit",
    "temp_c": "temperature.gpu",
    "util_pct": "utilization.gpu",
    "fan_pct": "fan.speed",
}


def _num(text):
    try:
        v = float(text)
    except (TypeError, ValueError):
        return None
    return v


def parse_line(line):
    """One nvidia-smi csv row (``noheader,nounits``) to a dict."""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 11:
        return None
    rec = {
        "sm_mhz": _num(parts[0]),
        "mem_mhz": _num(parts[1]),
        "pstate": parts[2] or None,
        "power_w": _num(parts[3]),
        "power_limit_w": _num(parts[4]),
        "temp_c": _num(parts[5]),
        "util_pct": _num(parts[6]),
        "fan_pct": _num(parts[7]),
        "display_active": parts[8] or None,
        "display_attached": parts[9] or None,
        "throttle": parts[10] or None,
    }
    if rec["sm_mhz"] is None:
        return None
    return rec


def sample_once():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        rec = parse_line(line)
        if rec is not None:
            return rec
    return None


def _agg(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2.0
    return {"n": n, "min": vals[0], "median": med, "max": vals[-1],
            "mean": round(sum(vals) / n, 3)}


def summarise(samples):
    """Statistics over ``samples`` (list of ``(t, dict)``)."""
    if not samples:
        return None
    rows = [s for _, s in samples]
    out = {}
    for key in ("sm_mhz", "mem_mhz", "power_w", "temp_c", "util_pct"):
        a = _agg([r.get(key) for r in rows])
        if a:
            out[key] = a
    pstates = {}
    for r in rows:
        if r.get("pstate"):
            pstates[r["pstate"]] = pstates.get(r["pstate"], 0) + 1
    out["pstates"] = dict(sorted(pstates.items()))
    out["dominant_pstate"] = (max(pstates, key=pstates.get) if pstates else None)
    out["throttle"] = sorted({r["throttle"] for r in rows if r.get("throttle")})
    out["display_active"] = sorted({r["display_active"] for r in rows
                                    if r.get("display_active")})
    out["samples"] = len(rows)
    out["span_s"] = round(samples[-1][0] - samples[0][0], 2)
    return out


def verdict(steady):
    """Validity of a steady-window summary."""
    if not steady or not steady.get("sm_mhz"):
        return {"valid": None, "reason": "no clock samples"}
    sm = steady["sm_mhz"]
    mem = (steady.get("mem_mhz") or {}).get("median")
    reasons = []
    if sm["median"] < MIN_STEADY_SM_MHZ:
        reasons.append(f"steady SM median {sm['median']:.0f} MHz < "
                       f"{MIN_STEADY_SM_MHZ:.0f} MHz (min {sm['min']:.0f}, "
                       f"max {sm['max']:.0f}, pstate {steady.get('dominant_pstate')})")
    if mem is not None and mem < MIN_STEADY_MEM_MHZ:
        reasons.append(f"steady memory median {mem:.0f} MHz < "
                       f"{MIN_STEADY_MEM_MHZ:.0f} MHz (the card dropped to a "
                       f"low P-state even if SM was pinned)")
    return {
        "valid": not reasons,
        "reason": "; ".join(reasons) if reasons else None,
        "min_steady_sm_mhz": MIN_STEADY_SM_MHZ,
        "min_steady_mem_mhz": MIN_STEADY_MEM_MHZ,
    }


class ClockSampler(threading.Thread):
    """Background nvidia-smi clock stream.

    ``interval`` is the sample period in seconds. Samples are kept with a
    monotonic timestamp so any window (whole run, steady decode, a specific
    episode) can be summarised afterwards.
    """

    def __init__(self, interval=0.5):
        super().__init__(daemon=True, name="clock-sampler")
        self.interval = float(interval)
        self.samples = []
        self._stop_evt = threading.Event()
        self._proc = None
        self.started_at = None

    def run(self):
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", f"--query-gpu={QUERY}",
                 "--format=csv,noheader,nounits",
                 "-lms", str(max(50, int(self.interval * 1000)))],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, bufsize=1)
        except OSError:
            return
        self.started_at = time.monotonic()
        for line in self._proc.stdout:
            if self._stop_evt.is_set():
                break
            rec = parse_line(line)
            if rec is not None:
                self.samples.append((time.monotonic(), rec))

    def stop(self):
        self._stop_evt.set()
        p = self._proc
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
        self.join(timeout=5)
        return self

    def window(self, t0, t1=None):
        t1 = time.monotonic() if t1 is None else t1
        return [(t, s) for (t, s) in self.samples if t0 <= t <= t1]

    def record(self, decode_ms=None, end=None):
        """Whole-run and steady-window record.

        ``decode_ms`` is the summed per-token time the driver printed. The
        steady window is the tail of the run that the decodes occupy, plus a
        2 s margin for the sampler and the driver's final flush.
        """
        end = time.monotonic() if end is None else end
        whole = summarise(self.samples)
        rec = {"whole": whole, "query": QUERY,
               "interval_s": self.interval,
               "samples": [[round(t - (self.started_at or t), 3), s]
                           for t, s in self.samples]}
        steady = None
        if decode_ms:
            span = (float(decode_ms) + 2000.0) / 1000.0
            steady = summarise(self.window(end - span, end))
        if steady is None:
            # No decode information: the last third of the run is the best
            # available proxy for steady state.
            if self.samples:
                t0 = self.samples[0][0]
                steady = summarise(self.window(end - (end - t0) / 3.0, end))
        rec["steady"] = steady
        rec.update(verdict(steady))
        return rec


def pin(sm_min, sm_max, mem_min=None, mem_max=None):
    """Set application clock floors/ceilings with nvidia-smi (needs admin)."""
    out = []
    ok = True
    try:
        r = subprocess.run(["nvidia-smi", "-lgc", f"{int(sm_min)},{int(sm_max)}"],
                           capture_output=True, text=True, timeout=30)
        ok = ok and r.returncode == 0
        out.append((r.stdout + r.stderr).strip())
        if mem_min is not None and mem_max is not None:
            r = subprocess.run(["nvidia-smi", "-lmc",
                                f"{int(mem_min)},{int(mem_max)}"],
                               capture_output=True, text=True, timeout=30)
            ok = ok and r.returncode == 0
            out.append((r.stdout + r.stderr).strip())
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "output": str(e)}
    return {"ok": ok, "output": "\n".join(x for x in out if x)}


def unpin():
    """Restore default clock management (needs admin)."""
    out = []
    ok = True
    try:
        for flag in ("-rgc", "-rmc"):
            r = subprocess.run(["nvidia-smi", flag], capture_output=True,
                               text=True, timeout=30)
            ok = ok and r.returncode == 0
            out.append((r.stdout + r.stderr).strip())
    except (OSError, subprocess.SubprocessError) as e:
        return {"ok": False, "output": str(e)}
    return {"ok": ok, "output": "\n".join(x for x in out if x)}


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _reg(path, value):
    out = _run(["reg", "query", path, "/v", value])
    if not out or "REG_" not in out:
        return None
    return out


def machine_state():
    """Settings that decide whether the card may downclock or lose boost."""
    plan = _run(["powercfg", "/getactivescheme"]) or ""
    plan = plan.replace("\r", "").replace("\n", " ").strip()
    graphics = r"HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers"
    cls = (r"HKLM\SYSTEM\CurrentControlSet\Control\Class"
           r"\{4d36e968-e325-11ce-bfc1-08002be10318}\0000")
    nv = {}
    for value in ("PowerMizerEnable", "PowerMizerLevel", "PowerMizerLevelAC",
                  "PerfLevelSrc", "DisableDynamicPstate", "PreferMaximumPerformance"):
        raw = _reg(cls, value)
        if raw:
            nv[value] = raw.split()[-1]
    out = {
        "power_plan": plan or None,
        "power_plan_guid": None,
        "hags_HwSchMode": None,
        "nvidia_registry": nv,
        "nvidia_smi": None,
    }
    if "GUID:" in plan:
        try:
            out["power_plan_guid"] = plan.split("GUID:")[1].split()[0]
        except IndexError:
            pass
    raw = _reg(graphics, "HwSchMode")
    out["hags_HwSchMode"] = raw.split()[-1] if raw else "absent (HAGS off)"
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version,power.limit,"
             "power.default_limit,power.max_limit,display_active,display_attached",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        out["nvidia_smi"] = r.stdout.strip().splitlines()[0] if r.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


def other_gpu_users(exclude_pid=None):
    """PIDs of processes that currently hold a context on the GPU.

    A measurement taken while another process holds the card reports that
    process's behaviour, not the model's, so ``sabah bench`` refuses instead of
    printing a number. An empty or unavailable listing is treated as "none".
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        pid = int(line)
        if exclude_pid is not None and pid == exclude_pid:
            continue
        out.append(pid)
    return out


def clock_series_csv(rec, path):
    """Write the raw samples of a ``record()`` dict as CSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("t_s,sm_mhz,mem_mhz,pstate,power_w,temp_c,util_pct,throttle\n")
        for t, s in rec.get("samples", []):
            f.write(f"{t},{s.get('sm_mhz')},{s.get('mem_mhz')},{s.get('pstate')},"
                    f"{s.get('power_w')},{s.get('temp_c')},{s.get('util_pct')},"
                    f"{s.get('throttle')}\n")


def format_steady(rec):
    s = rec.get("steady") or {}
    sm = s.get("sm_mhz") or {}
    mem = s.get("mem_mhz") or {}
    pw = s.get("power_w") or {}
    flag = "VALID" if rec.get("valid") else "INVALID"
    if not sm:
        return f"clocks: {flag} (no samples)"
    return (f"clocks: {flag} steady SM {sm['min']:.0f}/{sm['median']:.0f}/"
            f"{sm['max']:.0f} MHz, mem {mem.get('median', 0):.0f} MHz, "
            f"{s.get('dominant_pstate')}, {pw.get('median', 0):.1f} W, "
            f"{s.get('throttle')}")


if __name__ == "__main__":
    print(json.dumps({"now": sample_once(), "machine": machine_state()}, indent=2))
