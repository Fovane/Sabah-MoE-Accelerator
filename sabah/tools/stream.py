"""
Sabah Accelerator - the shipped streaming path, driven by one command.

Everything the bridge needs is an environment variable, and every number this
module reports is parsed out of the driver's own output, out of the runtime's own
counters, or out of the per-step times the driver writes next to them. Nothing
here projects, estimates or simulates: the planner owns projections, and they are
labelled as such there.

    sabah run    run the streaming decode once, per regime
    sabah matrix run a few configurations and write results/matrix.json
    sabah gate   run the correctness switches as an automated pass/fail gate

A generation has three regimes and `run` reports all three, because averaging
them is how this project mis-quoted its own speed for a while:

* **fill**  - step 0, the prompt, whose cost is the tier's cold fill (12.68 GB
              at ~1.1 GB/s on the reference rig), not prompt processing.
* **warm-up** - the next ~128 decodes, while the tier is still discovering its
              working set (~70 MB/token of disk there).
* **steady state** - after that the working set is resident (3 MB/token) and the
              token is the no-disk floor, ~67 ms on the reference rig.

The driver (`llama-sabah-diag.exe`) is patch 0002 of the llama.cpp integration;
its path is `--diag` or `SABAH_DIAG_EXE`. The runtime is `--rt` or
`SABAH_RT_LIB`, defaulting to the library built in this tree.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

DEFAULT_RT = os.path.join(ROOT, "sabah", "runtime", "cuda", "sabah_rt.dll")

# The placement requirements (see patches/llama.cpp/README.md). Without both of
# these the expert ops never reach the bridge.
BASE_ENV = {
    "SABAH_LLAMA": "1",
    "GGML_OP_OFFLOAD_MIN_BATCH": "1",
    "SABAH_LLAMA_DIRECT_IO": "1",
    "SABAH_LLAMA_IO_DEPTH": "8",
    "SABAH_LLAMA_TRACE": "1",
}

# The reproduction flags from the README, and nothing else.
BASE_FLAGS = [
    "-dev", "CUDA0", "-ngl", "99", "--cpu-moe",
    "-ot", r"\.per_layer_token_embd\.weight=CPU",
]

# What the driver and the runtime write into the run directory. They are wiped
# before every run: a file left by an earlier configuration - or by a run that
# died before finishing - must never be read back as this run's numbers.
RUN_OUTPUTS = ("step_ms.f64", "prompt_tokens.i32", "fed.i32", "argmax.i32",
               "logits.f32", "prompt.txt", "run.json", "sabah_status_live.json")


def clean_run_dir(out_dir):
    """Remove the transient outputs of a previous run from `out_dir`."""
    for name in RUN_OUTPUTS:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


class RunError(RuntimeError):
    pass


def sha256(path, limit=None):
    h = hashlib.sha256()
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
            n += len(b)
            if limit and n >= limit:
                break
    return h.hexdigest()


def _num(text, pattern, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def parse_driver(text):
    """Everything the driver and the runtime printed, as plain numbers."""
    out = {}
    # Step 0 is the prefill: it must not be counted or timed as a decoded token.
    steps = [float(m.group(2)) for m in
             re.finditer(r"prompt step (\d+) .*?([\d.]+) ms", text)
             if int(m.group(1)) >= 1]
    out["n_decodes"] = len(steps)
    if steps:
        srt = sorted(steps)
        n = len(srt)
        med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2.0
        out["decode_ms_median"] = med
        out["decode_ms_sum"] = sum(steps)
        out["decode_ms_per_token"] = sum(steps) / n
        out["decode_tok_per_s"] = 1000.0 / out["decode_ms_per_token"]

    def grab(label, pattern, cast=float):
        v = _num(text, pattern, cast)
        if v is not None:
            out[label] = v

    grab("calls", r"SABAH_DIAG calls=(\d+)", int)
    grab("tokens", r"SABAH_DIAG calls=\d+ tokens=(\d+)", int)
    grab("lookups", r"SABAH_DIAG calls=\d+ tokens=\d+ lookups=(\d+)", int)
    grab("overflow", r"overflow=(\d+)", int)
    grab("verify_ok", r"verify_ok=(\d+)", int)
    grab("verify_fail", r"verify_fail=(\d+)", int)
    grab("unbuffered_reads", r"SABAH_DISK unbuffered_reads=(\d+)", int)
    grab("read_bytes", r"SABAH_DISK unbuffered_reads=\d+ bytes=(\d+)", int)
    grab("waves", r"waves=(\d+)", int)
    grab("mapped_fallbacks", r"mapped_fallbacks=(\d+)", int)
    grab("staging_bytes", r"staging_bytes=(\d+)", int)
    grab("io_overlap", r"SABAH_IO read_ms=[\d.]+ wave_ms=[\d.]+ overlap=([\d.]+)")
    grab("phase_ids_ms", r"SABAH_PHASE calls=\d+ ids=([\d.]+)ms")
    grab("phase_fetch_ms", r"ids=[\d.]+ms fetch=([\d.]+)ms")
    grab("phase_wait_ms", r"fetch=[\d.]+ms wait=([\d.]+)ms")
    grab("phase_slot_ms", r"wait=[\d.]+ms slot=([\d.]+)ms")
    grab("phase_ptr_ms", r"slot=[\d.]+ms ptr=([\d.]+)ms")
    grab("phase_prime_ms", r"ptr=[\d.]+ms prime=([\d.]+)ms")
    grab("id_reuse", r"SABAH_IDS reused=(\d+)", int)
    grab("id_group_max", r"SABAH_IDS reused=\d+ group_max=(\d+)", int)
    grab("id_check", r"SABAH_IDS reused=\d+ group_max=\d+ check=(\d+)", int)
    grab("id_check_ok", r"check=\d+ check_ok=(\d+)", int)
    grab("id_check_fail", r"check_ok=\d+ check_fail=(\d+)", int)
    # The exact cache counters come from the runtime's exit snapshot, not from
    # SABAH_LLAMA_STATUS_FILE: the status file is rewritten after every expert
    # call, which costs 144 synchronous writes per token and poisons any timing
    # taken in the same pass (~2 ms per call here, ~0.3 s per token).
    i = text.find("SABAH_SNAPSHOT ")
    if i >= 0:
        j = text.find("\n", i)
        try:
            snap = json.loads(text[i + len("SABAH_SNAPSHOT "):j if j >= 0 else len(text)])
        except ValueError:
            snap = None
        if isinstance(snap, dict):
            out["snapshot"] = snap
            for k in ("calls", "tokens", "lookups", "hits", "misses", "evictions",
                      "overflow", "resident_bytes", "bytes_fetched",
                      "verify_ok", "verify_fail", "empty_calls"):
                if k in snap:
                    out[k] = snap[k]
    phases = [out.get(k) for k in ("phase_ids_ms", "phase_fetch_ms", "phase_slot_ms",
                                   "phase_ptr_ms", "phase_prime_ms")]
    if all(v is not None for v in phases) and out.get("n_decodes"):
        out["phase_sum_ms_per_call"] = sum(phases) / max(out["calls"] or 1, 1)
    return out


def status(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            t = f.read().strip()
        return json.loads(t) if t and t != "null" else {}
    except (ValueError, OSError):
        return {}


def step_ms(out_dir):
    """The driver's per-step wall times. Index 0 is the prompt (the fill)."""
    path = os.path.join(out_dir, "step_ms.f64")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 8
    return list(struct.unpack("<%dd" % n, raw[: n * 8]))


def prompt_tokens(out_dir):
    """The ids the driver actually fed. A templated prompt is not a short one:
    `-p "Hello"` is 53 tokens, and `tokens=` in the trace line is not this."""
    path = os.path.join(out_dir, "prompt_tokens.i32")
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        raw = f.read()
    n = len(raw) // 4
    return list(struct.unpack("<%di" % n, raw[: n * 4]))


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def regimes(out_dir, warm_tokens=128, fill_ms=None):
    """Split one run into fill / warm-up / steady state.

    The split is not cosmetic. `-n 32` minus `-n 8` differencing (the decode
    budget method) lands inside the warm-up, so it prices the marginal token of
    a generation that has just started - 190.9 ms/token on the reference rig
    against a 67 ms steady state. Quoting either one alone is a mis-measurement.
    """
    ms = step_ms(out_dir)
    if len(ms) < 2:
        return {}
    fill = ms[0] if fill_ms is None else fill_ms
    dec = ms[1:]
    warm = dec[:warm_tokens]
    steady = dec[warm_tokens:]
    out = {
        "fill_ms": fill,
        "decodes": len(dec),
        "warmup": {"n": len(warm), "median_ms": _median(warm),
                   "mean_ms": (sum(warm) / len(warm)) if warm else None},
    }
    if steady:
        m = _median(steady)
        out["steady"] = {"n": len(steady), "median_ms": m,
                         "mean_ms": sum(steady) / len(steady),
                         "tok_per_s": (1000.0 / m) if m else None}
    total = sum(dec)
    if total > 0:
        out["decode_only_tok_per_s"] = 1000.0 * len(dec) / total
        out["end_to_end_tok_per_s"] = 1000.0 * len(dec) / (total + fill)
    return out


def run(model, prompt="Hello", tokens=32, context=512, threads=8,
        hot_bytes="9.8e9", diag="", rt="", extra_env=None, extra_flags=None,
        out_dir=None, timeout=1800, quiet=False, counters=False):
    """Run the streaming decode once. Returns the measured record.

    The cache counters (hits, misses, evictions, resident bytes) come from the
    runtime's exit snapshot, which is exact, so nothing here needs the status
    file. `counters=True` still sets `SABAH_LLAMA_STATUS_FILE` - the live-server
    switch - and the runtime rewrites it after *every* expert call: 144
    synchronous file writes per token here, worth ~0.3 s per token. Never time a
    run with it set.
    """
    diag = diag or os.environ.get("SABAH_DIAG_EXE", "")
    if not diag:
        raise RunError("no driver: pass --diag <llama-sabah-diag.exe> or set SABAH_DIAG_EXE\n"
                       "(build it from patches/llama.cpp/0002-sabah-diag-capture-tool.patch)")
    if not os.path.exists(diag):
        raise RunError("driver not found: %s" % diag)
    rt = rt or os.environ.get("SABAH_RT_LIB", "") or DEFAULT_RT
    if not os.path.exists(rt):
        raise RunError("runtime not found: %s (build it, or pass --rt)" % rt)
    if not os.path.exists(model):
        raise RunError("model not found: %s" % model)

    out_dir = out_dir or os.path.join(ROOT, "results", "_run")
    os.makedirs(out_dir, exist_ok=True)
    status_path = os.path.join(out_dir, "sabah_status_live.json")

    env = dict(os.environ)
    env.update(BASE_ENV)
    env["SABAH_RT_LIB"] = os.path.abspath(rt)
    env["SABAH_LLAMA_HOT_BYTES"] = str(hot_bytes)
    if counters:
        env["SABAH_LLAMA_STATUS_FILE"] = status_path
    else:
        env.pop("SABAH_LLAMA_STATUS_FILE", None)
    env["SABAH_DIAG_OUT"] = out_dir
    env["SABAH_LLAMA_PREFETCH"] = "0"
    env.pop("SABAH_LLAMA_LAYER_MERGE", None)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    argv = [os.path.abspath(diag), "-m", os.path.abspath(model)]
    argv += BASE_FLAGS
    argv += ["-t", str(threads), "-c", str(context), "-n", str(tokens), "-p", prompt]
    if extra_flags:
        argv += [str(f) for f in extra_flags]

    # A status file or a step stream left by an earlier run in this directory must
    # never be read back as this run's output.
    clean_run_dir(out_dir)

    t0 = time.time()
    proc = subprocess.run(argv, env=env, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)
    wall = time.time() - t0
    rec = parse_driver(proc.stdout + proc.stderr)
    rec["wall_s"] = wall
    rec["exit_code"] = proc.returncode
    st = status(status_path) if counters else {}
    if st:
        rec["resident_bytes"] = st.get("resident_bytes")
        rec["misses"] = st.get("misses")
        rec["hits"] = st.get("hits")
        rec["evictions"] = st.get("evictions")
        rec["cached_bytes_fetched"] = st.get("bytes_fetched")
        rec["status_verify_ok"] = st.get("verify_ok")
        rec["status_verify_fail"] = st.get("verify_fail")
    rec["hot_bytes"] = str(hot_bytes)
    rec["model"] = os.path.abspath(model)
    rec["tokens_requested"] = tokens
    rec["prompt"] = prompt
    rec["driver"] = os.path.abspath(diag)
    rec["runtime"] = os.path.abspath(rt)
    rec["counters_enabled"] = bool(counters)
    pt = prompt_tokens(out_dir)
    if pt:
        rec["prompt_tokens_count"] = len(pt)
        rec["prompt_tokens"] = pt
    # The driver writes the per-step times next to its other output; they are the
    # only way to see that a run reached the steady state at all.
    reg = regimes(out_dir, fill_ms=rec.get("prefill_ms"))
    if reg:
        rec["regimes"] = reg
        rec["fill_ms"] = reg.get("fill_ms")
        for k in ("decode_only_tok_per_s", "end_to_end_tok_per_s"):
            if reg.get(k) is not None:
                rec[k] = reg[k]
        if reg.get("steady"):
            rec["steady_ms_median"] = reg["steady"]["median_ms"]
            rec["steady_tok_per_s"] = reg["steady"]["tok_per_s"]
    if rec.get("read_bytes") and rec.get("n_decodes"):
        rec["read_bytes_per_decoded_token"] = rec["read_bytes"] / rec["n_decodes"]
    if not quiet:
        print_breakdown(rec)
    return rec


def print_breakdown(r):
    print("  model        %s" % (os.path.basename(r.get("model") or "") or r.get("driver", "")))
    print("  exit code    %d   wall %.1f s" % (r.get("exit_code", -1), r.get("wall_s", 0.0)))
    reg = r.get("regimes") or {}
    if reg.get("fill_ms") is not None:
        print("  fill         %.1f s before the first token (%s prompt tokens)"
              % (reg["fill_ms"] / 1000.0, r.get("prompt_tokens_count", "?")))
    if reg.get("warmup"):
        w = reg["warmup"]
        print("  warm-up      %d tokens, median %.1f ms  (the tier is still filling)"
              % (w["n"], w["median_ms"] or 0.0))
    if reg.get("steady"):
        s = reg["steady"]
        print("  steady       %d tokens, median %.1f ms = %.2f tok/s  (working set resident)"
              % (s["n"], s["median_ms"] or 0.0, s["tok_per_s"] or 0.0))
    if r.get("decode_only_tok_per_s") is not None:
        print("  overall      %.2f tok/s over the decodes, %.2f tok/s including the fill"
              % (r["decode_only_tok_per_s"], r.get("end_to_end_tok_per_s") or 0.0))
    if r.get("read_bytes") is not None:
        print("  disk         %.2f GB read, %.1f MB per decoded token (the fill included)"
              % (r["read_bytes"] / 1e9, (r.get("read_bytes_per_decoded_token") or 0) / 1e6))
    if r.get("resident_bytes") is not None:
        print("  cache        %s hits, %s misses, %s evictions, resident %.2f GB of %s"
              % (r.get("hits"), r.get("misses"), r.get("evictions"),
                 (r.get("resident_bytes") or 0) / 1e9, r.get("hot_bytes")))
    ph = ["phase_ids_ms", "phase_fetch_ms", "phase_wait_ms", "phase_slot_ms",
          "phase_ptr_ms", "phase_prime_ms"]
    if all(r.get(k) is not None for k in ph):
        print("  phases       " + "  ".join("%s=%.1fms" % (k.replace("phase_", "").replace("_ms", ""), r[k])
                                           for k in ph))
    print("  checks       verify %s/%s, mapped_fallbacks=%s, overflow=%s"
          % (r.get("verify_ok"), r.get("verify_fail"),
             r.get("mapped_fallbacks"), r.get("overflow")))


def environment_fingerprint(model, rt, diag):
    fp = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "python": sys.version.split()[0]}
    for key, path in (("runtime_sha256", rt), ("driver_sha256", diag)):
        try:
            fp[key] = sha256(path)
        except OSError:
            fp[key] = None
    try:
        st = os.stat(model)
        fp["model"] = {"path": os.path.abspath(model), "bytes": st.st_size,
                       "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime))}
    except OSError:
        fp["model"] = {"path": model}
    head = os.path.join(ROOT, ".git", "HEAD")
    fp["repo_head"] = _git_head()
    return fp


def _git_head():
    try:
        out = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


MATRIX = [
    # label, env overrides, extra flags, tokens (or None for the default)
    ("tier-9.8G", {}, [], None),
    ("tier-1G", {"SABAH_LLAMA_HOT_BYTES": "1e9"}, [], None),
    ("tier-9.8G-verify", {"SABAH_LLAMA_VERIFY_BYTES": "1",
                          "SABAH_LLAMA_VERIFY_LIMIT": "2000",
                          "SABAH_LLAMA_ID_CHECK": "1"}, [], 8),
    # The steady state only exists after the tier has learned its working set
    # (~130 decodes on the reference rig), so one row has to be long.
    ("tier-9.8G-steady", {}, [], 256),
]


def matrix(model, diag="", rt="", tokens=32, tokens_verify=8, out="",
           hot_bytes="9.8e9", prompt="Hello", threads=8, context=512):
    """Run the configurations the README quotes, and write them as one JSON."""
    recs = []
    for label, env, flags, row_tokens in MATRIX:
        n = row_tokens or (tokens_verify if "verify" in label else tokens)
        print("-- %s (n=%d)" % (label, n))
        rec = run(model, prompt=prompt, tokens=n, context=context, threads=threads,
                  hot_bytes=env.get("SABAH_LLAMA_HOT_BYTES", hot_bytes),
                  diag=diag, rt=rt, extra_env=env, extra_flags=flags)
        rec["label"] = label
        rec["model"] = model
        recs.append(rec)
        print()
    doc = {"format": "sabah.results.matrix/1",
           "environment": environment_fingerprint(model, os.path.abspath(rt or DEFAULT_RT), diag),
           "runs": recs}
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
            f.write("\n")
        print("wrote %s" % out)
    return doc


def gate_checks(rec):
    """The pass/fail checks a completed run must satisfy. Pure, so it is tested."""
    return [
        ("the bridge executed expert ops", (rec.get("calls") or 0) > 0, rec.get("calls")),
        ("every served byte matched the GGUF", rec.get("verify_ok", 0) > 0 and
         rec.get("verify_fail", 0) == 0, "%s ok / %s fail" % (rec.get("verify_ok"),
                                                              rec.get("verify_fail"))),
        ("the id reuse is byte for byte", rec.get("id_check_ok", 0) > 0 and
         rec.get("id_check_fail", 0) == 0,
         "%s ok / %s fail" % (rec.get("id_check_ok"), rec.get("id_check_fail"))),
        ("no mapped fallback", (rec.get("mapped_fallbacks") or 0) == 0,
         rec.get("mapped_fallbacks")),
        ("decode finished", rec.get("exit_code") == 0, rec.get("exit_code")),
    ]


def gate_notes(rec):
    """Counters that are reported but never gated.

    `overflow` is over-budget allocations, not bad expert ids: when every
    candidate victim is still referenced by the call in flight, the tier
    allocates past its budget rather than dangle a pointer. That is documented
    behaviour and the byte comparison in gate_checks is what proves the routed
    result stayed exact. This function exists because an earlier version of the
    gate hard-failed on the counter under the name "an expert id escaped the
    model", which was wrong about what the counter means.
    """
    return [("over-budget allocations", rec.get("overflow"))]


def gate(model, diag="", rt="", tokens=8, hot_bytes="9.8e9",
         verify_limit=2000, prompt="Hello", threads=8, context=512):
    """The correctness switches, as a pass/fail gate."""
    print("-- gate: byte-compare what the runtime serves, and prove the id reuse")
    rec = run(model, prompt=prompt, tokens=tokens, context=context, threads=threads,
              hot_bytes=hot_bytes, diag=diag, rt=rt,
              extra_env={"SABAH_LLAMA_VERIFY_BYTES": "1",
                         "SABAH_LLAMA_VERIFY_LIMIT": str(verify_limit),
                         "SABAH_LLAMA_ID_CHECK": "1"})
    checks = gate_checks(rec)
    notes = gate_notes(rec)
    print()
    ok = True
    for label, good, detail in checks:
        print("  [%s] %-36s %s" % ("ok" if good else "!!", label, detail))
        ok = ok and good
    for label, detail in notes:
        print("  [--] %-36s %s (documented; correctness is the checks above)"
              % (label, detail))
    print()
    print("gate: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1
