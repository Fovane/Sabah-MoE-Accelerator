"""Run llama-sabah-diag for one execution path with the frozen v1 settings.

Paths differ ONLY in these variables:
  cpu    GGML_OP_OFFLOAD_MIN_BATCH=1e9          llama.cpp CPU expert MUL_MAT_ID
  cuda   GGML_OP_OFFLOAD_MIN_BATCH=1            llama.cpp native CUDA expert MUL_MAT_ID
  sabah  as cuda + SABAH_LLAMA=1 (+ checks)     Sabah native MUL_MAT_ID
  stock  (none)                                 llama.cpp defaults (benchmark only)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
LLAMA_BIN = os.environ.get("SABAH_LLAMA_BIN", r"D:/sabah_scaling/llama-sabah-clean/build/bin/Release")
DIAG = os.path.join(LLAMA_BIN, "llama-sabah-diag.exe" if os.name == "nt" else "llama-sabah-diag")
MODEL = os.environ.get("SABAH_MODEL", "D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                                      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
RTLIB = os.path.join(V5, "sabah", "runtime", "cuda", "sabah_rt.dll" if os.name == "nt" else "libsabah_rt.so")

# frozen by docs/V1_CORRECTNESS_CONTRACT.md
COMMON = ["-c", "4096", "-b", "2048", "-ub", "512", "-ngl", "99", "--cpu-moe", "-t", "8",
          "--temp", "0", "--seed", "1"]
HOT_BYTES = 1 << 30
SELFCHECK_K = 2


def run(path, out, manifest, n_predict, names="", layers="", steps="all", mmid=False, rows="all",
        force_dir="", multiseq=False, rtlib=RTLIB, hot_bytes=HOT_BYTES, selfcheck=SELFCHECK_K,
        verify=True, extra_env=None):
    os.makedirs(out, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SABAH_", "GGML_OP_OFFLOAD"))}
    env.update(SABAH_DIAG_OUT=out, SABAH_DIAG_MANIFEST=manifest, SABAH_DIAG_NAMES=names,
               SABAH_DIAG_LAYERS=layers, SABAH_DIAG_STEPS=steps, SABAH_DIAG_ROWS=rows)
    if mmid:
        env["SABAH_DIAG_MMID"] = "1"
    if force_dir:
        env["SABAH_DIAG_FORCE_DIR"] = force_dir
    if multiseq:
        env["SABAH_DIAG_MULTISEQ"] = "1"
    if path == "cpu":
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1000000000"
    elif path in ("cuda", "sabah"):
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    elif path != "stock":
        raise ValueError(path)
    status = os.path.join(out, "sabah_status.json")
    if path == "sabah":
        env.update(SABAH_LLAMA="1", SABAH_RT_LIB=rtlib, SABAH_LLAMA_TRACE="1",
                   SABAH_LLAMA_HOT_BYTES=str(hot_bytes), SABAH_LLAMA_SELFCHECK=str(selfcheck),
                   SABAH_LLAMA_STATUS_FILE=status)
        if verify:
            env["SABAH_LLAMA_VERIFY_BYTES"] = "fetch"  # every fetch + every 64th hit
    if extra_env:
        env.update(extra_env)
    cmd = [DIAG, "-m", MODEL] + COMMON + ["-n", str(n_predict)]
    t0 = time.time()
    with open(os.path.join(out, "stderr.txt"), "w", encoding="utf-8", errors="replace") as ferr:
        rc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=ferr).returncode
    wall = time.time() - t0
    snap = None
    for line in open(os.path.join(out, "stderr.txt"), encoding="utf-8", errors="replace"):
        if line.startswith("SABAH_SNAPSHOT "):
            snap = json.loads(line[len("SABAH_SNAPSHOT "):])
    rec = dict(path=path, rc=rc, wall_s=wall, cmd=cmd, manifest=manifest, multiseq=multiseq,
               n_predict=n_predict, names=names, steps=steps, rows=rows, mmid=mmid,
               force_dir=force_dir, rtlib=rtlib if path == "sabah" else None,
               env={k: v for k, v in env.items() if k.startswith(("SABAH_", "GGML_OP_OFFLOAD"))},
               snapshot_at_exit=snap)
    json.dump(rec, open(os.path.join(out, "path.json"), "w"), indent=1)
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", choices=["cpu", "cuda", "sabah", "stock"])
    ap.add_argument("out")
    ap.add_argument("manifest")
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--names", default="")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--mmid", action="store_true")
    ap.add_argument("--rows", default="all")
    ap.add_argument("--force-dir", default="")
    ap.add_argument("--multiseq", action="store_true")
    ap.add_argument("--rtlib", default=RTLIB)
    ap.add_argument("--hot-bytes", type=int, default=HOT_BYTES)
    ap.add_argument("--selfcheck", type=int, default=SELFCHECK_K)
    a = ap.parse_args()
    r = run(a.path, a.out, a.manifest, a.n, a.names, "", a.steps, a.mmid, a.rows, a.force_dir,
            a.multiseq, a.rtlib, a.hot_bytes, a.selfcheck)
    print(json.dumps(dict(path=r["path"], rc=r["rc"], wall_s=round(r["wall_s"], 1),
                          snapshot=r["snapshot_at_exit"])))
    sys.exit(r["rc"])
