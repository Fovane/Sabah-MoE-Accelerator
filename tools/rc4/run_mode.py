"""Run llama-sabah-diag in one of the three execution modes with identical settings."""
import os, sys, json, subprocess, argparse, time, hashlib
BIN = r"D:/sabah_scaling/llama-sabah-clean/build/bin/Release/llama-sabah-diag.exe"
MODEL = r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
RTLIB = r"D:/sabah_scaling/v5/sabah/runtime/cuda/sabah_rt.dll"
PROMPT = "Reply with exactly: SABAH_OK"
COMMON = ["-m", MODEL, "-c", "512", "-ngl", "99", "--cpu-moe", "-t", "8",
          "-b", "512", "-ub", "512", "--temp", "0", "--seed", "1"]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["cpu", "cuda", "sabah", "stock"])
    ap.add_argument("out")
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--names", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--steps", default="all")
    ap.add_argument("--mmid", action="store_true")
    ap.add_argument("--force", default="")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--hot-bytes", default=str(1 << 30))
    ap.add_argument("--verify-bytes", action="store_true")
    ap.add_argument("--rtlib", default=RTLIB)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    env = dict(os.environ)
    for k in list(env):
        if k.startswith(("SABAH_", "GGML_OP_OFFLOAD")):
            env.pop(k)
    env["SABAH_DIAG_OUT"] = a.out
    env["SABAH_DIAG_NAMES"] = a.names
    env["SABAH_DIAG_LAYERS"] = a.layers
    env["SABAH_DIAG_STEPS"] = a.steps
    if a.mmid: env["SABAH_DIAG_MMID"] = "1"
    if a.force: env["SABAH_DIAG_FORCE"] = a.force
    # the ONLY differences between modes are these variables
    if a.mode == "cpu":
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1000000000"
    elif a.mode == "stock":
        pass                      # llama.cpp defaults: batch>=32 offloaded, decode on CPU
    else:
        env["GGML_OP_OFFLOAD_MIN_BATCH"] = "1"
    if a.mode == "sabah":
        env.update(SABAH_LLAMA="1", SABAH_RT_LIB=a.rtlib, SABAH_LLAMA_TRACE="1",
                   SABAH_LLAMA_HOT_BYTES=a.hot_bytes)
        if a.verify_bytes: env["SABAH_LLAMA_VERIFY_BYTES"] = "1"
    cmd = [BIN] + COMMON + ["-n", str(a.n), "-p", a.prompt]
    t0 = time.time()
    with open(os.path.join(a.out, "stderr.txt"), "w", encoding="utf-8", errors="replace") as ferr:
        rc = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, stderr=ferr).returncode
    wall = time.time() - t0
    cfg = dict(mode=a.mode, rc=rc, wall_s=wall, cmd=cmd,
               env={k: v for k, v in env.items() if k.startswith(("SABAH_", "GGML_OP_OFFLOAD"))})
    metrics = None
    for line in open(os.path.join(a.out, "stderr.txt"), encoding="utf-8", errors="replace"):
        if line.startswith(("SABAH_METRICS", "SABAH_DIAG ")):
            metrics = metrics or {}
            metrics.update({kv.split("=")[0]: int(kv.split("=")[1]) for kv in line.split()[1:]})
    cfg["sabah_metrics"] = metrics
    json.dump(cfg, open(os.path.join(a.out, "mode.json"), "w"), indent=1)
    print(json.dumps(dict(mode=a.mode, rc=rc, wall_s=round(wall, 1), metrics=metrics)))
    return rc
if __name__ == "__main__":
    sys.exit(main())
