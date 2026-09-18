"""Record the exact environment of a v1 validation attempt."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from run_path import DIAG, LLAMA_BIN, MODEL, RTLIB, COMMON, HOT_BYTES, SELFCHECK_K  # noqa: E402


def sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def sh(cmd, cwd=None):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout.strip()
    except OSError as e:
        return str(e)


def main(out, contract_commit):
    sent = os.environ.get("SABAH_SENTINEL_DIR", "D:/sabah_rc4/sentinels")
    env = dict(
        V1_CONTRACT_COMMIT=contract_commit,
        implementation_commit=sh(["git", "rev-parse", "HEAD"], V5),
        worktree_clean=sh(["git", "status", "--porcelain"], V5) == "",
        llama_cpp_base="96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b",
        patches={p: sha(os.path.join(V5, "patches", "llama.cpp", p)) for p in
                 ("0001-sabah-native-mul-mat-id.patch", "0002-sabah-diag-capture-tool.patch")},
        binaries={os.path.basename(p): sha(p) for p in
                  [RTLIB, DIAG] + [os.path.join(LLAMA_BIN, f) for f in
                                   ("llama-server.exe", "ggml-cuda.dll", "ggml-base.dll", "llama.dll", "ggml-cpu.dll")]
                  if os.path.exists(p)},
        sentinel_binaries={f: sha(os.path.join(sent, f)) for f in sorted(os.listdir(sent)) if f.endswith(".dll")},
        model=dict(path=MODEL, sha256_shard1="4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082",
                   sha256_source="RC2 record; see results/full_model_first_token.json"),
        corpus={f: sha(os.path.join(V5, "tests", "v1_validation", f))
                for f in ("prompts.jsonl", "ubatch.jsonl", "manifest.json")},
        gpu=sh(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total", "--format=csv,noheader"]),
        gpu_processes_before=sh(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"]),
        cuda=sh(["nvcc", "--version"]).splitlines()[-1:],
        os=platform.platform(), python=platform.python_version(),
        settings=dict(common_args=COMMON, hot_bytes=HOT_BYTES, selfcheck_k=SELFCHECK_K, verify_bytes="fetch"),
    )
    json.dump(env, open(out, "w"), indent=1)
    print(json.dumps({k: env[k] for k in ("V1_CONTRACT_COMMIT", "implementation_commit", "worktree_clean")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
