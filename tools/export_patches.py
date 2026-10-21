"""Export the llama.cpp changes as per-feature patches (Phase 5.1).

The pinned upstream commit is recorded in each patch header, and the file lists
are the ones the feature touched, so a rebase or audit can tell which patch
owns which hunk.

    python tools/export_patches.py
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA = os.path.join(ROOT, "third_party", "llama.cpp")
OUT = os.path.join(ROOT, "patches", "llama.cpp")

ITEM4 = [
    "ggml/src/ggml-backend.cpp",
    "ggml/src/ggml-cuda/CMakeLists.txt",
    "ggml/src/ggml-cuda/ggml-cuda.cu",
    "ggml/src/ggml-cuda/sabah-seg.cpp",
    "ggml/src/ggml-cuda/sabah-seg.h",
]

ITEMB = [
    "common/common.cpp",
    "common/common.h",
    "src/llama-context.cpp",
    "src/llama-context.h",
    "src/llama-graph.cpp",
    "src/llama-graph.h",
    "src/llama-kv-cache.cpp",
    "src/llama-kv-cache.h",
    "src/llama-memory-hybrid-idx.cpp",
    "src/llama-memory-hybrid-idx.h",
    "src/llama-memory-hybrid.cpp",
    "src/llama-memory-hybrid.h",
    "src/llama-memory-recurrent.cpp",
    "src/llama-memory-recurrent.h",
    "src/llama-memory.h",
    "src/models/qwen4exp.cpp",
    "tools/sabah-diag/sabah-diag.cpp",
]


def git(*args):
    return subprocess.run(["git", *args], cwd=LLAMA, capture_output=True, text=True)


def main():
    pinned = git("rev-parse", "HEAD").stdout.strip()
    print("pinned llama.cpp:", pinned)

    os.makedirs(OUT, exist_ok=True)
    for name, files in (("0004-sabah-cuda-graph-props-tightening.patch", ITEM4),
                        ("0005-sabah-prefill-group-experiment.patch", ITEMB)):
        p = subprocess.run(["git", "diff", "--no-color", "--", *files],
                           cwd=LLAMA, capture_output=True, text=True)
        header = ("# llama.cpp pinned commit: %s\n"
                  "# feature: %s\n"
                  "# files:\n%s\n"
                  % (pinned,
                     "segmented CUDA graphs (item 4), property check tightened" if "0004" in name
                     else "layer-group prefill experiment (item B), off by default",
                     "\n".join("  # " + f for f in files) + "\n"))
        body = p.stdout.replace("\r\n", "\n")
        out = os.path.join(OUT, name)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write(header)
            f.write(body)
        n = len(body.splitlines())
        print(f"wrote {out}  ({n} diff lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
