"""One source of truth for where the model lives.

Every tool that needs the model path goes through `model_dir()` /
`model_shard()`, so nothing in this repository carries a drive-letter
assumption. The single setting is the environment variable:

    SABAH_MODEL_DIR = C:\\Strata-data\\models\\IQ3_S

`sabah/tools/stream.py` exports it for every launcher run and
`sabah/tools/cli.py` prints it, so `sabah run`, `sabah gate` and `sabah bench`
share one value with `tools/phase0_measure.py`.

`DEFAULT_MODEL_DIR` exists only so the tools still run when the variable is
unset on a fresh checkout; it is *documented* rather than assumed, and a run
that uses it says so in the log.
"""
from __future__ import annotations

import os

#: the directory that holds both IQ3_S shards, as a fallback when the
#: environment does not name one
DEFAULT_MODEL_DIR = os.path.join("C:" + os.sep, "Strata-data", "models", "IQ3_S")

#: the two shard file names, in load order
SHARDS = (
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
)

#: what the driver is given: shard 1. llama.cpp resolves shard 2 from the
#: "-00001-of-00002" suffix, so the second file only has to be present.
DRIVER_SHARD = SHARDS[0]


def model_dir() -> str:
    """The configured model directory.

    Precedence: SABAH_MODEL_DIR, then DEFAULT_MODEL_DIR. The returned path is
    always absolute and uses the native separator.
    """
    raw = os.environ.get("SABAH_MODEL_DIR")
    if raw and raw.strip():
        d = raw.strip().strip('"').rstrip("\\/")
        if not os.path.isabs(d):
            d = os.path.abspath(d)
        return d
    return DEFAULT_MODEL_DIR


def model_shard(n: int = 0) -> str:
    """Full path of shard `n` (default: the one the driver loads)."""
    return os.path.join(model_dir(), SHARDS[n] if 0 <= n < len(SHARDS) else DRIVER_SHARD)


def model_path() -> str:
    """The path passed to the driver / the harness."""
    return os.path.join(model_dir(), DRIVER_SHARD)


def shards_present() -> list:
    """(name, size) for each shard that exists in the configured directory."""
    out = []
    for name in SHARDS:
        p = os.path.join(model_dir(), name)
        if os.path.isfile(p):
            out.append((name, os.path.getsize(p)))
    return out


def describe() -> str:
    """One line for logs: the directory, and whether both shards are there."""
    d = model_dir()
    src = "SABAH_MODEL_DIR" if os.environ.get("SABAH_MODEL_DIR") else "default"
    have = shards_present()
    if len(have) == len(SHARDS):
        total = sum(s for _, s in have) / (1024.0 ** 3)
        return f"model dir {d} ({src}), {len(SHARDS)} shards, {total:.1f} GB"
    return f"model dir {d} ({src}), shards found: {[n for n, _ in have]}"


if __name__ == "__main__":
    print(describe())
    for name, size in shards_present():
        print(f"  {name}  {size / 1024.0 ** 3:.1f} GB")
