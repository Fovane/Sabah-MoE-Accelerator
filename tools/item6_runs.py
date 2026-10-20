"""Item 6: real workloads at -c 131072 with the slab item 3 chose.

Four measurements, pinned clocks, all through phase0_measure.py so the record
shape matches every other table:

  chat      prompts/chat_multiturn.txt, 1024 decode tokens
  code      prompts/coding_session.txt, 1024 decode tokens
  p64k      prompts/p64k.txt, 1 decode token (prefill only)
  p128k     prompts/p128k.txt, 1 decode token (prefill only)

The prefill pair is what answers "does chunked prefill re-stream the pool":
the per-prompt `bytes_fetched` is compared against the prompt length.

    .venv\\Scripts\\python.exe tools\\item6_runs.py --slab 6.0e9
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable
MEASURE = os.path.join(HERE, "phase0_measure.py")

RUNS = [
    ("item6-chat", "prompts/chat_multiturn.txt", 1024),
    ("item6-code", "prompts/coding_session.txt", 1024),
    ("item6-prefill64k", "prompts/p64k.txt", 1),
    ("item6-prefill128k", "prompts/p128k.txt", 1),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slab", default="6.0e9")
    ap.add_argument("--context", type=int, default=131072)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    for label, prompt, tokens in RUNS:
        if a.only and a.only not in label:
            continue
        argv = [PY, MEASURE, "--label", label, "--tokens", str(tokens),
                "--context", str(a.context), "--hot-bytes", a.slab,
                "--prompt-file", os.path.join(ROOT, prompt), "--gpu-mem"]
        t0 = time.time()
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
        wall = time.time() - t0
        p = os.path.join(ROOT, "results", "phase0", label, "measurement.json")
        if not os.path.exists(p):
            print(f"{label}: FAILED exit={r.returncode} wall={wall:.0f}s")
            print((r.stdout or "")[-1500:])
            print((r.stderr or "")[-800:])
            continue
        rec = json.load(open(p, encoding="utf-8"))
        reg = rec.get("regimes") or {}
        steady = (reg.get("steady") or {}).get("median_ms")
        mem = rec.get("memory") or {}
        gm = rec.get("gpu_memory") or {}
        print(f"{label}: wall={wall:.0f}s exit={rec.get('exit_code')} "
              f"prompt_tokens={rec.get('prompt_tokens_count')} "
              f"steady={steady} ms/tok "
              f"({(1000.0/steady if steady else 0):.2f} tok/s) "
              f"bytes/tok={(rec.get('read_bytes_per_decoded_token') or 0)/1e6:.0f} MB "
              f"hit={100.0*(rec.get('hits') or 0)/max(1,(rec.get('hits') or 0)+(rec.get('misses') or 0)):.1f}% "
              f"vram_peak={mem.get('vram_peak_mib')} "
              f"ded_peak={gm.get('dedicated_peak_mib')} "
              f"shared_peak={gm.get('shared_peak_mib')} "
              f"clock_ok={rec.get('clock_valid')}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
