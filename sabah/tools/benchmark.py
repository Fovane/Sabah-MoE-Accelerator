"""Comparable reference benchmark harness.

This command measures the installed reference engine when requested and emits
machine-readable provenance.  It never calls a block benchmark a full-model
speedup, and it reports speedup as unavailable unless both full-model sides
were measured under the same workload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import time

from sabah.core.model_inspector import inspect_model
from sabah.server.openai_proxy import find_llama_cli


def sha256_file(path: str, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _parse_tok_s(text: str) -> float | None:
    vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second", text)
    if vals:
        return float(vals[-1])
    # llama-cli's simple-io progress line uses a shorter unit.
    vals = re.findall(r"Generation:\s*([0-9]+(?:\.[0-9]+)?)\s*t/s", text)
    return float(vals[-1]) if vals else None


def run_reference(model: str, exe: str, prompt: str, tokens: int,
                  context: int, threads: int = 0) -> tuple[float | None, str, list[str], int]:
    cmd = [exe, "-m", model, "-p", prompt, "-n", str(tokens), "-c", str(context),
           "--temp", "0", "--no-display-prompt", "--single-turn", "--simple-io",
           "--perf",
           "--n-gpu-layers", "0", "--cpu-moe"]
    if threads > 0:
        cmd += ["-t", str(threads), "-tb", str(threads)]
    t0 = time.time()
    run = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", timeout=3600)
    text = run.stdout
    return _parse_tok_s(text), text, cmd, run.returncode


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sabah benchmark")
    ap.add_argument("model")
    ap.add_argument("--prompt", default="Reply with exactly: SABAH_OK")
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--llama-cli", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    prof = inspect_model(args.model)
    if not prof.supported:
        print("unsupported model; refusing benchmark")
        return 2
    exe = find_llama_cli(args.llama_cli)
    meta = {
        "schema": 1,
        "sabah_commit": os.environ.get("SABAH_COMMIT", "unknown"),
        "model": os.path.abspath(args.model),
        "model_sha256": None,
        "architecture": prof.architecture,
        "os": platform.platform(),
        "python": platform.python_version(),
        "prompt": args.prompt,
        "generation_tokens": args.tokens,
        "context": args.context,
        "sampling": {"temperature": 0, "decoding": "greedy"},
        "reference_command": None,
        "reference_tok_s": None,
        "sabah_tok_s": None,
        "measured_speedup": None,
        "measured_speedup_available": False,
        "notes": ["full-model Sabah backend is not integrated; speedup unavailable"],
    }
    print("SABAH - full-model benchmark (reference provenance)")
    print("model      : %s" % os.path.basename(args.model))
    print("architecture: %s" % prof.architecture)
    print("reference  : %s" % exe)
    if args.dry_run:
        meta["reference_command"] = [exe, "-m", args.model, "-p", args.prompt,
                                      "-n", str(args.tokens)]
        print("dry-run    : no model bytes read")
    else:
        print("hashing model shards ...")
        meta["model_sha256"] = sha256_file(args.model)
        tok_s, output, cmd, returncode = run_reference(
            args.model, exe, args.prompt, args.tokens, args.context, args.threads)
        meta["reference_command"] = cmd
        meta["reference_tok_s"] = tok_s
        meta["reference_returncode"] = returncode
        meta["reference_returned_output"] = bool(output)
        meta["reference_output_tail"] = output[-4000:]
        if returncode != 0:
            print("Reference: FAILED (return code %d)" % returncode)
            if args.json_out:
                with open(args.json_out, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            return 2
        print("Reference: %s tok/s" % ("%.3f" % tok_s if tok_s is not None else "unparsed"))
    print("Sabah:     unavailable")
    print("Speedup:   Measured speedup unavailable.")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print("wrote      : %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
