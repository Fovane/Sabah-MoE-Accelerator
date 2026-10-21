#!/usr/bin/env python3
"""Item 4 verification: run the SEG matrix and compare it to a reference.

One invocation runs the whole matrix for one SABAH_CUDA_GRAPH_SEG value:

  * -c 512 and -c 131072, two real prompts, 1024 tokens each
  * one 640-token run at -c 2048 that crosses the n_kv 256 and 512 boundaries

Each run goes through tools/phase0_measure.py (same env as `sabah run`), so the
record, the clocks and the byte counter are all written by the normal path.
Then fed.i32, argmax.i32 and logits.f32 are compared against the reference.

    python tools/seg_verify.py --tag ref  --seg 0      # reference
    python tools/seg_verify.py --tag seg1 --seg 1 --ref ref
    python tools/seg_verify.py --tag seg48 --seg 48 --ref ref
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
OUT = os.path.join(ROOT, "results", "phase0")

# (name, context, tokens, prompt file or None, literal prompt)
# At -c 512 the two real prompts (1630 and 1092 tokens) do not fit, so the 512
# cases use short prompts and still cross the n_kv 256 boundary. The 128K cases
# use the real prompts. kv640 crosses n_kv 256 and 512, where the compute buffer
# and the KV padding move and every segment has to recapture.
MATRIX = [
    ("c512-1", 512, 400, None, "Hello"),
    ("c512-2", 512, 400, None,
     "def fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n\nWhat is the time complexity?"),
    ("c128k-A", 131072, 1024, "prompts/chat_multiturn.txt", None),
    ("c128k-B", 131072, 1024, "prompts/coding_session.txt", None),
    ("kv640", 2048, 640, None, "Hello"),
]


def run_one(tag, seg, name, ctx, tokens, prompt_file, prompt):
    label = "%s-%s" % (tag, name)
    out_dir = os.path.join(OUT, label)
    main = os.path.join(out_dir, "main")
    if os.path.isdir(main):
        out_dir = main
    fed = os.path.join(out_dir, "fed.i32")
    if os.path.exists(fed) and os.path.getsize(fed) == 4 * tokens:
        print("  %-10s cached" % name)
        return out_dir
    cmd = [PY, os.path.join(HERE, "phase0_measure.py"), "--label", label,
           "--context", str(ctx), "--tokens", str(tokens),
           "--hot-bytes", "6.0e9", "--out-root", os.path.join(ROOT, "results", "phase0")]
    if prompt_file:
        cmd += ["--prompt-file", os.path.join(ROOT, prompt_file)]
    else:
        cmd += ["--prompt", prompt]
    env = dict(os.environ)
    env["SABAH_CUDA_GRAPH_SEG"] = str(seg)
    env["SABAH_CUDA_GRAPH_SEG_STATS"] = "1"
    log = os.path.join(OUT, label + ".log")
    with open(log, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        print("  %-10s exit=%d, see %s" % (name, rc, log))
        return None
    print("  %-10s done" % name)
    return out_dir


def resolve(d):
    """Prompt-file runs land in <label>/main/, literal-prompt runs in <label>/."""
    m = os.path.join(d, "main")
    if os.path.isdir(m) and os.path.exists(os.path.join(m, "logits.f32")):
        return m
    return d


def digest(path):
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def seg_stats(log):
    """The SABAH_SEG summary lines, for the report."""
    if not os.path.exists(log):
        return []
    with open(log, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [ln.strip() for ln in text.splitlines() if ln.startswith("SABAH_SEG")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seg", required=True, help="SABAH_CUDA_GRAPH_SEG value")
    ap.add_argument("--ref", default="", help="reference tag to compare against")
    args = ap.parse_args()

    print("tag=%s seg=%s ref=%s" % (args.tag, args.seg, args.ref or "-"))
    results = {}
    for (name, ctx, tokens, pf, prompt) in MATRIX:
        d = run_one(args.tag, args.seg, name, ctx, tokens, pf, prompt)
        results[name] = d

    print("\n%-10s %-16s %-16s %-16s %s" % ("case", "logits", "argmax", "fed", "verdict"))
    ok = True
    for (name, ctx, tokens, pf, prompt) in MATRIX:
        d = resolve(results.get(name)) if results.get(name) else None
        if d is None:
            print("%-10s %s" % (name, "run failed"))
            ok = False
            continue
        if not args.ref:
            print("%-10s %-16s %-16s %-16s reference" %
                  (name, digest(os.path.join(d, "logits.f32")),
                   digest(os.path.join(d, "argmax.i32")), digest(os.path.join(d, "fed.i32"))))
            continue
        ref = resolve(os.path.join(OUT, "%s-%s" % (args.ref, name)))
        row, same = [], True
        for f in ("logits.f32", "argmax.i32", "fed.i32"):
            a, b = digest(os.path.join(ref, f)), digest(os.path.join(d, f))
            same = same and a is not None and a == b
            row.append(b if a == b else "%s!=%s" % (a, b))
        ok = ok and same
        print("%-10s %-16s %-16s %-16s %s" % (name, row[0], row[1], row[2],
                                              "IDENTICAL" if same else "DIFF"))
        for ln in seg_stats(os.path.join(OUT, "%s-%s.log" % (args.tag, name))):
            print("    " + ln)
    print("\n%s" % ("ALL IDENTICAL" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
