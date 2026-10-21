#!/usr/bin/env python
"""Build a long *real-text* prefill prompt from this repository.

Used for item 6: one 64K-token and one 128K-token prefill whose content is
actual source and documentation rather than repeated filler, so the router
sees a realistic distribution instead of one token repeated thousands of
times.

    python tools/make_long_prompt.py --target-tokens 65536 --out prompts/p64k.txt
    python tools/make_long_prompt.py --target-tokens 131072 --out prompts/p128k.txt

Token count is approximated at 3.6 characters per token for this vocabulary;
the diag prints the exact count it tokenizes, and that printed count is what
the report uses.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHARS_PER_TOKEN = 3.6

# Real, text-only sources in reading order: docs first, then CUDA/C++ runtime,
# then tools. Everything that is generated, binary or a third_party checkout is
# skipped so the prompt is stable across machines.
GLOBS = [
    "README.md",
    "docs/*.md",
    "sabah/runtime/cuda/*.cu",
    "sabah/runtime/cuda/*.cuh",
    "sabah/tools/*.py",
    "tools/*.py",
    "results/phase0/REPORT.md",
]


def collect() -> list[Path]:
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in GLOBS:
        for p in sorted(ROOT.glob(pattern)):
            if p.is_file() and p not in seen:
                seen.add(p)
                files.append(p)
    return files


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    target_chars = int(args.target_tokens * CHARS_PER_TOKEN)
    files = collect()
    if not files:
        print("no sources found", file=sys.stderr)
        return 1

    chunks: list[str] = []
    total = 0
    passes = 0
    while total < target_chars:
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            header = f"\n\n===== {p.relative_to(ROOT).as_posix()} (pass {passes + 1}) =====\n\n"
            chunks.append(header)
            chunks.append(text)
            total += len(header) + len(text)
            if total >= target_chars:
                break
        passes += 1
        if passes > 200:
            break

    body = "".join(chunks)[:target_chars]
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {out}  {len(body):,} chars  ~{len(body)/CHARS_PER_TOKEN:,.0f} tokens  "
          f"{passes} pass(es) over {len(files)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
