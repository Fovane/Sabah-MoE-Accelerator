"""Restore a real path where the scrub pass left a placeholder in *code*.

`scrub_paths.py` only scrubs evidence, but the first full-tree pass (commit
5c5b0a5) also rewrote code that reads a model path: `tools/phase0_measure.py`,
`tools/_build_*.bat`, `tools/_cuda_flags*.txt` and the launcher defaults. That
turned `D:\Strata-data\...` into `D:\Strata-data\...` and made every harness
run fail with "failed to load model".

This walks the code and config files and replaces the placeholders with the
real path, but never touches a line that *defines* a placeholder - rewriting
`PLACEHOLDER = "C:\Projects\quantization\Sabah-MoE-Accelerator"` into `PLACEHOLDER = "C:\Projects\..."` would break
the scrubber itself, which is exactly what happened once.

    python tools/fix_model_path.py [--check]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CODE = ("tools/", "sabah/", "patches/")

# a line that declares a placeholder constant, not a use of one
DEF_RE = re.compile(r"""^\s*("?)(PLACEHOLDER|DRIVE|ROOT)\1\s*=""")


def is_def(line):
    return DEF_RE.match(line) is not None


def restore(txt, root):
    lines = txt.split("\n")
    out = []
    for line in lines:
        if is_def(line):
            out.append(line)
            continue
        # D:\... -> D:\...
        line = re.sub(r"\D:\+", "D:\\\\", line)
        line = line.replace("D:", "D:")
        line = line.replace("D:/", "D:/")
        line = line.replace("C:\Projects\quantization\Sabah-MoE-Accelerator", root)
        out.append(line)
    return "\n".join(out)


def code_files():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    for p in r.stdout.splitlines():
        if not p.startswith(CODE):
            continue
        full = os.path.join(ROOT, p)
        if os.path.isfile(full):
            yield p, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(ROOT)

    dirty, fixed = [], []
    for rel, full in code_files():
        try:
            t = open(full, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if "D:" not in t and "C:\Projects\quantization\Sabah-MoE-Accelerator" not in t:
            continue
        dirty.append(rel)
        if not args.check:
            new = restore(t, root)
            if new != t:
                with open(full, "w", encoding="utf-8", newline="") as f:
                    f.write(new)
                fixed.append(rel)

    verb = "would restore" if args.check else "restored"
    print(f"{verb} placeholders in {len(dirty)} code files")
    for d in dirty:
        print("  ", d)
    return 1 if (dirty and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
