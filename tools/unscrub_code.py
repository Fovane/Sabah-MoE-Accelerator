"""Restore the code and config files the scrubber should never have touched.

`scrub_paths.py` scrubs every tracked file, which includes the build scripts, the
harness defaults and the tooling: it rewrote `C:\\...` model paths into
`D:\\...` and broke `tools/_build_*.bat`, `tools/_cuda_flags.txt`,
`tools/phase0_measure.py` and `sabah/tools/stream.py`. Evidence files under
`results/` and prompts may be scrubbed; executable and config files may not.

    python tools/unscrub_code.py [--check]
"""
from __future__ import annotations

import argparse
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# code and config: these are read by tools, so a placeholder breaks them
CODE = (
    "tools/", "sabah/", "patches/", "docs/", "install.bat", "pyproject.toml",
    "QUICKSTART.md", "README.md", "CHANGELOG.md", "THIRD_PARTY.md",
)


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return r.stdout.splitlines()


def placeholder_in(text, ph):
    return ph in text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    dirty = []
    restore = []
    for rel in tracked():
        if not rel.startswith(CODE):
            continue
        full = os.path.join(ROOT, rel)
        if not os.path.isfile(full):
            continue
        try:
            raw = open(full, "rb").read()
        except OSError:
            continue
        if b"$DRIVE" not in raw and b"C:\\Projects\\quantization\\Sabah-MoE-Accelerator" not in raw:
            continue
        dirty.append(rel)
        restore.append(rel)

    print(f"code/config files carrying a placeholder: {len(dirty)}")
    for d in dirty:
        print("  ", d)
    if restore and not args.check:
        subprocess.run(["git", "checkout", "HEAD", "--", *restore], cwd=ROOT,
                       capture_output=True, text=True)
        print("restored from HEAD:", len(restore))
    return 1 if (dirty and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
