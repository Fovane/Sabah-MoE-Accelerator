"""Restore the model path the scrubber rewrote in tools/ and build scripts.

The scrub pass (commit 5c5b0a5) rewrote every tracked file including the code
that reads a model path, which turned `D:\Strata-data\...` into
`D:\Strata-data\...` and made every harness run fail to load the model.
This restores the drive letter in the files that need it.

    python tools/fix_model_path.py [--check]
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# real paths, per machine
DRIVES = {
    "D": [r"D:\Strata-data\models\IQ3_S"],
    "C": [r"C:\Strata-data\models\IQ3_S",
          r"C:\Projects\quantization\Sabah-MoE-Accelerator"],
}


def fix(path):
    with open(path, "rb") as f:
        raw = f.read()
    if b"$DRIVE" not in raw:
        return False
    txt = raw.decode("utf-8", errors="surrogateescape")
    # D:\Strata-data... -> D:\Strata-data...
    txt = re.sub(r"\D:\\", "D:\\\\", txt)
    txt = re.sub(r"\D:/", "D:/", txt)
    # the repo's own root, where a tool needs it
    txt = txt.replace("C:\\Projects\\quantization\\Sabah-MoE-Accelerator", os.path.abspath(ROOT).replace("\\", "\\\\"))
    with open(path, "wb") as f:
        f.write(txt.encode("utf-8", errors="surrogateescape"))
    return True


def main():
    check = "--check" in sys.argv
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    rels = [p for p in r.stdout.splitlines()
            if p.startswith(("tools/", "sabah/", "patches/")) and 
            os.path.isfile(os.path.join(ROOT, p))]
    touched = [p for p in rels if fix(os.path.join(ROOT, p)) or
               b"$DRIVE" in open(os.path.join(ROOT, p), "rb").read()]
    if check:
        still = [p for p in rels
                 if b"$DRIVE" in open(os.path.join(ROOT, p), "rb").read()]
        print(f"code files still carrying a placeholder: {len(still)}")
        for p in still:
            print("  ", p)
        return 1 if still else 0
    print(f"restored placeholders in {len(touched)} code/config files")
    for p in touched:
        print("  ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
