"""List every tracked file that mentions the model directory, with context.

    python tools/find_model_refs.py
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFS = [
    (re.compile(r"D:\\Strata-data"), "D:\\Strata-data"),
    (re.compile(r"D:/Strata-data"), "D:/Strata-data"),
    (re.compile(r"Strata-data"), "Strata-data (any drive)"),
]


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [p for p in r.stdout.splitlines() if p]


def main():
    show_all = "--all" in sys.argv
    groups = {}
    for rel in tracked():
        f = os.path.join(ROOT, rel)
        if not os.path.isfile(f):
            continue
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        hits = []
        for i, line in enumerate(t.splitlines(), 1):
            for rx, label in REFS:
                if rx.search(line):
                    hits.append((i, label, line.strip()[:110]))
        if hits:
            groups[rel] = hits

    print(f"tracked files mentioning the model directory: {len(groups)}")
    for rel, hits in sorted(groups.items()):
        print(f"\n== {rel}")
        for i, label, line in hits[:8]:
            print(f"   {i:5d} [{label}] {line}")
        if len(hits) > 8:
            print(f"   ... {len(hits) - 8} more")


if __name__ == "__main__":
    raise SystemExit(main())
