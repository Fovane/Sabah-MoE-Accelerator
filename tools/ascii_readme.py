"""Replace the unicode dashes in README.md with ASCII, per the repo's own rule.

    python tools/ascii_readme.py [--check]

Only ASCII replacements: em dash -> " - ", en dash -> "-", arrow -> "->",
times -> "x", ellipsis -> "...". Nothing else is touched, so the diff is
exactly the characters replaced.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")

REPL = [
    ("\u2014", " - "),
    ("\u2013", "-"),
    ("\u2192", "->"),
    ("\u00d7", "x"),
    ("\u2026", "..."),
]


def main():
    check = "--check" in sys.argv
    s = open(PATH, encoding="utf-8").read()
    n = sum(s.count(a) for a, _b in REPL)
    if check:
        print("unicode offenders in README:", n)
        return 1 if n else 0
    if not n:
        print("nothing to replace")
        return 0
    for a, b in REPL:
        s = s.replace(a, b)
    with open(PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print("replaced %d unicode characters" % n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
