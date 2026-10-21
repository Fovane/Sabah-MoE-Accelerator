"""Final hygiene report (Phase 5.2): size, paths, personal data.

    python tools/hygiene_report.py

Runs the checks the release checklist asks for and prints one line per check,
so the report can be pasted without re-deriving it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def check_sizes():
    r = git("ls-files", "-z")
    tracked = [p for p in r.stdout.split("\0") if p]
    big = []
    for p in tracked:
        f = os.path.join(ROOT, p)
        if os.path.isfile(f) and os.path.getsize(f) > 50 * 1024 * 1024:
            big.append((os.path.getsize(f) // (1024 * 1024), p))
    return len(tracked), big


def check_paths():
    """Tracked files carrying an absolute path or the account name."""
    r = git("ls-files")
    user = os.environ.get("USERNAME", "")
    drive = re.compile(r"[A-Za-z]:\\+")
    hits = []
    for p in r.stdout.splitlines():
        f = os.path.join(ROOT, p)
        if not os.path.isfile(f):
            continue
        try:
            t = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if drive.search(t) or (user and user in t):
            hits.append(p)
    return hits


def check_untracked():
    r = git("status", "--porcelain")
    return [l[3:] for l in r.stdout.splitlines() if l.startswith("?? ")]


def main():
    n, big = check_sizes()
    print(f"tracked files                 : {n}")
    print(f"tracked files over 50 MB      : {len(big)}")
    for s, p in big:
        print(f"    {s} MB  {p}")

    hits = check_paths()
    print(f"tracked files with a local path or the user name: {len(hits)}")
    for p in hits[:10]:
        print("   ", p)

    unt = check_untracked()
    print(f"untracked paths in the tree   : {len(unt)}")
    for p in unt[:6]:
        print("   ", p)

    ok = not big and not hits
    print("VERDICT:", "CLEAN" if ok else "NEEDS WORK")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
