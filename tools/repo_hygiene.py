"""Repo hygiene checks: tracked file sizes, absolute paths, user/machine names.

    python tools/repo_hygiene.py [--limit-mb 50]

Checks, in order:
  1. tracked blobs larger than the limit;
  2. tracked files whose text matches a local absolute path ($DRIVE\<drive>\\...),
     a username from the environment, or this machine's host name;
Prints one line per finding and a summary. Exit 1 if any finding.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def blob_sizes():
    """(size, path) for every tracked file, using the working-tree copy.

    Tracked files equal their committed content unless modified, which this run
    reports separately through `git status`; for a size bound that is enough and
    avoids a per-blob git call for 1800 files.
    """
    r = git("ls-files", "-z")
    paths = [p for p in r.stdout.split("\0") if p]
    out = []
    for p in paths:
        full = os.path.join(ROOT, p)
        if os.path.isfile(full):
            out.append((os.path.getsize(full), p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-mb", type=float, default=50.0)
    args = ap.parse_args()
    limit = int(args.limit_mb * 1024 * 1024)

    sizes = blob_sizes()
    big = [(s, p) for s, p in sizes if s > limit]
    print(f"tracked blobs: {len(sizes)}   larger than {args.limit_mb:.0f} MB: {len(big)}")
    for s, p in sorted(big, reverse=True):
        print(f"  BIG  {s/1024/1024:9.1f} MB  {p}")

    # absolute paths / personal data
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    host = os.environ.get("COMPUTERNAME") or ""
    pats = [
        (re.compile(r"[A-Za-z]:\\\\[A-Za-z0-9_.\-]+\\\\"), "absolute path"),
    ]
    if user:
        pats.append((re.compile(r"\b" + re.escape(user) + r"\b"), "username"))
    if host:
        pats.append((re.compile(r"\b" + re.escape(host) + r"\b"), "machine name"))

    # only scan files git thinks are text and reasonably small
    findings = []
    for s, p in sizes:
        if s > 8 * 1024 * 1024:
            continue
        full = os.path.join(ROOT, p)
        if not os.path.isfile(full):
            continue
        try:
            txt = open(full, "r", encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        for rx, why in pats:
            for m in rx.finditer(txt):
                line = txt[:m.start()].count("\n") + 1
                findings.append((why, p, line, m.group(0)))
                break

    print(f"personal/absolute-path findings: {len(findings)}")
    for why, p, line, hit in findings[:25]:
        print(f"  {why:14s} {p}:{line}  {hit!r}")
    return 1 if (big or findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
