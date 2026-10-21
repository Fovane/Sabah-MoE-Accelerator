"""Scrub local absolute paths and the account name out of tracked files.

The harness records the argv and the environment of every run, which leaks the
workspace path, the model path and (through `USERNAME`) the account name. This
replaces them with stable placeholders so the evidence keeps its shape, its
numbers and its hashes.

JSON and JSONL are rewritten by parsing and replacing inside string values only,
so no escape sequence can be broken. Logs, markdown and text are replaced as
raw text (no structural keys there).

    python tools/scrub_paths.py [--check]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLACEHOLDER = "$WORKSPACE"
DRIVE = "$DRIVE"

DRIVE_RE = re.compile(r"(?<![A-Za-z0-9$.\\])([A-Za-z]):\\\\")


def scrub_str(s, root):
    s = s.replace(root, PLACEHOLDER)
    s = s.replace(root.replace("\\", "/"), PLACEHOLDER)
    s = DRIVE_RE.sub(DRIVE + "\\\\", s)
    user = os.environ.get("USERNAME")
    if user:
        s = s.replace(user, "$USER")
    return s


def scrub_value(v, root):
    if isinstance(v, str):
        return scrub_str(v, root)
    if isinstance(v, list):
        return [scrub_value(x, root) for x in v]
    if isinstance(v, dict):
        return {k: scrub_value(x, root) for k, x in v.items()}
    return v


def wash(path, root, write):
    try:
        text = open(path, "r", encoding="utf-8", errors="strict").read()
    except (UnicodeDecodeError, OSError):
        return False
    if root.lower() not in text.lower() and DRIVE_RE.search(text) is None:
        return False
    if path.endswith(".json"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return False
        new = json.dumps(scrub_value(obj, root), indent=1, ensure_ascii=False) + "\n"
    elif path.endswith(".jsonl"):
        out = []
        for line in text.splitlines():
            if not line.strip():
                out.append(line)
                continue
            try:
                out.append(json.dumps(scrub_value(json.loads(line), root),
                                      ensure_ascii=False))
            except json.JSONDecodeError:
                out.append(line)
        new = "\n".join(out) + "\n"
    else:
        new = scrub_str(text, root)
    if new != text and write:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return True


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    return [os.path.join(ROOT, p) for p in r.stdout.splitlines()
            if p.endswith((".json", ".jsonl", ".log", ".md", ".txt", ".csv", ".py", ".bat"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(ROOT)

    dirty = []
    for f in tracked():
        if wash(f, root, write=not args.check):
            dirty.append(os.path.relpath(f, ROOT))
    print(f"scrubbed {len(dirty)} tracked files")
    for d in dirty:
        print("  ", d)
    return 1 if (dirty and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
