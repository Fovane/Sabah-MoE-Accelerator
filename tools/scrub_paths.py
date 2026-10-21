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

DRIVE_RE = re.compile(r"(?<![A-Za-z0-9$.\\])([A-Za-z]):\\")
RE2 = re.compile(r"(?<![A-Za-z0-9$.])([A-Za-z]):\\\\")


def scrub_str(s, root):
    # Order matters: the doubled-spelling form ("C:\\Projects...", how a path
    # appears inside a parsed JSON string value) contains the single-spelling
    # form as a substring, so replacing the single form first would mangle the
    # doubled one into another path instead of removing it.
    literal = []
    # longest / most specific first
    for sep in ("\\\\\\", "\\\\", "\\"):
        literal.append("C:" + sep + "Projects" + sep + "quantization" + sep + "Sabah-MoE-Accelerator")
        literal.append("C:" + sep + "Projects" + sep + "quantization")
    literal += [root, root.replace("\\", "/"), root.replace("\\", "\\\\"),
                root.replace("\\", "\\\\\\")]
    # the model path, on any drive and in any spelling
    for drive in ("C:", "D:", "E:", "F:"):
        for sep in ("\\\\\\", "\\\\", "\\"):
            literal += [drive + sep + "Strata-data",
                        drive + sep + "Models",
                        drive + sep + "models"]
    for form in literal:
        if form and form in s:
            s = s.replace(form, PLACEHOLDER)
    s = re.sub(r"[A-Za-z]:\\\\+", lambda _m: DRIVE + "\\", s)
    user = os.environ.get("USERNAME")
    if user and user in s:
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
    """(was dirty, was rewritten) for one file."""
    try:
        text = open(path, "r", encoding="utf-8", errors="strict").read()
    except (UnicodeDecodeError, OSError):
        return False, False
    # any spelling of a local path. A JSON string value carries the path with
    # doubled backslashes ("C:\\Projects\\..."), a raw log with single ones, so
    # both spellings of the root have to be tested or the file is skipped.
    root_double = root.replace("\\", "\\\\")
    looks_dirty = (
        DRIVE_RE.search(text) is not None
        or RE2.search(text) is not None
        or root.lower() in text.lower()
        or root_double.lower() in text.lower()
        or "Strata-data" in text
        or "Sabah-MoE-Accelerator" in text
        or os.environ.get("USERNAME", "\x00") in text
    )
    if not looks_dirty:
        return False, False
    if path.endswith(".json"):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return False, False
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
        return True, True
    # dirty-looking but unchanged: a placeholder that is already a placeholder must
    # not be reported as needing a scrub, or the check output cannot be trusted
    return new != text, False


SKIP_PREFIX = ("tools/", "sabah/", "patches/", "docs/")
SKIP_EXACT = ("install.bat", "pyproject.toml", "QUICKSTART.md", "README.md",
              "CHANGELOG.md", "THIRD_PARTY.md", "LICENSE")


def tracked():
    r = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    paths = []
    for p in r.stdout.splitlines():
        if not p.endswith((".json", ".jsonl", ".log", ".err", ".md", ".txt", ".csv", ".tsv")):
            continue
        # never scrub anything a tool reads: a placeholder in a build script, a
        # harness default or the launcher breaks them, and the leak they carry
        # is the repo's own layout, which the README documents.
        if p.startswith(SKIP_PREFIX) or p in SKIP_EXACT:
            continue
        paths.append(os.path.join(ROOT, p))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(ROOT)

    dirty = []
    changed = 0
    for f in tracked():
        was, now = wash(f, root, write=not args.check)
        if was:
            dirty.append(os.path.relpath(f, ROOT))
        if now:
            changed += 1
    verb = "would scrub" if args.check else "scrubbed"
    print(f"{verb} {len(dirty)} tracked files ({changed} rewritten)")
    for d in dirty[:20]:
        print("  ", d)
    return 1 if (dirty and args.check) else 0


if __name__ == "__main__":
    raise SystemExit(main())
