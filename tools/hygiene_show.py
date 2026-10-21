"""Show what repo_hygiene.py still finds in the named files."""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
user = os.environ.get("USERNAME") or ""
pats = []
drive = re.compile(r"[A-Za-z]:\\+")
if drive:
    pats.append((drive, "drive-letter path"))
if user:
    pats.append((re.compile(r"\b" + re.escape(user) + r"\b"), "username"))
workspace = re.compile(re.escape(os.path.abspath(ROOT)), re.IGNORECASE)
pats.append((workspace, "workspace path"))

for rel in sys.argv[1:]:
    t = open(os.path.join(ROOT, rel), encoding="utf-8", errors="ignore").read()
    for rx, why in pats:
        hits = list(rx.finditer(t))
        if hits:
            m = hits[0]
            print(f"{rel}: {why}: {len(hits)} hit(s), match {m.group(0)!r}")
            print("   ", repr(t[max(0, m.start() - 40):m.start() + 30]))
