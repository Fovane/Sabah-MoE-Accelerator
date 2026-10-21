"""Create the GitHub release for v1.1.0 from the prepared release notes.

    python tools/gh_release.py [--draft]

Uses the token from git-credential-manager (same one push uses) and the body of
RELEASE_NOTES_rc1.md. Nothing is printed except the release URL.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTES = os.path.join(ROOT, "RELEASE_NOTES_rc1.md")
OWNER = "Fovane"
REPO = "Sabah-MoE-Accelerator"
TAG = "v1.1.0"


def token():
    p = subprocess.run(["git", "credential", "fill"],
                       input="protocol=https\nhost=github.com\n\n",
                       capture_output=True, text=True, cwd=ROOT)
    if p.returncode != 0:
        raise SystemExit("could not read the credential")
    for line in p.stdout.splitlines():
        if line.startswith("password="):
            return line[len("password="):]
    raise SystemExit("no password in the credential")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", action="store_true")
    args = ap.parse_args()

    body = open(NOTES, encoding="utf-8").read()
    payload = {
        "tag_name": TAG,
        "target_commitish": "main",
        "name": "v1.1.0 - streaming MoE that can be checked",
        "body": body,
        "draft": args.draft,
        "prerelease": False,
    }
    req = urllib.request.Request(
        "https://api.github.com/repos/%s/%s/releases" % (OWNER, REPO),
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + token(),
            "Accept": "application/vnd.github+json",
            "User-Agent": "sabah-release",
            "Content-Type": "application/json",
        })
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode("utf-8"))
            print("created:", d.get("html_url"))
            print("upload_url:", d.get("upload_url"))
            return 0
    except urllib.error.HTTPError as e:
        print("HTTP %s: %s" % (e.code, e.read().decode("utf-8", "replace")[:600]))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
