"""Correct the KV labels: the shipped default is K=f16 with V=q4_0, not f16."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

OLD_A = """| KV type | steady ms/token | tok/s | expert reads | disk bytes |
|---|---|---|---|---|
| f16 (default) | **30.05** | **33.28** | 33,557 | 22.7 GB |
| q8_0 | 188.32 | 5.31 | **84,055** | 56.7 GB |
"""
NEW_A = """| KV type | steady ms/token | tok/s | expert reads | disk bytes |
|---|---|---|---|---|
| build default, `K=f16` / **`V=q4_0`** | **30.05** | **33.28** | 33,557 | 22.7 GB |
| `-ctk q8_0 -ctv q8_0` | 188.32 | 5.31 | **84,055** | 56.7 GB |
"""

OLD_B = """* **`-ctk q8_0 -ctv q8_0` is an opt-in "numerically equivalent" mode** - the
  outputs are not byte-identical to the f16 path, and on this rig it is a
  6.3x decode-time regression, not a free saving;
"""
NEW_B = """* **`-ctk q8_0 -ctv q8_0` is an opt-in "numerically equivalent" mode** - the
  outputs are not byte-identical to the default path, and on this rig it is a
  6.3x decode-time regression, not a free saving, while the default it replaces
  already quantises V (`-ctv` default is `q4_0` in this build);
* **the label "f16" was wrong and is corrected here**: the "fast" column above
  was measured without any `-ctk/-ctv` flags, so it is the build default
  `K=f16` + **`V=q4_0`**, not an unquantised cache. A fully unquantised
  `-ctk f16 -ctv f16` run is **not measured**, and the V quantisation is exactly
  the mechanism REPORT 28 describes, so the default may already be paying part
  of the price it claims to charge `q8_0` for;
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    for old, new in ((OLD_A, NEW_A), (OLD_B, NEW_B)):
        assert old in s, "NOT FOUND: " + old[:60]
        s = s.replace(old, new, 1)
    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("corrected the KV labels in REPORT 28")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
