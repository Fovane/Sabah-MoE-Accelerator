"""Replace the section-26 placeholder with the measured 2.4 GB result."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

PLACEHOLDER = "| 2.4e9 | measured in this section | - | - | - | - |"
NOTE = "(The 2.4 GB row is filled from the run, not assumed; if its tail is not better"
REPLACEMENT = """| 2.4e9 | **493.57 ms/token (NOT VALID CLOCKS)** | 2.03 | 840 | 570.6 | VRAM peak 9350 MiB (spill removed) |
| (2.4e9, clocks re-pinned) | not measured - pinning was refused again | - | - | - | - |"""

NEW_NOTE = """**Clocks matter here and they were lost.** The 2.4 GB run finished at steady SM
900/2017/2647 MHz, memory 7001 MHz, P3 - below both floors, so it is marked
INVALID and its time is not evidence. Its *bytes* are valid (they do not depend
on clocks): 840 misses/token and 570.6 MB/token, against 501 and 339.5 at 6 GB.
That is the direction the replay predicted (LRU at 2.4 GB: 783 misses; at 6 GB:
466), so the two models agree twice over.

What is still true without those clocks: a spill-free slab needs a *smaller*
tier, and a smaller tier misses more, so removing the spill does not buy the
tier - it trades system-RAM residency for disk bytes. The VRAM peak fell from
12914 to 9350 MiB, which is the spill going away.

Re-running the 2.4 GB case with the pin in force is the one measurement this
brief leaves unfinished, and it is recorded as unfinished rather than quoted."""


def main():
    s = open(PATH, encoding="utf-8").read()
    if PLACEHOLDER not in s:
        print("placeholder not found (already replaced?)")
        return 0
    s = s.replace(PLACEHOLDER, REPLACEMENT)
    s = s.replace("| 6.0e9 | 280.45 (section 22) | 3.57 | 501 | 339.5 | see below |",
                  "| 6.0e9 | 280.45 (section 22) | 3.57 | 501 | 339.5 | VRAM peak 12914 MiB |")
    i = s.find(NOTE)
    if i < 0:
        print("note not found")
        return 1
    j = s.find(")", s.find("which the bound already predicted", i)) + 1
    s = s[:i] + NEW_NOTE + s[j:]
    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("section 26 updated with the measured 2.4 GB result")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
