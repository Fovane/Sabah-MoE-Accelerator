"""Append sections 21-22 (the 32K prefill, decode on the filled context) to REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 21. Phase 2.3: one real 32K-class prefill, at the best measured setting

`p2-32k-ub1024`: the item-6 128K prompt's first 90,317 characters (28,413
tokens), `-c 32768`, `-b 2048`, `-ub 1024`, slab 6.0e9, SEG=48, TILE3, `-t 8`.
Clocks across the whole run: steady SM 2805/2827/2850 MHz, memory 13801 MHz, P1
- **VALID**. `N=1`.

| | value |
|---|---|
| prefill wall | **1174.0 s** (19.6 min) |
| chunks | 14 (2048 tokens each, the last partial) |
| prefill disk | **1013.2 GB** (35.7 GB per 1000 tokens) |
| ms/token | **41.3** |
| disk per 2048-token chunk | 64.8 / 68.3 / 67.3 / 70.5 GB (first four) |
| VRAM peak | 13873 MiB |
| host WS peak | 9737 MiB, host available minimum 2533 MiB |
| exit | 0 |

Disk read time `read_ms = 4648.9 s` against a 1174 s wall, i.e. the disk was
overlapped 3.85x with compute - the same property the small sweep showed, so the
chunk time is set by the slower of the two, not by the disk alone.

### Projection to 128K, and why it was not run

Scaling the measured rate:

* disk: 35.7 GB per 1000 tokens x 131.07 = **4679 GB**
* wall at the measured 41.3 ms/token: **5416 s = 90.3 min**

Both exceed the brief's 60-minute threshold, so the full 128K prefill was **not
run**. The one setting that would bring it under 60 min is `-ub 2048`: it halves
the chunk read (section 20: 37.1 GB vs 65.9 GB per chunk), which projects to
~2340 GB and ~45 min. That setting is not qualified to recommend - its outputs
are not byte-identical and its only wall-time evidence is the N=1 outlier of
section 20 - so it stays opt-in and the 128K prefill stays a projection.

## 22. Phase 2.4: decode on the FILLED context - the number the README depends on

Same run, after the 28,413-token prefill, 2048 decode steps over a ~30K-token KV
depth. `tools/run_detail.py`, per-regime with min/median/max:

| regime | ms/token (min/med/max) | MB/token | misses/token |
|---|---|---|---|
| warm-up (first 128) | 0.00 / **312.45** / 796.25 | 342.5 | 499.5 |
| "steady" (last 256) | 103.17 / **309.95** / 664.96 | 343.5 | **504.0** |

tok/s: 1/median = **3.2** over the tail window, 3.61 over all 2048 decodes.

**The steady window is not steady.** At step 2048 the run still misses 504
experts per token - with 48 layers and top-10 that is 10.5 misses per layer,
i.e. every routed expert is a miss and the slab is not retaining this prompt's
working set. So there is no "steady state" to quote at this depth: the run is
still warming up when it ends.

What this does to the README's headline:

* the item-4 128K number (42.74 -> 30.36 ms/token) was measured on a **6817
  character** prompt, i.e. a ~3.7K-deep KV when the tail window was read. This
  run is the first measurement at a ~30K-deep KV, and it reads **309.95 ms**.
* the two differ by 10x, and the difference is the expert working set, not the
  KV depth: the same 271.69 ms tail appears in item 6 for the coding trace at
  only ~3.2K depth (REPORT 15.4), because that trace routes as widely.
* therefore the README quotes 30-33 tok/s as a **short-prompt, warm-tier**
  number only, and this run is the evidence for the limitation. No 128K-decode
  headline is claimed.

Cache counters for the whole run: `hits=40578015 misses=2432415
evictions=2423659` - misses and evictions are within 0.4% of each other, which is
the arithmetic of a thrashing tier rather than of a cold start.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 21. Phase 2.3" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended sections 21-22, %d bytes" % len(s + ADD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
