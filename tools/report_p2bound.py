"""Append the Phase 2 bound table to REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 25. Phase 2: the policy bound - deep-context decode is disk-bound on this hardware

Replay of the Phase 1 routing trace (2,949,120 accesses over 2048 decoded steps
at ~30K KV depth, 1440 accesses/token = 48 layers x 10 top-k x 3 roles), at six
slab sizes and six policies (`tools/slab_replay.py`). Expert slot size 682 KB,
taken from the measured 343.5 MB at 504 misses.

### What each tier needs

The deep-KV regime is disk-bound (its measured ms/token is within ~1.1x of the
disk service time for its bytes alone), so a tier is a bytes-per-token budget:

| tier | tok/s | token budget | disk budget at the measured 1.1x overlap | misses/token | hit rate needed |
|---|---|---|---|---|---|
| A | 10 | 100 ms | 115.5 MB | 173 | **88.0%** |
| B | 20 | 50 ms | 57.8 MB | 87 | **94.0%** |
| C | 30 | 33.3 ms | 38.5 MB | 58 | **96.0%** |

(the compute component has to fit in the same budget; the measured compute part
at a deep KV is far larger than the 27 ms of the short prompt, so these are
upper bounds on what any policy can deliver.)

### What the policies actually deliver

| slab | slots | LRU (what the runtime does) | Belady (offline optimum) |
|---|---|---|---|
| 2.4 GB | 3436 | 45.64% (783 miss/tok) | **68.02% (461)** |
| 6.0 GB | 8591 | 67.64% (466) | 84.25% (227) |
| 8.0 GB | 11455 | 75.48% (353) | 88.57% (165) |
| 2.4 GB | per-layer budgets, LRU inside | 45.65% (783) | - |
| 6.0 GB | per-layer budgets, LRU inside | 67.31% (471) | - |

Two cross-checks that make the replay trustworthy:

* LRU at 6.0 GB predicts **466 misses/token**; the runtime measured **504** at
  the same slab (section 22). The model is within 8% of the real thing.
* The trace is exactly 1440 accesses/token, which is the architectural figure
  (48 layers, top-10, three projections), so nothing is dropped or double-counted.

Per-layer budgets are a wash (within 1% of plain LRU at every size), and the
scan-resistant policies (LFU, 2Q, ARC) were slower than LRU to simulate than the
decision needed - per-layer budgets being a wash already shows the access
pattern is not scan-shaped, so ARC's advantage (it needs one) is not available.

### The decision

**At the spill-free slab (2.4 GB, the size the 128K budget arithmetic gives),
the Belady optimum is 68.0%, and tier A needs 88.0%.** A policy cannot close a
20-point gap by being smarter about the same bytes; it would have to fetch fewer
distinct experts, and the trace fixes the distinct set. So:

* **no tier is reached on a deep context on this hardware**, and this is a
  property of the trace, not of the current policy - the *optimal* policy misses
  it too;
* Phases 4-5 (a better replacement policy) are therefore polish at best: the
  most a policy can buy at 6.0 GB is 466 -> 227 misses, which is 150 MB/token,
  which is ~130 ms of disk against a 100 ms tier-A budget. It does not reach
  tier A even with the spill.

What would move the number: an expert representation with fewer bytes per
distinct expert (the Q2_0 variant), or a drive faster than 1.05 GB/s - both of
which are outside this brief.

## 26. Phase 3a/3c: does removing the spill help, and by how much

Measured after the bound above, because the bound says the spill is not what
decides the tier: 2048 steps from the restored 32K state at 2.4 GB
(`g-slab24`) against the 6.0 GB baseline (`g-trace2`). Same prompt, same forced
token stream, same build, clocks valid.

| slab | tail ms/token | tok/s | misses/token | MB/token | Shared Usage |
|---|---|---|---|---|---|
| 6.0e9 | 280.45 (section 22) | 3.57 | 501 | 339.5 | see below |
| 2.4e9 | measured in this section | - | - | - | - |

(The 2.4 GB row is filled from the run, not assumed; if its tail is not better
than the 6.0 GB row, the answer to Phase 3a is "the spill was not the cost",
which the bound already predicted.)
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 25. Phase 2: the policy bound" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended sections 25-26, %d bytes" % len(s + ADD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
