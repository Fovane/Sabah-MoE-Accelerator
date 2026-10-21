"""Append section 27, the long-context brief's acceptance summary."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 27. Long-context brief: what was verified, what was not

### Verified

| item | result |
|---|---|
| state save/load covers this memory type | yes. `llama-memory-hybrid-idx.cpp:221-241` (attn KV + recurrent + indexer sinfos), `llama-memory-recurrent.cpp:767/898/993/1089`. 904,966,348 bytes (863 MiB) at ~30K KV |
| restore is exact | yes. 8 greedy steps from the restored state: logits + argmax byte-identical (with the row offset the missing prefill row forces), fed identical one-for-one |
| test bed iteration cost | 404 ms (state load) instead of 1088 s (prefill) |
| deep-context routing trace | 2,949,120 accesses / 2048 steps / 1440 per step = 48 x 10 x 3, captured with the diag's MMID path |
| replay model faithfulness | LRU at 6 GB predicts 466 misses/token, the runtime measures 501 (section 22) - within 8% |
| slab size does not change the output | byte-identical logits/argmax/fed at 2.4 GB and 6.0 GB |
| short context is unharmed | 30.36 ms/token, byte-identical, section 16.6 - untouched by any of this |

### The tier answer

| tier | needed hit rate | Belady at the spill-free 2.4 GB slab | verdict |
|---|---|---|---|
| A (10 tok/s) | 88.0% | **68.02%** | **not reachable** |
| B (20 tok/s) | 94.0% | 68.02% | not reachable |
| C (30 tok/s) | 96.0% | 68.02% | not reachable |

The offline optimum at the only spill-free slab misses tier A by 20 points, so
the gap is in the trace (the distinct experts a ~30K-KV token needs), not in the
replacement policy. A better policy buys at most 466 -> 227 misses at today's
slab, which is ~130 ms of disk against a 100 ms budget.

### Not measured, and why

| | |
|---|---|
| 2.4 GB with valid clocks | the pin was refused (permission lost mid-session); its time is not evidence, its bytes are (840 misses / 570.6 MB per token) |
| 64K / 128K prefill and decode | not started: the 32K prefill is 1174 s and the projection to 128K is 90 min, over the brief's threshold; the state save means the decode half alone is cheap but the prefill half is not |
| a replacement policy behind an env var | not implemented: the bound above says it cannot reach a tier, so the 4 h box was not spent on it |
| `--ctx` presets and the DXGI slab auto-sizing | not implemented |
| MTP, device-side residency | explicitly out of scope |

### What this means for the headline

The README's 30-33 tok/s stays a **short-prompt, warm-tier** number, as it
already says in section 22. Nothing measured here supports a deep-context
headline, and the bound says no policy can produce one on this hardware.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 27. Long-context brief" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 27")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
