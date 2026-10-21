"""Append section 23 (Phase 3: slab sizing, Shared Usage, warm-up) to REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 23. Phase 3: the slab does not fit next to a deep KV cache

### 23.1 Shared Usage vs context depth (the RAM-flat contract, measured)

The contract is that host RAM stays flat because nothing about expert delivery
depends on it. `tools/p3_shared.py` runs a short decode at each context while
sampling the GPU's Windows "Shared Usage" counter - the part of the video memory
the OS puts in system RAM - and subtracts the idle baseline (43.0 MiB, the
display compositor). `N=1` per context, slab 6.0e9, `-b 2048`, `-ub 2048`:

| context | VRAM peak | host WS peak | host avail min | Shared Usage, idle -> max in decode | delta |
|---|---|---|---|---|---|
| 512 | 11935 MiB | 4867 MiB | 6568 MiB | 43.0 -> 152.7 MiB | **+110 MiB** |
| 32768 | 14451 MiB | 5466 MiB | 5889 MiB | 41.8 -> 390.7 MiB | **+349 MiB** |
| 131072 | 15994 MiB | 8901 MiB | 2502 MiB | 42.4 -> 3559.8 MiB | **+3517 MiB** |

So the contract holds at short context and **fails at 128K**: as the KV cache
grows, the 6.0 GB slab no longer fits beside it in 16.3 GB of VRAM, and WDDM
puts ~3.4 GB of the slab in system RAM. The host working set grows with it
(4867 -> 8901 MiB). This is the same 3.3 GB cliff `phase0b-vram` recorded; here
it is quantified against the counter rather than inferred.

What the slab should be, from the budget: at `-c 131072` the non-slab part of
the peak is 15994 - (5722 - 3517) = 13789 MiB, so a slab that fits is
16283 - 13789 - margin, i.e. about **2.4 GB**, not 6.0 GB. That is the sizing
rule the brief asks for (DXGI budget, minus KV, compute buffer, dense weights,
safety margin) and it is **not** what the current default does. The cost of
applying it is the hit rate, and 23.2 shows the hit rate is already the wall.

### 23.2 The coding trace: hit rate vs slab size

Same trace, same configuration as item 6 (`-c 131072`, `-n 2048`, `-b 2048`,
`-ub 512`, `-t 8`), only the slab changed. Outputs are **byte-identical** between
the two slabs, so this is a resource question, not a numerics one:

| | slab 6.0e9 | slab 9.8e9 |
|---|---|---|
| prefill | 77.1 s | 353.7 s |
| warm-up (first 128) ms/token | 296.2 | 563.5 |
| warm-up misses/token | 479.4 | 286.9 |
| warm-up MB/token | 326.1 | 194.2 |
| tail (last 256) ms/token | 255.3 | 244.3 |
| tail misses/token | 408.4 | **213.2** |
| tail MB/token | 277.1 | **144.1** |
| host available min | 2080 MiB | 112 MiB |

More slab halves the misses and the disk per token (-48% on both) and moves the
tail 4%, but it makes the **warm-up 90% slower** (296 -> 563 ms) and the prefill
4.6x slower, because the larger tier costs more to fill and to keep bookkeeping
on. The 9.8e9 run also leaves 112 MiB of host RAM free.

So the answer to the brief's warm-up question is a measurement, not an
implementation: **a prompt-informed warm-up would have to beat a baseline where
simply giving the tier more room does not shorten the warm-up in ms, it
lengthens it.** Not implemented, and the trend says why. The coding trace's
failure to warm up in 2048 steps is also not a slab-size problem: at 9.8e9 the
tail still misses 213 experts/token, i.e. the trace's per-token expert union
(373 misses/token at 6.0e9) does not stabilise at all. REPORT 15.4 stands.

### 23.3 The reproduction check

`p3-code-6000000000-seg48` reproduces `item6c-code-2048` on the same prompt:
tail misses/token 408.4 vs 408.4, MB/token 277.1 vs 277.1 (identical), ms/token
255.3 vs 251.12 (-1.7%), prefill 77.1 s vs 76.3 s, disk 78.6 GB both. Clocks
valid on both (steady SM 2827-2850 MHz). The item-6 numbers are therefore
reproducible on this machine, and the remaining difference is the clock/machine
state, not the artefact.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 23. Phase 3" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 23, %d bytes" % len(s + ADD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
