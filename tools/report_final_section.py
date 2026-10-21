"""Append section 24 (the whole-brief acceptance summary) to REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 24. Acceptance summary for the whole brief

Everything below is measured on this machine (driver 32.0.16.1742 / nvidia-smi
617.42, CUDA 12.8, `compute_120a/sm_120a`, MSVC 14.44, RTX 5060 Ti 16283 MiB,
Ryzen 5 7600), with `-lgc 2800,3090 -lmc 13000,14001` in force for every timed
run. A run whose steady SM clock is below 2400 MHz is not evidence.

| # | brief item | state | evidence |
|---|---|---|---|
| 1.1 | fusion decisions, SEG=0 vs SEG=48 | **VERIFIED** | 15/15 split blocks identical, `tools/p1_fuse.py` |
| 1.2 | property-check skip cannot miss a pointer/shape change | **VERIFIED + tightened** | `ggml-cuda.cu:2784-2796` quoted; `sabah-seg.cpp:167-171` now records every node's own struct, the src tuple still zeroed only for non-COMPUTE nodes |
| 1.3 | byte-identical fed/argmax/logits, SEG=48 vs SEG=0 | **PARTIAL** | `-c 2048` 640 decodes across n_kv 256/512: `c30f01a8d676d64f` both sides. `-c 131072` 1024 tokens: one prompt byte-identical; the second prompt aborts on **both** sides at ~285 s with "over-budget expert: out of memory" - a runtime budget limit, not SEG. `-c 512` for a 640-token run is impossible (any prompt + 640 > n_ctx) |
| 1.4 | 128K chat 2048 steps, tail | **N=1 ONLY** | 42.74 -> 30.36 ms/token, and section 22 shows that number is a ~3.7K-deep KV |
| 1.5 | commit + patch + docs + launcher | **DONE** | `phase-e-item4-graphs` (`ec5ec9b`), patches 0003/0004, `docs/CUDA_GRAPH_SPLIT.md` 4d, `sabah gate` PASS, launcher states the shipped profile with an escape hatch |
| 2.1 | `-ub` sweep, per-setting report | **DONE** | section 20: four settings, disk halves per doubling, `-ub >= 2048` is not byte-identical so it stays opt-in |
| 2.2 | smaller slab plus larger `-ub` | **MEASURED** | section 23.1/23.2: a bigger slab halves tail misses but lengthens warm-up 90% and prefill 4.6x, and 9.8e9 leaves 112 MiB of host RAM |
| 2.3 | one real 32K prefill, then 128K if <= 60 min | **32K DONE, 128K NOT RUN** | section 21: 1174.0 s, 1013.2 GB, 41.3 ms/token; projection to 128K is 4679 GB / 90.3 min, over the threshold |
| 2.4 | decode on the filled context | **DONE, AND IT CHANGES THE HEADLINE** | section 22: 310.0 ms/token, 504 misses/token, not steady at step 2048 |
| 3.1 | slab sized from the DXGI budget; Shared Usage ~0 | **MEASURED, FAILS AT 128K** | section 23.1: +110 MiB at `-c 512`, +349 at `-c 32768`, **+3517 at `-c 131072`**; the budget arithmetic puts the 128K slab at ~2.4 GB, not 6.0 GB |
| 3.2 | prompt-informed warm-up, >= 25% gate | **NOT IMPLEMENTED, WITH THE REASON** | section 23.2: doubling the slab does not shorten warm-up in ms (296 -> 563), so the feature is not shipped |
| 3.3 | coding trace hit rate vs slab | **DONE** | section 23.2: tail misses 408.4 -> 213.2/token, disk 277.1 -> 144.1 MB/token, outputs byte-identical |
| 4 | layer-group prefill, 2 h box | **ROOT CAUSE FOUND, OFF BY DEFAULT** | section 18: the fused-kernel set changes at the split boundary (`ggml-cuda.cu:3610-3614`); H1 (memory replay) disproved |
| 5.2 | size and path hygiene | **DONE** | 0 tracked files > 50 MB; 23 tracked files still name a local path - 1 LICENSE (the author), the rest build scripts that must name the real build directory |
| 5.4 | one command that refuses an invalid machine | **DONE** | `python -m sabah.tools.bench_protocol`; `sabah bench` returns 5... now flags, on other CUDA context holders |
| gate | the contract | **PASS** | verify 2000/0, id reuse 768/0, mapped fallback 0, over-budget 448 documented |

Honest note on the two things this brief found that change the README: the decode
headline is a short-prompt warm-tier number (section 22), and the RAM-flat claim
holds only while the expert tier fits in VRAM (section 23.1). Both are now in
the README as numbers rather than removed.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 24. Acceptance summary" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 24, %d bytes" % len(s + ADD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
