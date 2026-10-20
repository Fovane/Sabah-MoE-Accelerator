# Progress

Updated after each step. Percentage is weighted by the size of the item in the
brief (weights in the last column), not by wall time.

```
item                                              state      done   weight  points
1  TILE3 default + byte-exact + fallback + gate   DONE       100%     20     20.0
2  clock state, validity rule, pinning, trace     commit     90%     20     18.0
3  pinned slab sweep at -c 131072, N=3            tooling    15%     15      2.3
4  CUDA-graph split, one-layer PoC, >=8 ms gate   code only  30%     20      6.0
5  dense op time/bytes/GB/s + n-gram lookup       code only  25%     10      2.5
6  real chat + coding at 128K, real 64K/128K prefill  not started 5%  15      0.8
                                                                   ---    -----
                                                                   100     49.6
```

## What "done" means per item

1. DONE. Four byte-identical pairs (2 prompts x `-c 2048` / `-c 131072`, 1024
   tokens), guard `maxnsub=16` proved to take the fallback branch, `sabah gate`
   PASS, committed `a2de373`. Report section 10 written.
2. Code and measurements are finished (0.5 s sampler, valid/invalid rule at
   2400/10000 MHz, `-lgc`/`-lmc` finding, 1 s idle->decode->idle trace,
   `docs/CLOCK_HYGIENE.md`, report section 9). Only the commit is missing.
3. Tooling finished (`tools/slab_sweep.py` with clock guard and
   Dedicated/Shared sampling). Two 6.0e9 anchor points measured at 128K
   (prompt A 176.0 ms, prompt B 47.4 ms). The 4-slab x N=3 sweep has not run.
4. `ggml-cuda.cu` split-graph implementation is written but not yet compiled
   or measured. Build is the gate for both 4 and 5.
5. Per-node CUDA-event hook (`GGML_CUDA_OP_TIMES`) written, not compiled. No
   bytes/GB/s numbers yet.
6. Needs a real multi-turn and a real coding prompt plus a 64K and a 128K
   prefill; nothing measured yet.

## Gate

The llama.cpp rebuild (item 4 + 5) is a single shared gate: if it compiles, both
items proceed; if the split does not project 8 ms/token, item 4 stops at the PoC
by design.
