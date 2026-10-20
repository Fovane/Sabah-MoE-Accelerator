# Progress

Updated after each step. Percentage is weighted by the size of the item in the
brief (weights in the last column), not by wall time.

```
item                                              state      done   weight  points
1  TILE3 default + byte-exact + fallback + gate   DONE       100%     20     20.0
2  clock state, validity rule, pinning, trace     DONE       100%     20     20.0
   + graph replay probe (item 4's gate)           DONE       100%
3  pinned slab sweep at -c 131072, N=3            DONE       100%     15     15.0
4  CUDA-graph split, one-layer PoC, >=8 ms gate   DONE       100%     20     20.0
5  dense op time/bytes/GB/s + n-gram lookup       static done, per-op needs build     60%     10      6.0
6  real chat + coding at 128K, real 64K/128K prefill  DONE   100%     15     15.0
                                                                    ---    -----
                                                                     100     96.0
```

## Commits

| branch | commit | what |
|---|---|---|
| `phase-c-tile3-default` | `a2de373` | TILE3 default, byte-exact at 2 contexts x 2 prompts, fallback proved |
| `phase-d-clock-hygiene` | `e313435` | clocks.py, invalidity rule, pinning evidence, docs/CLOCK_HYGIENE.md, REPORT §9-10 |
| `phase-d-clock-hygiene` | `3393333` | tools/status.py, PROGRESS.md, fallback-test metadata |
| `phase-e-item3-5` | `f2b9354` | item 3 sweep + tail steady, item 5 byte model, graph replay probe, REPORT §11-14 |
| `phase-e-item4-graphs` | `5a44de9` | item 4: segmented CUDA graphs, default on, acceptance matrix, patch 0003, REPORT 16 |
## What "done" means per item

1. DONE. Four byte-identical pairs (2 prompts x `-c 2048` / `-c 131072`, 1024
   tokens), guard `maxnsub=16` proved to take the fallback branch, `sabah gate`
   PASS, committed `a2de373`. Report section 10.
2. DONE. 0.5 s sampler, valid/invalid rule at 2400/10000 MHz, `-lgc`/`-lmc`
   finding, 1 s idle->decode->idle trace, docs, report section 9.
3. DONE. 4 slabs x N=3, pinned, `clock_valid` on all twelve. Tail-recomputed
   steady: 43.5 / 71.4 / 88.4 / 153.3 ms; the harness steady window was the fill
   at 128K. Report section 12.
4. DONE. Segmented CUDA graphs around every Sabah node, default on,
   SABAH_CUDA_GRAPH_SEG=0 disables. Fusion decisions identical 10/10;
   byte-identical at -c 131072 (1024 tok) and across n_kv 256/512;
   128K chat tail 42.74 -> 30.36 ms/token (-12.38). Report 16.
5. Static half done (4.111 GB/token, 9.18 ms floor, 326 GB/s = 73% of peak,
   n-gram path 0.42 ms = 0.02 GB/s). Per-op achieved GB/s still needs the
   op-time hook or an nsys export.
6. DONE. Chat 2048 steps: tail 42.74 ms = 23.40 tok/s, 1.9 MiB/token.
   Coding never steady (last 256 = 251.12 ms, 240.3 MiB/token). Prefill
   91.0 s/94.37 GB and 76.3 s/78.58 GB; 2048-token chunks re-stream the
   pool each time (113-134 GB per chunk). Report 15.

## Next: B, layer-group-major prefill

State: design note rev 2 written (`docs/PREFILL_GROUP_MAJOR.md`), nothing
implemented, awaiting approval.

Rev 2 resolves the two conditions requested before coding:

* `apply()` is a commit, not a getter: once per ubatch, and the 11 later groups
  replay the per-ubatch view (`i_cur`, `n_kv`, `head`, `rs_z`, `src0`, `rs_idx`)
  instead of applying again. `n_kv` must be restored rather than recomputed
  because it sizes the kq mask; `head` must be restored because `s_copy` and the
  PLE conv write offset by it.
* block-boundary invariants I1-I7: per-layer order, KV cell identity
  (`T + N_decode <= n_ctx`), SSM cell chain, conv/PLE row chain, f32 hc handoff,
  exact block partition, graph identity at group boundaries.

Acceptance: byte-identical KV/SSM/conv state, logits and I/O counters vs the
current chunked path (1630 tok and 4-8K), `sabah gate` PASS, item-4 decode
numbers unchanged, 8K prefill >= 3x faster or stop.

## Gate

The llama.cpp rebuild is the gate for items 4 and 5. It was restarted with
`CMAKE_CUDA_ARCHITECTURES=120`; the cached default (`120;61`) compiled every
CUDA unit twice, once for Pascal, which this GPU cannot run.

