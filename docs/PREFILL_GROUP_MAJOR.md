# Layer-group-major prefill: design note

Status: proposal, nothing implemented. Pinned llama.cpp `96ffdc4`. All model
numbers below are read from the GGUF, not assumed.

## 1. Why

Measured (REPORT 15.2): chunked prefill re-streams the whole expert pool per
2048-token chunk.

| chunk | ms | GB read | MB/token |
|---|---|---|---|
| 0 | 117985.9 | 113.17 | 57.6 |
| 5 | 132512.7 | 134.04 | 64.7 |

At ~0.96 GB/s that is 63-71 min for 64K and 126-142 min for 128K: 64 chunks x
~120 GB = **~7.7 TB** to prefill one 128K prompt. The experts are 57 GB of the
83.6 GB model (per layer: `ffn_down_exps` IQ4_NL 472 MB, `ffn_gate_exps` and
`ffn_up_exps` IQ3_S 360 MB each = 1.19 GB, 512 experts); the rest is dense and
stays resident in VRAM (3983 MiB CUDA0 model buffer).

Goal: each expert read **once per prompt**, i.e. <= 57 GB (expected 40-50 GB
when only the routed union is read), independent of the number of chunks.

## 2. Scheme

Group the 48 layers into G groups. For group g, run **all** prompt chunks
through the layers of g, then advance to g+1.

```
for g in groups:                 # outer: expert working set
    pin group g experts          # <= G * 1.19 GB, must fit the slab
    for c in chunks:             # inner: tokens, in order
        h[c] = layers[l0(g)..l1(g))(h[c])
```

Slab fit: G=4 gives 4.76 GB of experts and leaves 1.24 GB of the 6.0e9 slab for
routing jitter. G=5 is 5.95 GB and leaves no headroom. 12 groups at G=4.

Cross-group state is **not** one `[n_embd, T]` tensor: this model keeps
hyper-connection residual streams `[n_embd, hc=4, T]`
(`src/models/qwen4exp.cpp:265`, `hparams.dsv4_hc_mult=4`). At f32 that is
40,960 B/token:

| prompt | f32 | f16 |
|---|---|---|
| 1630 tok | 66.8 MB | 33.4 MB |
| 8K tok | 335 MB | 168 MB |
| 32K tok | 1.34 GB | 0.67 GB |
| 128K tok | **5.37 GB** | 2.68 GB |

f16 is not acceptable for the equivalence test (it rounds the residual stream),
so f32 is the reference. 5.37 GB does not fit next to a 8.9 GB process working
set (10.5 GB free at start, 2.8 GB minimum during a run). Therefore:

* `T <= 32768`: one block, full group-major, 1.34 GB, disk <= 57 GB.
* `T > 32768`: tile the prompt into 32K blocks; each block is group-major. RAM
  stays 1.34 GB, disk becomes `ceil(T/32768) * 57 GB` = 228 GB at 128K. That is
  still ~34x less than 7.7 TB. (Alternative: spill the hc streams with direct
  I/O, 5.37 GB write + 10 x 5.37 GB read = 59 GB of extra I/O; not better.)

PLE (`per_layer_token_embd`, 160 x 320001536, IQ4_NL, runs on CPU via `-ot`) is
gathered **once per prompt** into `[160, T]` (0.8 MB at 128K) and reused by
every group; the gather ids and the per-layer embedding are inputs to each
group graph.

## 3. llama.cpp integration (pinned `96ffdc4`)

Current path, for reference:

| what | where |
|---|---|
| decode: `init_batch`, `get_ubatch`, `process_ubatch` | `src/llama-context.cpp:1749-1829` |
| `apply()` then graph params then compute | `src/llama-context.cpp:1336-1396` |
| memory interface | `src/llama-memory.h:88` (`init_batch`), `:60` (`apply`) |
| KV apply (allocates cells, `seq_add`) | `src/llama-kv-cache.cpp:2715`, `:1155` |
| KV views per layer | `src/llama-kv-cache.cpp:1266` `get_k`, `:1286` `get_v` |
| recurrent/SSM apply | `src/llama-memory-recurrent.cpp:1252`, head at `:149` |
| hybrid memory (KV + recurrent) | `src/llama-memory-hybrid.cpp:67`, `:253` |
| model layer loop | `src/models/qwen4exp.cpp:377-419` |
| hc residual streams | `src/models/qwen4exp.cpp:265`, `:385`, `:412`, `:429` |
| PLE gather | `src/models/qwen4exp.cpp:366`, `381` |
| graph params struct | `src/llama-graph.h:772` |
| graph types | `src/llama-graph.h:37` |

Four additions, all prefill-only:

1. `llm_graph_params` gets `int il0, il1; ggml_tensor * inp_hidden;
   ggml_tensor * out_hidden; ggml_tensor * ple_embd;` and a new
   `LLM_GRAPH_TYPE_PREFILL_GROUP` (`llama-graph.h:37`, `:772`).
2. `src/models/qwen4exp.cpp`: a new builder that loops `il in [il0, il1)`,
   starts from `inp_hidden` instead of `token_embd`, writes `out_hidden`
   (f32, same layout as the hc streams), and includes `output_norm` +
   `output` only when `il1 == n_layer`. Embedding and PLE only when
   `il0 == 0`. The existing `build_qwen4exp` is untouched, so decode graph
   building does not change.
3. `llama_context`: a new `prefill_group_ubatch(ubatch, mctx, g, first, last)`
   modelled on `process_ubatch` but without output bookkeeping; the decode
   loop stays as is.
4. Memory: `init_batch` runs **once per chunk**; `mctx->apply()` runs **only
   for the first group** of that chunk. Later groups reuse the same
   `mctx`/ubatch, so `get_k`/`get_v` return the same cells
   (`llama-kv-cache.cpp:2753` uses `sinfos[i_cur]`, fixed at `init_batch`) and
   the recurrent state views are the same. This is the one semantic change:
   `apply()` must not allocate cells twice for the same positions.

## 4. Why the reordering is still correct

* Full-attention layer L in group g: chunk c reads the KV written by chunks
  `< c` **of layer L**. The chunk loop is inner and ascending inside g, and all
  chunks of g run before g+1, so layer L sees its tokens in order and its KV is
  complete before the next group starts.
* SSM/recurrent layer L: state after chunk c depends only on chunks <= c of L,
  processed in order. `full_attention_interval=4` means 36 SSM layers and 12
  attention layers, interleaved; both are inside some group and each sees the
  same ordered chunk sequence.
* Positions/KV cells are a property of (sequence, token), not of the group: the
  group loop changes only *which layers* consume them in a pass.
* Each group graph sees, for a given chunk, exactly the inputs the full graph
  would have at that layer boundary (same hc streams, same PLE, same positions),
  and applies the same layers in the same order. Hence bit-identical
  (see section 6).

## 5. Slab interaction and disk accounting

* At a group boundary, pin the new group's experts and drop the previous
  group's: with G=4 the working set is disjoint from the old one, so a plain
  LRU thrashes for the first chunk of every group unless the eviction is
  explicit. Proposal: an explicit "group pin" call that marks the group's 4x512
  expert entries as `pinned` and evicts non-pinned entries first.
* Within a group, chunk 1..n-1 must be pure hits. Measured gate: misses/token
  in chunks >= 1 of a group must be ~0 (the same tail-window counter as item 6,
  reported per chunk).
* Disk per group <= 4 x 1.19 GB = 4.76 GB; total <= 57 GB, expected 40-50 GB
  (union of routed experts). Every byte read is accounted by
  `sabah_rt_get_metrics`, so the acceptance run reports bytes/prompt directly.

## 6. Equivalence (acceptance test)

Per chunk and per layer the shapes, kernels and inputs are the same as the
current chunked path; the only difference is the order of graph execution.
Therefore, with f32 hidden-state buffers:

* final KV cache, SSM/conv state and logits must be **byte-identical** to the
  current chunked prefill on a 1630-token prompt and on a 4-8K prompt;
* compare `logits.f32`, `argmax.i32` and the KV/state dumps from `sabah-diag`
  (`SABAH_DIAG_NAMES`), not just the sampled ids.

If any byte differs, the design is wrong; there is no tolerance.

## 7. Cost model (measured basis: 0.96 GB/s, 120 GB per 2048-token chunk)

| prompt | today | group-major (G=4) | RAM |
|---|---|---|---|
| 1630 tok | 91 s, 94 GB | ~50-60 s, <= 57 GB (one chunk: no win expected) | 67 MB |
| 8K tok | ~8 min, 480 GB | ~1 min, <= 57 GB | 335 MB |
| 128K tok | ~130 min, 7.7 TB | ~4 min, 228 GB (4 x 32K blocks) | 1.34 GB |

Stop rule (user): if 8K prefill is not at least 3x faster than chunked after
implementation, stop and report.

## 8. Risks

* `apply()` must be idempotent per chunk (section 3.4). If the KV allocator is
  not deterministic for a second call, the cell map must be captured on the
  first call and replayed.
* hc streams at 128K need the 32K tiling; the tiling cost is 4x the expert
  streaming and must be measured, not assumed.
* 12x more graph builds for 128K (768 vs 64). Host cost per build is unknown;
  it is in the same order as the current per-chunk build (a few tens of ms), so
  768 x 40 ms ~ 30 s, acceptable against a 4 min target.
* The `indexer` top-k (`attention.indexer.top_k=2048`) is per layer and per
  chunk; it is inside the layer, so the reordering does not affect it.

## 9. What is not in scope

Device-side residency, prefetch, MTP: not now.
