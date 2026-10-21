# Layer-group-major prefill: design note

Status: implemented behind `SABAH_PREFILL_GROUP` (default 0); the equivalence
gate is met only for the single-ubatch, G=4 case -- see section 6.1 for the
measured envelope and the fallback that is in place. Pinned llama.cpp `96ffdc4`.
All model numbers below are read from the GGUF, not assumed.

Rev 2 adds the two conditions requested before coding:

* section 3.3/3.4: source audit of `apply()` and of memory-context reuse;
* section 4: the state invariants at a 32K block boundary.

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
through the layers of g, then advance to g+1. Three nested loops, and the
nesting order is load-bearing:

```
for block b:                     # <= 32768 tokens: bounds host-RAM hc state
    for group g:                 # outer: expert working set
        pin group g experts      # <= G * 1.19 GB, must fit the slab
        for ubatch u in b:       # inner: tokens, in order
            h[u] = layers[l0(g)..l1(g))(h[u])
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
**recomputed per group graph** from the same ubatch instead of being cached:
`build_inp_ple` (`qwen4exp.cpp:1171-1190`) is a `ggml_get_rows` on an id input,
identical bits every time, and the read is ~90 B/token/group (138 MB over a
whole 128K prompt). No host buffer, no extra graph parameter.

## 3. llama.cpp integration (pinned `96ffdc4`)

### 3.1 Current path, for reference

| what | where |
|---|---|
| `init_batch` once per decode, `next()` per ubatch | `src/llama-context.cpp:1752`, `:1983` |
| `apply()` then graph params then compute | `src/llama-context.cpp:1336-1341` |
| memory interface | `src/llama-memory.h:88` (`init_batch`), `:60` (`apply`) |
| KV apply (allocates cells, `seq_add`) | `src/llama-kv-cache.cpp:2725`, `:1097` |
| KV views per layer | `:2753` `get_k`, `:2757` `get_v` |
| recurrent/SSM apply | `src/llama-memory-recurrent.cpp:1252`, `find_slot` `:505` |
| hybrid memory (KV + recurrent) | `src/llama-memory-hybrid.cpp:67`, `:253` |
| model layer loop | `src/models/qwen4exp.cpp:377-426` |
| hc residual streams | `src/models/qwen4exp.cpp:265`, `:372`, `:410`, `:422` |
| PLE gather | `src/models/qwen4exp.cpp:366`, `:381` |
| graph params / graph types | `src/llama-graph.h:772`, `:37` |

### 3.2 Four additions, all prefill-only

1. `llm_graph_params` gets `int il0, il1; ggml_tensor * inp_hidden;
   ggml_tensor * out_hidden;` and a new `LLM_GRAPH_TYPE_PREFILL_GROUP`
   (`llama-graph.h:37`, `:772`).
2. `src/models/qwen4exp.cpp`: the existing builder is parameterised by
   `[il0, il1)`; `res_hc` starts from `inp_hidden` (`il0 > 0`) or from
   `build_inp_embd` (`il0 == 0`); embedding + PLE only for `il0 == 0`; output
   norm + `output` only for `il1 == n_layer`. The decode path keeps calling it
   with `[0, n_layer)`, so it is untouched.
3. `llama_context`: a new `prefill_group_ubatch(ubatch, mctx, g)` modelled on
   `process_ubatch` without the output bookkeeping; the decode loop stays as is.
4. Memory: `apply()` once per ubatch (group 0) plus a per-ubatch replay for
   groups 1..11, as specified in 3.3/3.4.

### 3.3 `apply()` is a commit, not a getter (source audit)

`init_batch` runs **once per `llama_decode`** (`llama-context.cpp:1752`), and
`do { ... } while (mctx->next())` walks the ubatches (`:1983`). `process_ubatch`
calls `mctx->apply()` before building the graph (`:1337`). So `apply()` is the
only memory mutation point, once per ubatch.

What it mutates:

| module | chain | effect |
|---|---|---|
| KV | `apply()` `:2725` -> `apply_ubatch` `:1097` | writes pos/token/seq into the cells named by `sinfos[i_cur]`; a cell that is reused runs the purge path; then `n_kv = get_n_kv(sinfos[i_cur])` `:2726` |
| recurrent | `:1252` -> `find_slot` `:505` | writes `cells[].pos/seq_id/tail/src0`, sets `head = min`, `n = max-min+1`, `used` (`:665-710`), `rs_z` |
| hybrid | `llama-memory-hybrid.cpp:253` | forwards both, and `next()` advances both sub-contexts (`:240`) |

`sinfos[u]` is produced by `prepare()` (`:751-815`), which runs the allocation
for *all* ubatches of the call and then rolls `v_cells`/`v_heads` back
(`:797-808`). So `sinfos[u]` is only valid if every earlier ubatch of that call
was committed, in order.

Therefore `apply()` is **not re-entrant for the same ubatch**:

* KV: a second `apply_ubatch` for the same positions finds the cells occupied
  and takes the reuse/purge path (`:1097`) -> different cell map.
* recurrent: `find_slot` for the same ubatch logs "non-consecutive token
  position" (`:659-664` -- a `LLAMA_LOG_WARN`, not an abort) and advances
  `head` again -> the SSM and conv cell chain shifts silently.

Conclusion: `apply()` exactly once per ubatch, at group 0; groups 1..11 must
*replay the view* of that ubatch. Nothing else in the memory module needs to
change.

The graph reads memory only through this list (verified complete):

| read | site | depends on |
|---|---|---|
| `get_k` / `get_v` | `:2753`, `:2757` | `n_kv` + `sinfos[i_cur].s0/s1`, pure views |
| `cpy_k` / `cpy_v` (cell write) | `:2761`, `:2765` | `sinfos[i_cur]` via the idxs input (`:2761`) |
| `get_n_kv` | `:2741` -> `:1250` | `v_cells[].used_max_p1()`, **grows** as later ubatches commit |
| `get_prev_tokens` (PLE n-grams) | `:2813` -> `:1836` | committed cells looked up by position; positions before the ubatch are committed in group 0 and never change |
| `get_head`, `get_rs_z`, `s_copy` | `recurrent.cpp:1282`, `:1286`, `:1306` | `mem->head`, `mem->rs_z`, `cells[i + head].src0` |
| `get_r_l` / `s_l` / `p_l` | `:1294-1304` | whole tensors, no index |
| conv state write | `qwen4exp.cpp:1126`, `:1160-1163` | dst row `(slot*mem_size + kv_head)*row_size`, `kv_head = get_head()` |

### 3.4 What a replay must restore, and why

Two readings are position-dependent and change after group 0 commits the block:

1. **`n_kv` must be restored, not recomputed.** After group 0 the whole block is
   committed, so `get_n_kv()` returns end-of-block. `n_kv` sizes
   `attn_inp_kq_mask` (`llama-graph.cpp:34-41`; reuse check `:59`) -> a wider
   mask changes the FA tile/reduction width and can change the low bits, and is
   strictly more work. Snapshot `n_kv` per ubatch inside `apply()`.
2. **`head` must be restored.** `s_copy` (`:1306-1308`) and the PLE conv write
   (`qwen4exp.cpp:1163`) both offset by `mem->head`, which is end-of-block after
   group 0.

So the memory context gains a replay, not a second apply:

```
apply()   # g == 0: commit, then snapshot {i_cur, n_kv, head, rs_z,
          #           cells[head..head+n).src0, rs_idx}
seek(u)   # g  > 0: i_cur = u; n_kv = snap[u].n_kv; mem->head = snap[u].head;
          #         mem->rs_z = ...; src0 fields = ...; rs_idx = ...
```

* `is_full` is `false` for the ubatch constructor (`recurrent.h:194`; only the
  no-ubatch ctor sets it, `:1233`), so `get_head`/`get_rs_z` need no branch.
* `rs_idx` is reset inside `s_copy` (`:1320`) and only exists when
  `n_rs_seq > 0`; the gate log prints `n_rs_seq`, and the snapshot includes it
  only when nonzero.
* The snapshot is per-ubatch metadata (hundreds of bytes), not KV data and not
  the state tensors.

Each group writes only the layers it owns (`r_l[il]`, `s_l[il]`, `p_l[il]`,
`layers[il].k/v`), so every per-layer write still happens exactly once.

### 3.5 Graph reuse and buffers

`allow_reuse` (`llama-graph.h:815`) compares ubatch shape, `n_outputs` and
samplers -- not `gtype`/`mctx` -- so `il0/il1` join that comparison; otherwise a
group graph could be reused for the next group. With the replay in place the
input-side checks agree with group 0 (`inp_rs::can_reuse` compares
`head`/`rs_z`, `llama-graph.cpp:357`; kq mask compares `ne[0] == n_kv`, `:59`).

`sched_reserve()` sizes buffers for `n_tokens = min(n_ctx, n_ubatch)`
(`llama-context.cpp:596`). A group graph adds the f32 hc input tensor, so
reserve must also build one `PREFILL_GROUP` graph at that size before the first
group runs.

## 4. Invariants at a 32K block boundary

Notation: `T` prompt tokens, `p in [0,T)`; blocks `B_k = [32768k, 32768(k+1))`;
`u` = the ubatches of a block (the `init_batch` split, contiguous, in order,
`n_ubatch <= 512`); groups `G_j = layers [4j, 4j+4)`; `c(p)` = the KV/recurrent
cell of position `p`; `head(u)` = `mem->head` at u's apply.

**I1 -- per-layer token order.** Every layer `L` executes the ubatches of the
prompt in ascending token order, each exactly once. *Holds:* block loop is outer
and ascending, `L` belongs to exactly one group, and the ubatch loop is
innermost and ascending; no ubatch is visited twice for `L`.

**I2 -- KV cell identity.** `c(p)` is assigned exactly once, by `apply()` of the
ubatch containing `p` in the group-0 pass, and is not reassigned later. *Holds:*
`apply` runs only for group 0, and the allocator never touches a live cell while
`used_max_p1 <= n_ctx`. *Constraint:* **`T + N_decode <= n_ctx`**, otherwise the
cache evicts positions that later tokens still attend to. This is already true
of the chunked path; B adds no requirement, but the 128K test must leave a decode
budget (section 6).

**I3 -- SSM state chain.** For layer `L` and ubatch `u`, token `i` maps to cell
`head(u)+i` (`recurrent.cpp:1307`); the graph reads `cells[head(u)+i].src0` and
writes the new state into `r_l/s_l[il]` at that cell. *Holds across a block
boundary because* `head` lives in the memory object (it survives `init_batch`),
advances once per ubatch in global order, and the replay restores exactly the
group-0 value for every group, so consecutive ubatches chain regardless of the
group or block they belong to.

**I4 -- conv / PLE state chain.** The PLE conv history of layer `L` at ubatch `u`
is the row at `(slot*mem_size + head(u))*row_size` of `p_l[il]`
(`qwen4exp.cpp:1160-1163`), prepended with `hist = (kern-1)*dil` columns
(`:1235-1238`); the new tail is written back to the same row at the end of the
ubatch. Because the history is prepended, an ubatch boundary is bit-exact to a
single-shot prefill (`:1235`), and a block boundary is just another ubatch
boundary. *Requirement:* a block boundary must coincide with an ubatch boundary
-- true by construction, a block is a whole `llama_decode` call and `init_batch`
never splits across calls.

**I5 -- hc handoff.** `res_hc` `[2560, 4]` f32 (40,960 B/token) is **per-token,
not per-sequence**: `res_hc` at layer 0 is a repeat of that token's embedding
and every `build_hc_combine` is elementwise, so a token's hc never depends on
another token's hc. There is therefore **no cross-block hc handoff**: the handoff
is cross-*group*, inside one `llama_decode` call. Group 0 taps `res_hc`
(`qwen4exp.cpp:422`, before the out-ids gather) into a device buffer, copies it
to host, and group 1 reads it back as `inp_hidden`: same type, same layout, a bit
copy, and only one ubatch's worth (40,960 B x ubatch tokens, e.g. 20.5 MB at
512). That is why `T > 32768` tiles for a different reason (the group loop cannot
span calls) and not because of hc size alone.

Measured consequence (section 6.1): this handoff is not the failure mode. The
device round-trip was dumped and is bit-exact per ubatch, and `hc_combine_L3`
(end of group 0) is byte-identical to the reference for all four ubatches.

## 6.2 Root cause of the multi-ubatch divergence (found, October 21)

Time-boxed investigation, one hypothesis at a time. Each row is one measured
test, 1888-token prompt unless stated, `-ub 2048` = one ubatch, `-ub 512` = 2/4
ubatches, both sides on the same binary.

| hypothesis | test | result |
|---|---|---|
| segmented CUDA graph recapture | SABAH_CUDA_GRAPH_SEG=0 both sides | DIFF, ruled out |
| op offload by batch size | `GGML_OP_OFFLOAD_MIN_BATCH=1000000` both sides | DIFF, ruled out |
| graph reuse across ubatches | `LLAMA_GRAPH_REUSE_DISABLE=1` | DIFF, ruled out |
| stock CUDA graph path | `GGML_CUDA_DISABLE_GRAPHS=1` | DIFF, ruled out |
| CPU-thread count | `-t 1` | run failed for another reason, ruled out |
| H1: recurrent-memory replay (`rs_z`, `head`, `s_copy/src`, cell pos) | capture `state_predelta`/`conv_state_at` at layers 24/25/44/45/46, one ubatch | **identical**, ruled out |
| H2: state from a second diag run | n/a, single ubatch already diverges at G=24 | superseded |
| H3: `n_rs_seq > 0` snapshots | n_rs_seq is 0 in every run here | not applicable |
| **fusion decisions** | `GGML_CUDA_DISABLE_FUSION=1` both sides | G=24 **IDENTICAL**, G=12 IDENTICAL, G=4 IDENTICAL |

**Root cause: the layer-group split changes which fusion patterns the CUDA
backend applies, and a fused kernel does not round like the unfused sequence.**

`ggml_cuda_try_fuse` refuses any pattern whose 8-node window contains a
Sabah-routed `MUL_MAT_ID` (`ggml/src/ggml-cuda/ggml-cuda.cu:3610-3614`):

```
3610:    for (int j = i; j < cgraph->n_nodes && j < i + 8; ++j) {
3611:        if (ggml_cuda_sabah_owns_node(cgraph->nodes[j])) {
3612:            return 0;
3613:        }
3614:    }
```

A group graph is a **contiguous subsequence** of the plain chunk graph, so a
pattern that is whole in the chunk graph straddles a group boundary and is
refused inside each group graph. The node-level bisect showed the first
differing value at layer 20's `hc_norm` (a `RESHAPE` of the
`rms_norm`+`mul` pair) at token 3, with a byte-identical input tensor and the
same backend and op reported by the capture (`buf CUDA0`, `op RESHAPE`), i.e.
the two nodes of that pair were fused in one path and not in the other.

Why the other G values looked fine is positional luck: G=12 and G=4 put the
boundaries where no pattern straddles, so their decisions coincide with the
chunk graph's. G=24 puts a boundary inside one.

This is the same risk class Phase 1.1's fusion trace exists to catch, and the
`SABAH_FUSE_TRACE` + `tools/fuse_diff.py` check compares *within* one graph
shape, so it cannot see a boundary move (section 6.3).

Consequences, all measured:

* **the feature cannot ship as byte-identical while fusion is default-on**,
  because the plain path it must match keeps fusion on;
* a "run the group path with fusion off" pairing is byte-identical to a
  fusion-off reference, but not to the shipping default;
* `G=2` (24 groups) does not reach the numerics at all: it aborts with
  `GGML_ASSERT(buffer)` in `ggml-backend.cpp:218`, a separate defect that is
  still open.

### State of the feature

Off by default (`SABAH_PREFILL_GROUP` unset = 0), and additionally gated to a
single ubatch per `llama_decode` call. It stays out of the release: the
byte-identical contract is not met for any multi-ubatch setting, and the single
-ubatch setting has no disk-I/O win (one ubatch already streams the experts
once per call).


**I6 -- block partition.** The blocks are disjoint and cover `[0,T)`, and every
token keeps its absolute position (the batch of a block uses the same `pos`
array as the current chunked path), so RoPE, the causal mask and the indexer
top-k see exactly what they see today.

**I7 -- graph identity at group boundaries.** Group 0 builds
`build_inp_embd` + `build_inp_ple` as today. The last group ends with the
unchanged out-ids gather, output norm and `output` (`qwen4exp.cpp:400-408`,
`:428+`) and only adds a tap of `res_hc`; intermediate groups are the same code
with `res_hc` from `inp_hidden`. The retained logit rows therefore come from the
same nodes in the same order -> byte-identical.

## 5. Slab interaction and disk accounting

* At a group boundary, pin the new group's experts and drop the previous
  group's: with G=4 the working set is disjoint from the old one, so a plain LRU
  thrashes for the first chunk of every group unless eviction is explicit.
  Proposal: an explicit "group pin" call that marks the group's 4x512 expert
  entries `pinned` and evicts non-pinned entries first.
* Within a group, chunk 1..n-1 must be pure hits. Measured gate: misses/token in
  chunks >= 1 of a group must be ~0 (same tail-window counter as item 6,
  reported per chunk).
* Disk per group <= 4 x 1.19 GB = 4.76 GB; total <= 57 GB, expected 40-50 GB
  (union of routed experts). Every byte read is accounted by
  `sabah_rt_get_metrics`, so the acceptance run reports bytes/prompt directly.

## 6. Equivalence (acceptance test)

Per layer, per ubatch the shapes, kernels and inputs are the same as the current
chunked path; the only difference is the order of graph execution, and the two
order-dependent readings are restored (3.4). Therefore, with f32 hc buffers:

* final KV cache, SSM/conv state, `logits.f32`, `argmax.i32` and the I/O counters
  must be **byte-identical** to the current chunked prefill on a 1630-token
  prompt and on a 4-8K prompt; compare the state dumps from `sabah-diag`
  (`SABAH_DIAG_NAMES`), not just the sampled ids;
* the run prints `n_rs_seq`, `n_kv` and `head` per ubatch on both sides, so a
  replay that restores the wrong view fails loudly instead of drifting;
* `sabah gate` must PASS, and the item-4 decode numbers must be unchanged
  (section 10).

If any byte differs, the design is wrong; there is no tolerance.

128K test (section 4, I2): prompt 124K tokens, `-c 131072`, 8K decode budget.
This is also the first real deep-KV decode measurement (see section 10).

## 6.1 Measured equivalence status (first implementation)

The gate is **not** met for the multi-ubatch case. What was measured, all with
`SABAH_DIAG_NAMES` captures paired per `(key, occ)` by `tools/cap_cmp.py`
(both sides same `G`, `-b`, `-ub`, prompt, `-n`; SEG as stated):

| case | G | ub | SEG | result |
|---|---|---|---|---|
| 53-token prompt, 1 ubatch | 4 | 512 | 48 | IDENTICAL |
| 1888-token prompt, 1 ubatch | 4 | 2048 | 48 | DIFF (first logits row) |
| 780-token prompt, **2 ubatches** | 24 | 512 | 0 | DIFF |
| 1888-token prompt, **4 ubatches** | 48 (= no grouping) | 512 | 0 | IDENTICAL |
| 1888-token prompt, 4 ubatches | 24 | 512 | 0 | DIFF |
| 1888-token prompt, 4 ubatches | 12 | 512 | 0 | DIFF |
| 1888-token prompt, 4 ubatches | 4 | 512 | 0 | DIFF |
| 1888-token prompt, 4 ubatches | 4 | 512 | 0, `GGML_OP_OFFLOAD_MIN_BATCH=1e6` | DIFF |
| 1888-token prompt, 1 ubatch | 4 | 2048 | 0 | IDENTICAL |
| 1888-token prompt, 1 ubatch | 24 | 2048 | 0 | DIFF |

Two conclusions, both measured, neither inferred:

* the segmented CUDA graph is **not** the cause: the divergence persists with
  `SABAH_CUDA_GRAPH_SEG=0` on both sides (so is the op-offload threshold);
* grouping is exact only for the single-ubatch case **and only at G=4**. Any
  second ubatch in the same `llama_decode` call, and G other than 4 with a single
  ubatch, drift.

Node-level bisect (`tools/b_step4.py`, 4 ubatches, G=4, SEG=0), pairing the
two callback captures per (ubatch, node) in graph order for layer 4
(`is_recr=1`, delta-net, `is_ple=0`; layer type read from the builder):

* end of group 0, `hc_combine/hc_mixed/hc_norm/hc_gate/L3`: **all identical**,
  so the hc carry and group 0 are bit-exact for all four ubatches;
* at layer 4 the first differing real (compute-time) capture is
  `hc_combine_L4` for ubatch 0, with `conv_output_raw_L4` (row 4),
  `q_conv/k_conv_L4` (row 64), `final_output_L4` (row 4) and
  `state_predelta_L4` (occ 2) also differing, while `conv_state_at_L4` (the conv
  state gather, occ 0/1) and `hc_mixed_L4` occ 0/1 are identical. The input hc
  to group 1 is bit-identical (the hc staging round-trip was dumped
  device-side, `tools/b_dbg`), so the drift is inside layer 4's delta-net block
  on replay, not in the hc handoff.

Capture-instrument caveat that invalidated the first bisect: the diag eval
callback fires **twice per node per graph compute** (16 captures for 4 ubatches
in one run), so `occ` is `[ub0, ub0, ub1, ub1, ub2, ub2, ub3, ub3]` and cannot be
mapped to tokens by row index. `index.jsonl` records `ne` and `nbytes` per
capture, which is what makes the pairing unambiguous.

### Fallback (implemented)

Until the replay is fixed, group a decode call only when its whole batch is one
ubatch and `G` divides the layer count with one group per 4 layers:

```cpp
group_batch = ... && n_tokens_all <= cparams.n_ubatch;
```

Multi-ubatch calls then take the plain path, which is the reference by
construction, and `SABAH_PREFILL_GROUP=0` still restores it for every case.
The measured envelope above says this fallback is *not* sufficient on its own --
G must be restricted as well (G=4 verified IDENTICAL at 1 ubatch/SEG=0) -- so the
fallback is a bisect aid, not an acceptance.


## 7. Cost model (measured basis: 0.96 GB/s, 120 GB per 2048-token chunk)

| prompt | today | group-major (G=4) | RAM |
|---|---|---|---|
| 1630 tok | 91 s, 94 GB | ~50-60 s, <= 57 GB (one chunk: no win expected) | 67 MB |
| 8K tok | ~8 min, 480 GB | ~1 min, <= 57 GB | 335 MB |
| 128K tok | ~130 min, 7.7 TB | ~4 min, 228 GB (4 x 32K blocks) | 1.34 GB |

Not yet in the table, must be measured in the first run:

* graph builds: 12 per ubatch (192 at 8K, 3072 at 128K). A group graph has 1/12
  of the nodes of a chunk graph, so the estimate is a few ms each; the first run
  reports the measured build time per group.
* hc D2H: 40 KB/token, 5.37 GB over a 128K prompt, pinned host buffer.

Stop rule (user): if 8K prefill is not at least 3x faster than chunked after
implementation, stop and report.

## 8. Risks

* Slab grouping (section 5) is the largest schedule risk: 4.76 GB pinned with
  1.24 GB of headroom, per group, 12 times.
* `n_rs_seq` is 0 in today's runs; if it ever becomes nonzero the replay must
  carry `rs_idx` (3.4). The gate asserts this rather than assuming.
* The hc tap is a new output in the last group graph; if the tap is not placed
  exactly before the out-ids gather (I5), the next block reads a gathered
  tensor and the equivalence test fails immediately -- this is the intended
  failure mode, not a silent one.
* 3072 `set_inputs` calls at 128K, each with a CPU PLE gather (~90 B/token).
* Graph build time per group is unknown (section 7).

## 9. What is not in scope

Device-side residency, prefetch, MTP: not now.

## 10. Decode guard

B is prefill-only: decode keeps `process_ubatch` as is, and the whole feature is
behind `SABAH_PREFILL_GROUP` (default 0 until the gate passes; 0 must restore the
current behaviour exactly). The item-4 result -- 128K chat tail 42.74 -> 30.36 ms
(32.93 tok/s), `sabah gate` PASS -- is the regression guard and is re-run after B
lands.

Target 30-35 tok/s at *true* 128K KV depth is not yet demonstrated: the item-4
runs reached only ~3.7K KV tokens. The 124K-prompt / 8K-decode test in section 6
is the first measurement at >= 100K depth, and it is part of B's acceptance, not
a follow-up.
