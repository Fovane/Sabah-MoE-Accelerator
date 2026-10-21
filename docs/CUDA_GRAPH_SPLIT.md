# Splitting CUDA graphs at the Sabah bridge

## Step 0: where the build time goes (measured, not projected)

Same command line ninja uses (CUDA 12.8, one arch `compute_120a/sm_120a`, MSVC 14.44,
`-j 1`, no Defender exclusion, warm cache), `-c` to a temp object:

| source | wall | result |
|---|---|---|
| pristine `ggml-cuda.cu` (git HEAD) | **6.8 s** | 1,560,800 B object, `.nv_fatbin` present |
| + the host-only op-times/diagnostic block | **>240 s** | killed in `cicc` (237 s CPU, one core) |

Cause: the added block is ordinary host code but uses `std::map<std::string,double>`,
`std::vector<std::string>` and `std::sort` with `std::greater<std::pair<double,std::string>>`.
`cicc` runs the TU for the device pass and grinds on that, not on the kernels.
Bisected by adding that block alone to the pristine file: 6.8 s -> >240 s.

Fix: no STL-heavy host code in `ggml-cuda.cu` again. The segmentation lives in a new
`ggml/src/ggml-cuda/sabah-seg.cpp` compiled by `cl.exe` (seconds). `ggml-cuda.cu` keeps a
routing hook, three one-line wrappers to `ggml_cuda_compute_forward` /
`ggml_cuda_try_fuse` / `ggml_cuda_is_view_or_noop`, and C-style event arrays for the
op-time diagnostics. Expected incremental cost: ~7 s + ~2 s + link, verified with `--ptx`
before any full build.

## Placement

In-backend split. A scheduler split would need a new backend (buffer type, device,
`graph_compute`, `supports_op`, `offload_op`) and a backend boundary with a tensor
hand-off and an event at every Sabah node (144/token). Both approaches issue the same
4 launches per block, so the scheduler variant buys no time and touches a more central
file. `ggml_cuda_graph_check_compability` is not modified: the hook in
`ggml_backend_cuda_graph_compute` runs before the native path.

## Order, capture, replay

`sabah_seg_compute` walks the single CUDA split's node list and splits at every
Sabah-owned `MUL_MAT_ID` (and at any `MUL_MAT_ID` that needs the synchronizing
fallback). Segments are maximal runs of other nodes:

```
launch(segment 0), eager(sabah), launch(segment 1), eager(sabah), ...
```

Every launch, every eager node and the bridge's own `cudaStreamSynchronize` use
`cuda_ctx->stream()`, so order is program order and a segment's kernels are complete
before the bridge reads the ids. Capture uses the same node runner as eager execution,
so replay runs the same kernels with the same arguments.

## Are the per-token destination pointers stable? (the point that decides the item)

Full-attention blocks (12): the KV write is `ggml_set_rows(k, k_cur, k_idxs)`
(`src/llama-kv-cache.cpp:1350`, V at `:1385`, `:1406`). The destination is the whole
per-layer K/V tensor, so its pointer is constant; `k_idxs` is a graph input filled by
`set_input`, so only its contents change. Attention reads a view whose length is
`get_n_kv()`, padded to 256 (`src/llama-kv-cache.cpp:1250-1263`).

Recurrent/SSM blocks (36): `src/models/delta-net-base.cpp:456-496` copies the conv state
into `conv_states_all` at `(s_slot*mem_size + kv_head)*row_size`, and
`:541-603` copies the delta-net state into `ssm_states_all` at `kv_head*row_size` plus
the snapshot view at `:597-603`. `kv_head = mctx_cur->get_head()` (`:458`), and
`llama_memory_recurrent_context::get_head()` returns `is_full ? 0 : mem->head`
(`src/llama-memory-recurrent.cpp:1282-1283`); `head` only moves on eviction or shift
(`init_update`, `:511`). One sequence, no eviction, so `kv_head` is constant and so are
the destination pointers.

Conclusion: no destination pointer changes per token in steady state. A shift or an
eviction moves `head`, the property scan sees the new pointer and rebuilds that segment
once - the same rule native llama.cpp graphs already follow. The only regular recapture
source is the 256-token `n_kv` boundary, which also forces the compute-buffer realloc.

## Warm-up and capture safety

Per segment: run eagerly on the first two calls, capture on the third if the property
tuple is unchanged. The pool is VMM (driver log says `VMM: yes`,
`ggml-cuda/common.cuh:684`), so growth maps a pre-reserved range instead of calling
`cudaMalloc`; the two warm-up passes already mapped every temporary. Any
`BeginCapture`/`EndCapture`/`Instantiate` error demotes that segment to eager
permanently and logs one line, so a capture problem costs speed, never correctness.

## Cache key and call count

Key is `(cgraph pointer, segment ordinal)`, not the first node pointer. `SEG_STATS`
prints `calls/token`. For reference: the per-layer token embedding the question mentions
is a gemma3n tensor (`src/llama-arch.h:493`, `llama-arch.cpp:553`); `qwen3next` has no
such tensor, so one CUDA split per token is expected. The key and the counter are
correct either way.

## Per-drain accounting (no new code needed)

`SABAH_LLAMA_TRACE=1` already prints `SABAH_PHASE ... | per_call us: ids= fetch= wait=
slot= ptr= prime=` (`sabah/runtime/cuda/sabah_rt.cu:3129-3137`), where `ids` is the
router-id download plus its stream synchronise, `slot` the residency lookup plus
allocation and eviction, `fetch` resolve plus disk reads plus H2D, `ptr` the pointer
table upload and hand-off. `SABAH_EV` adds the kernel and launch split. The report will
carry those per token next to the segment host costs (launch, capture, eager) and the
total ms/token, which is what decides whether device-side residency is the next step.
Stop rule unchanged: saving < 6 ms/token -> stop and explain the gap against 12.8 ms.

## Verification

`SEG=1` first, then `SEG=48`. Byte-identical `fed.i32`/`argmax.i32`/`logits.f32` against
the current TILE3 build at `-c 131072` and `-c 512`, prompts A and B, >= 1024 tokens;
plus one 640-token run that crosses the `n_kv` 256 and 512 boundaries to exercise
recapture and the compute-buffer realloc. Report captures/token, recapture cost
(capture + instantiate), GPU idle gap per layer (ms/token minus summed node GPU time
from a `GGML_CUDA_OP_TIMES` run plus `SABAH_EV` kernel time), steady ms/token with
pinned clocks N>=3, and `sabah gate`.

## Files

* new `ggml/src/ggml-cuda/sabah-seg.{h,cpp}` (+1 line in `ggml-cuda/CMakeLists.txt`)
* `ggml-cuda.cu`: routing hook, three wrappers, C-style op-time events
* rollback tag `rollback-item4-base`, branch `phase-e-item4-graphs`

## Build cost after the split (measured)

| change | wall |
|---|---|
| diag + Python only (no ggml-cuda.cu touch) | 4.2 s |
| `sabah-seg.cpp` added, hook slimmed (cmake regen, 2 TUs, link) | 2 min |
| same edit again (incremental) | 6.6 s |

The >30 min hang was the std::map/std::vector block going through the device
pass of `ggml-cuda.cu`. Moved to a host TU, the file compiles in seconds.
`-j 6`, no ccache.

## 4a: what actually gets split

GGUF metadata (`tools/gguf_head.py`): `general.architecture = qwen4exp`, 48
blocks, 512 experts. Shard `-00002-of-00002` holds exactly one tensor,
`per_layer_token_embd.weight`, IQ4_NL, `ne = 160 x 320001536` (51,200,245,760
elements, 28.8 GB at 4.5 bpw).

Loader: `src/models/qwen4exp.cpp:188` creates it with `TENSOR_READ_LAZY`.
`llm_graph_input_ple` builds the gather **once per token**, not once per layer;
`build_ple(inp, emb, hidden, il)` only consumes `ple_embd` inside each layer.

Backend: `sabah/tools/stream.py:59` passes
`-ot \.per_layer_token_embd\.weight=CPU`, so the GET_ROWS runs on the CPU
backend. **Measured** (SEG_STATS at `--tokens 1/2/4`, and the `FUSE_SPLIT`
trace): one `llama_decode` hands the CUDA backend **5 graph computes**, not one:

| graph compute | Sabah nodes | what it is |
|---|---|---|
| 1 | 0 | the split before the PLE gather |
| 2-5 | 36 each | 4 MoE splits, 12 layers each, 144 Sabah nodes per token |

So "one CUDA split per token" is wrong here, and so is "two". Consequences:

* The cache key is `(cgraph pointer, first COMPUTE node index)`, not just the
  cgraph pointer: all five splits share the pointer, and only the COMPUTE flags
  tell them apart (`sabah_cuda_seg_is_split` checks the flag too).
* `SEG_STATS` reports `calls`, `moe_calls` and `native_calls` (totals), plus
  `per moe call` averages. Per token: 5 calls, 4 of them MoE, so the per-token
  value is 4x the per-moe-call value.
* The per-token host cost of the eager path is paid 4 times, once per MoE
  split, which is why the segmentation has more to win than the single-split
  estimate assumed.

Why the baseline is eager: `ggml_cuda_graph_check_compability` clears
`use_cuda_graph` for any `MUL_MAT_ID` whose `src[0]` buffer is host memory when
`SABAH_LLAMA` is set. The MoE split runs node by node on every token today;
that host submission is what the segmented path removes.

## 4b: fusion across a boundary

`ggml_cuda_try_fuse` already refuses any pattern that has a Sabah node within
the next 8 nodes, and no pattern spans more than 8 nodes (comment at the guard).
Every segment end is either a Sabah node or the end of the split, so a pattern
can never cross a boundary.

That is asserted, not assumed: `sabah_cuda_run_nodes` keeps
`GGML_ASSERT(i + nodes_to_skip < i1)` on every fused node of every token.

The `fusions` counter is **not** comparable between the two paths: a pattern
inside a captured segment is counted once, at capture, and not once per token.
The unit check is therefore a decision dump. `SABAH_FUSE_TRACE=1` prints one
line per visited node (name, op, nodes_to_skip) plus one line per scheduler
split, and `tools/fuse_diff.py` compares the ordered decision lists per split
between SEG=0 and SEG=48 over one full token. Node names are used because node
indices are relative to a split.

A `MUL_MAT_ID` that needs the synchronizing fallback keeps the whole graph on
the native path (`needs_sync` in `sabah_cuda_seg_compute`): such a node can sit
inside a fusion pattern, and a wrong split there would change the fused set.

Multi-stream: the hook declines when `stream_context().concurrent_events` is
non-empty, so a fork/join chain is never cut by a boundary. On this model the
set is empty (checked at run time, printed in the report).

`tools/fuse_diff.py` compares decisions **inside one graph shape**. It cannot
see a decision change caused by moving a graph boundary: the fusion guard is
neighbourhood-based (`ggml-cuda.cu:3610-3614`), so a pattern that is whole in
one shape can straddle a boundary in another. The layer-group prefill
experiment hit exactly that (`docs/PREFILL_GROUP_MAJOR.md` 6.2): a different
fused set at one `rms_norm`+`mul` pair produced a 1.0 logit drift with no
pointer or shape change at all, so the property check was right and the
*numerics* moved.

## 4d: property check, tightened (October 21)

`prop_fill` now records `sizeof(ggml_tensor)` for **every** node, not only the
COMPUTE ones, and zeroes only the src tuple of a non-COMPUTE node
(`sabah-seg.cpp:167-171`). A view's `data` is `base + offset`, so a moved
backing allocation or a moved offset now forces a recapture on its own, even
when its only consumer sits in another split. The src tuple stays zeroed for
non-COMPUTE nodes because those buffers belong to another split and must not
invalidate this one.

Measured after the change: fusion decisions 15/15 split blocks identical
`SEG=0` vs `SEG=48` (`tools/p1_fuse.py`), and the `-c 2048` 640-token gate
reproduces `c30f01a8d676d64f` byte-identically on both sides.

## 4c: capture, instantiate and the boundary stall

Measured shape at `-c 512` (53-token prompt, 128 decode steps, SEG=48): the
graph has **36 Sabah nodes per MoE call** and produces **24 graph segments per
call, 96 per token** (two Sabah nodes that are adjacent leave no work between
them, so they do not create a segment). `SEG_STATS` reports totals plus per
moe-call averages, so `segments=96 per token = 4 x 24`.

A segment is captured once. `captures` after the first decode equals the number
of segments (96 here): the prefill builds the graph with `ne[1] = 53` and the
first decode token with `ne[1] = 1`, so every segment legitimately recaptures
at that transition. `SABAH_SEG_DEBUG=n` names the nodes that force it; the
measured answer is the batch dimension of the fused MoE views
(`ffn_moe_weighted-0`, `ne 2560,53 -> 2560,1`), not a buffer move.

After that, a capture happens only when the compute buffer or the KV padding
moves (the 256/512-token boundaries at `-c 2048`+). Expected steady
`captures/token` from the measured segment count is `96/256 = 0.375`, not the
0.57 that assumed 145 segments and one split per token.

`worst call host_ms` is the prefill call (9.5 s here, first execution of every
kernel); `no-capture` is the worst call with `captures == 0` and is the number
that shows segmentation overhead in steady state.

## 4d: property check

`SEG_STATS` reports `props=` ms per graph compute: the per-segment comparison of
a node copy plus `src->data`, `src->ne`, `src->nb` for every source, over every
COMPUTE node of the segment, once per graph compute.

Two details that matter for the cost:

* only COMPUTE nodes are compared. A non-COMPUTE node belongs to another split,
  its buffer can move without invalidating this segment, and including it made
  every token recapture (measured: `ffn_moe_weighted-0` views, `ne 2560,53`).
* the block is a fixed `SEG_PROP_BYTES` per node, compared with one `memcmp`,
  and the scratch buffer is reused. Measured cost at `-c 512`: 0.18 ms per graph
  compute, i.e. ~0.9 ms per token for 96 segments and 5 graph computes.

### Is the COMPUTE filter safe?

The native path is stricter, and this is what it does in this version
(`ggml-cuda.cu`, pinned `96ffdc4`):

```
2769:    if (cgraph->uid != 0 &&
2770:        cgraph->uid == graph->uid) {
2771:        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
2772:        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
2773:        return false;
2774:    }
...
2784:    for (int i = 0; i < cgraph->n_nodes; i++) {
2785:        ggml_cuda_graph::node_properties prop = {};
2786:        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));
2787:
2788:        for (int j = 0; j < GGML_MAX_SRC; ++j) {
2789:            if (cgraph->nodes[i]->src[j]) {
2790:                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
2791:                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
2792:                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
2793:            }
2794:        }
2795:
2796:        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
```

It compares **every** node of the split view, no flag check, and it does not
skip views (`ggml_cuda_is_view_or_noop` is only used in
`ggml_cuda_graph_check_compability`, line 2727). So the native path invalidates
on a view's pointer change too.

The filter here cannot miss a change that reaches a captured kernel, because of
where the comparison is anchored:

* every kernel in a captured segment is a COMPUTE node of this split, and its
  inputs are recorded from the **consumer** side: `src[j]->data`, `ne`, `nb`.
* a view or reshape that feeds such a kernel resolves to a concrete pointer, so
  a moved base allocation or a moved offset (KV row, recurrent-state slot)
  changes the consumer's `src[j]->data` and is caught.
* a non-COMPUTE node with no consumer inside the segment has no kernel in the
  segment either, so its buffer cannot affect the captured graph.
* `op_params`, op and shape of every COMPUTE node are in the node copy.

Gap shared with the native path: neither stores `src[j]->type`. A type change
without an `ne`/`nb`/op change is not detected by llama.cpp either, so this is
not a regression.

Evidence that it is not just theoretical: with the filter, `-c 512`, 400 tokens
gives `captures=108` (96 at the prefill-to-decode transition, 12 at the n_kv 256
boundary) and byte-identical output; without it, every token recaptured.

## 4e: acceptance (measured)

Default is now the segmented path: `SABAH_CUDA_GRAPH_SEG` unset means 48 layers
(144 Sabah nodes); `=0` restores the eager path (`SEG_STATS` then reports
`moe_calls=0 native_calls=<all> segments=0`).

| gate | result |
|---|---|
| fusion decisions, SEG=0 vs SEG=48, one token | 10/10 split blocks identical |
| `-c 131072`, chat prompt, 1024 tokens | logits `fa4db9f505f3496f`, argmax/fed `da61e07f62255adb` on both sides; same `reads`/`bytes` counters |
| 640 tokens across n_kv 256/512, `-c 2048` | logits `c30f01a8d676d64f`, argmax/fed `82d3653cfaf21968` on ref, SEG=48, default and `=0` |
| `-c 512`, 400 tokens across n_kv 256 | logits `9dd26e50c9ad2fae`, argmax/fed `0e397d6008bc22ce` |
| 128K chat, 2048 steps, tail 256 | 42.74 -> **30.36 ms/token** (23.40 -> 32.93 tok/s) |
| `sabah gate` | PASS (verify 2000/0, id reuse 768/0, mapped fallback 0) |

Where the 12.4 ms comes from: the eager path synchronizes the host once per
Sabah node (36 per MoE split) to route and fetch; between two such nodes the GPU
is idle. Capturing the runs in between removes the host round trip, so the
saving scales with the number of nodes, not with the tensor sizes. That is why
`-c 512` and 128K agree (11.9 and 12.4 ms) and why the fill phase, which is disk
bound, shows no saving either way (281.8 vs 300.1 ms at 128K, both I/O bound).

Capture cost and stall: 120 captures over 2048 tokens (96 at the prefill to
decode transition, the rest at the n_kv 256/512 boundaries), 0.06 per token;
`capture=` 0.011 ms per graph compute amortised, `worst call host_ms` 9.7 s and
that call has `captures 0` (it is the first prefill compute, not a stall). At
the 256-token boundary in the `-c 2048` run there is no step above its
neighbours.
