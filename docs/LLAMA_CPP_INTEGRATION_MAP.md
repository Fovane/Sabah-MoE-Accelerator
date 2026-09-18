# llama.cpp integration map

Status: archaeology complete; the full Sabah backend is **not integrated yet**.
This document records the real seam so that the next implementation does not
confuse tracing or llama.cpp's existing CPU-MoE offload with Sabah execution.

## Pinned dependency and working-tree state

The inspected checkout is `D:\llama-glm53`.

```text
HEAD: 96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b
describe: b10969-2-g96ffdc41c
architecture: qwen4exp
```

The checkout is dirty and contains pre-existing GLM53/Sabah tracing and
experimental tools. Those changes were preserved. The map below describes the
current source, not an assumed clean upstream checkout.

## Execution map

| file | function / symbol | relevant op or tensor | Sabah interception possibility | chosen strategy | reason |
|---|---|---|---|---|---|
| `src/llama-arch.cpp:440-451` | tensor-name map | `blk.%d.ffn_gate_inp`, `ffn_gate_up_exps`, `ffn_down_exps`, `ffn_up_exps` | The GGUF names and logical layer/expert axes are available at model load time. | Keep llama.cpp tensor ownership and use this as the address-map source. | This is the authoritative mapping boundary; changing the GGUF is unnecessary and unsafe. |
| `src/models/qwen4exp.cpp:150-252` | `llama_model_qwen4exp::load_arch_tensors` | `ffn_gate_inp`; expert tensors shaped as `[n_ff_exp, n_embd, n_expert]`; merged gate/up tensors | Expert tensor metadata, quant type, layer index and byte strides are known here. | Add validation/metadata plumbing only if needed; do not duplicate model loading in Python. | The full model loader already knows the exact shard/tensor and layout. |
| `src/models/qwen4exp.cpp:974-1019` | `graph::build_layer_ffn` | qwen4exp calls the common MoE builder, then adds the shared expert | This is the qwen4exp-specific seam, but it is too high-level to replace only the routed expert arithmetic without reimplementing graph behavior. | Leave the graph intact; route the backend below it. | Preserves hyper-connections, shared expert, recurrent/attention state and upstream behavior. |
| `src/llama-graph.cpp:1994-2154` | `llm_graph_context::build_moe_ffn` | `build_lora_mm(gate_inp, cur)`, gating, `ggml_argsort_top_k`, `ggml_get_rows` | `selected_experts` is the authoritative router output. A callback/trace is observation only, not runtime authority. | Consume IDs at the backend boundary; never replace the router or feed trace IDs back into the graph. | This is the semantic point that enforces “if E is selected, E executes”. |
| `src/llama-graph.cpp:1546-1580` | `build_lora_mm_id` | `ggml_mul_mat_id(ctx0, w, cur, ids)` | Every routed expert projection enters GGML as `MUL_MAT_ID` with the same ID tensor. | Intercept `MUL_MAT_ID` data movement/execution, not the tokenizer or full graph. | One narrow seam covers merged gate/up, separate gate/up and down projections. |
| `src/llama-graph.cpp:2207-2340` | routed expert FFN body | gate/up `MUL_MAT_ID`, SiLU/SwiGLU, down `MUL_MAT_ID`, routed weighted aggregate | A custom backend/op can preserve the existing graph around the expert projections. | Strategy A/C hybrid: backend-level residency hook first; custom op only if a compact persistent tensor is required. | It minimizes graph divergence while allowing exact IDs and quantized source bytes to remain visible to the CUDA backend. |
| `src/llama-arch.cpp:896-902` | tensor operation metadata | expert tensors are associated with `GGML_OP_MUL_MAT_ID` | This tells scheduler/backend code which tensors are routed expert weights. | Use this metadata to identify expert weights; do not infer by tensor names alone. | It survives model naming/layout variants supported by llama.cpp. |
| `ggml/src/ggml-backend.cpp:1703-1818` | scheduler split input copy | host MoE weight + first node `GGML_OP_MUL_MAT_ID`; exact IDs read from `node->src[2]` | This is the smallest existing data-movement interception. It already gathers used IDs and copies contiguous expert ranges. | Extend this hook to a native Sabah residency adapter only after the adapter can provide a persistent, correctly indexed device view. | The authoritative IDs are available here, but the current code only fills an ephemeral full-shaped copy and does not call Sabah. |
| `ggml/src/ggml-cuda/ggml-cuda.cu:1910-1960` | `ggml_cuda_mul_mat_id` | quantized `src0`, activation `src1`, device ID tensor `src2` | Exact device execution already exists, including quantized fast/fallback paths. | Prefer feeding this path with Sabah-managed resident expert storage rather than rewriting quantized kernels prematurely. | Reusing llama.cpp CUDA kernels reduces numerical and maintenance risk. |
| `common/arg.cpp:2756-2771` | `--cpu-moe` / `--n-cpu-moe` | expert tensor buffer overrides | This is the current supported way to keep experts on host and let scheduler copy selected ranges. | Keep as reference baseline; expose Sabah as a separate explicit backend mode later. | Calling this path “Sabah” would be false: there is no Sabah LRU, telemetry or `sabah_rt.dll` call. |

## Expert address mapping

For qwen4exp, the model loader establishes this logical mapping:

```text
llama block il
  -> blk.il.ffn_gate_up_exps / ffn_down_exps
  -> GGML tensor expert axis ne[2]
  -> expert e at byte offset e * nb[2]
  -> source GGUF shard/range owned by llama_model_loader
```

The exact byte offset must be taken from the loaded tensor's `nb[2]`, not from
an assumed constant. The scheduler already uses the same value at
`ggml-backend.cpp:1712-1713`. Sabah's `ExpertId(block, expert)` can represent
the logical key, but a native adapter still needs the tensor pointer, role,
quant type, row stride and source byte range for each role.

The following representatives are mandatory mapping tests before claiming an
integrated path: `(0,0)`, `(0,17)`, `(0,511)`, `(2,0)`, `(2,17)`, and
`(47,511)`, for every expert role present in that layer. The tests must verify
that layer, role, shard and `nb[2]` agree and that no cross-layer alias occurs.

## What is already proven versus what is not

Already proven in the Sabah repository:

- exact CUDA block arithmetic for the supported expert quant types;
- the Python `HotTier` LRU, eviction/refetch and telemetry in the standalone
  block path;
- real llama.cpp loading and reference generation for the target GGUF.

Not proven by the current llama.cpp modifications:

- a llama.cpp graph calling `sabah_rt.dll`;
- persistent Sabah residency across full-model graph executions;
- Sabah execution of the authoritative `selected_experts` IDs;
- full-model logits, hidden-state or greedy-token equality;
- a Sabah-backed API server.

The existing `GLM53_SABAH_TRACE` and `GLM53_MOE_TRACE` changes materialize or
record tensors such as `ffn_moe_topk`; they are useful diagnostics but cannot be
the production routing authority.

## Chosen implementation strategy

The smallest correct target is a reversible **Strategy A/C hybrid**:

1. Keep `qwen4exp`'s full llama.cpp graph unchanged when Sabah is disabled.
2. Add an optional native adapter at the scheduler/backend boundary where the
   exact `MUL_MAT_ID` IDs and host expert tensor are both available.
3. Give that adapter a persistent per-device residency table keyed by
   `(tensor identity, block, expert, role)` and an exact source-range callback.
4. Execute the existing CUDA `MUL_MAT_ID` quantized kernels against the
   adapter's resident representation, or add a narrowly scoped custom GGML op
   if the current full-index tensor contract cannot represent a compact hot
   tier without remapping IDs.
5. Keep reference and Sabah modes explicit. A requested Sabah benchmark must
   fail loudly when the native adapter is unavailable; it must never silently
   fall back to `--cpu-moe`.

This is intentionally a target architecture, not a claim that the current
checkout has completed it. The present scheduler copy path allocates/uses a
full-shaped backend tensor and copies only selected ranges into it. It has no
persistent LRU slot mapping and cannot be relabelled as Sabah execution.

## Concrete blocker for v1.0

The current `sabah_rt.cu` API is a standalone single-vector block executor:
`sabah_moe_block(d_x, d_out, d_h, d_ptrs, d_w, ...)`. It does not expose a
GGML backend interface, tensor-backed `MUL_MAT_ID`, batched activations, or a
source-range callback. The Python `HotTier` owns the cache through ctypes, so
llama.cpp cannot call it during graph execution.

Bridging this requires a native adapter and either:

- a persistent full-index device representation (likely virtual/sparse
  addressing), or
- a new compact expert op that carries authoritative IDs plus a slot map while
  keeping the original IDs for router weights and aggregate ordering.

Until that native seam is implemented and tested through the real graph, Gate F
remains `NOT READY`; no v1.0 package or Sabah API claim is justified.

## Dependency policy

The release should pin the above llama.cpp commit (or a clean commit plus a
small documented patch set). The current `D:\llama-glm53` tree contains
experimental, unrelated local additions and is not a shippable dependency
snapshot. A future release artifact must record the exact clean base commit,
Sabah patch, CUDA build flags and native runtime DLL beside its results.
