# Reproducible llama.cpp integration

The exact upstream base and the patches used by the correctness gate. The
development checkout at `D:\llama-glm53` is not part of the release and must
not be used as the source of truth.

1. Clone `https://github.com/ggml-org/llama.cpp.git`.
2. Check out `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`.
3. Apply `0001-sabah-native-mul-mat-id.patch` (required).
4. Optionally apply `0002-sabah-diag-capture-tool.patch` (diagnostics only).
5. Build with CUDA, e.g.
   `cmake -S . -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89`.

## 0001 — native Sabah `MUL_MAT_ID`

Opt-in via `SABAH_LLAMA=1`; without it llama.cpp is unchanged. With it:

- the scheduler keeps the original host GGUF expert tensor in the graph
  instead of building a device-shaped copy for expert `MUL_MAT_ID`
- `ggml_cuda_mul_mat_id` hands the op to `sabah_rt_mul_mat_id_v2`, passing the
  graph's own ids, input tensor and output tensor
- CUDA graphs are disabled while such an op is present, because the bridge
  reads ids on the host (the same rule llama.cpp applies to its synchronizing
  `MUL_MAT_ID` fallback)
- no fused kernel may consume a Sabah-routed `MUL_MAT_ID`; the op always
  reaches the bridge

Runtime ABI loaded from `SABAH_RT_LIB`: `sabah_rt_init`,
`sabah_rt_mul_mat_id_v2`, `sabah_rt_last_error`, `sabah_rt_get_metrics`.

**v2 is not compatible with v1**, deliberately. v1 omitted the input row
stride and count (`src1->nb[1]`, `src1->ne[1]`) and fed every selected expert
the first input row. That is wrong for the down projection, where each expert
has its own activation. A v1 library and a v2 patch refuse to load together.

### Placement — required

With `--cpu-moe`, llama.cpp sends an expert op to CUDA only when its batch is at
least `GGML_OP_OFFLOAD_MIN_BATCH` tokens (default 32). Smaller batches,
including every single-token decode, run on the CPU and never reach Sabah.
**Set `GGML_OP_OFFLOAD_MIN_BATCH=1`** so every expert op is executed by
Sabah. `SABAH_LLAMA_TRACE=1` prints `SABAH_DIAG calls=… tokens=… lookups=…` at
exit; `lookups` must equal `tokens × n_used`, and `calls` must equal
(expert ops per step) × steps.

### Diagnostics

| variable | effect |
|---|---|
| `SABAH_LLAMA_HOT_BYTES` | VRAM budget of the expert hot tier (default 2 GiB) |
| `SABAH_LLAMA_VERIFY_BYTES=1` | byte-compare every hit and fetch against the GGUF range |
| `SABAH_LLAMA_TRACE=1` | print cache and coverage counters at exit |

## 0002 — `llama-sabah-diag`

A greedy driver that observes the graph without changing it. It dumps full
logits, argmax and fed tokens per step. On request it also dumps named graph
tensors and, for expert projections, the exact `MUL_MAT_ID` input and ids. It
supports teacher forcing. See the header of `tools/sabah-diag/sabah-diag.cpp`.

Observing a tensor splits the graph at that node, which disables ggml-cuda
operator fusion across it. Instrumented and uninstrumented runs are therefore
reported separately, and token equivalence is always measured uninstrumented.
