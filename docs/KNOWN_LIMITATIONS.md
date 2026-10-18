# Known limitations

## Performance

- **The tested configuration is slower than stock llama.cpp.** v1.0 measured
  0.25× stock decode throughput (0.411 vs 1.619 tok/s) on the validation
  machine: RTX 4050 Laptop 6 GB, ~30 GB RAM. The 77 GB expert bank does not
  fit in RAM, the expert tier is 1 GiB with a 46% hit rate, and every miss is a
  synchronous allocate/copy. No performance work was done for v1.0.
- No end-to-end speedup has been measured on any machine.
- Multi-GPU H2D aggregate, contention and fixed-path sharding are
  UNVALIDATED; all multi-GPU figures are PROJECTED.
- **Context and the expert slab compete for the same VRAM, and losing is an
  abort, not a slow token.** At `-c 32768` a 9.8 GB slab leaves ~0.4 GB of
  headroom; the tier's documented over-budget allocation then has no device
  memory to take, the call returns null and the process aborts (`0xC0000409`,
  reproduced 3 of 3). Measured working configurations — 32K with an 8.0 GB slab,
  or with `-ub 8` — are in [`docs/CONTEXT_VRAM.md`](CONTEXT_VRAM.md).

## Correctness scope

- Validated on one machine, one model (`qwen4exp`, Qwen3.8-Flash-Next
  UD-Q4_K_XL) and 14 + 4 preregistered prompts. The statistical gates use 14
  prompt clusters × 32 teacher-forced steps. Gate I has ~0.79 power at its
  margin, and 14 clusters rule out only large differences in greedy flip
  propensity.
- Greedy output is not guaranteed to match llama.cpp token for token. Sabah's
  expert arithmetic is closer to float64 than llama.cpp's 8-bit-activation
  kernels, so greedy decoding can take the other branch at close calls. In
  validation this happened at 4 of 448 steps, vs 3 of 448 for llama.cpp's own
  CPU backend. One Sabah flip (`json_2`, step 2) was at a reference margin of
  1.35 logits, not a near tie.
- The full-graph comparison (gates H, I) was made on identically instrumented
  graphs. Observing tensors disables CUDA operator fusion across them, for the
  reference and for Sabah alike.
- llama.cpp's native CUDA Q5_1 down projection (MMQ path) deviates from float64
  by ~9e-3 beyond what its 8-bit activation quantization explains. The
  mechanism is not identified. Native CUDA is used only as a comparison anchor,
  never as an arithmetic oracle.

## Integration

- Only `qwen4exp` is accepted.
- Requires the pinned, patched llama.cpp (`96ffdc41c` + patches 0001/0002),
  CUDA and an NVIDIA GPU.
- Sabah executes decode steps only with `GGML_OP_OFFLOAD_MIN_BATCH=1`, which
  `sabah serve` sets. With llama.cpp's default (32), small batches run on the
  CPU and never reach Sabah.
- The HTTP path is `llama-server` with the bridge loaded — the same backend
  `sabah serve` spawns — but it was only exercised at 31-64 generated tokens,
  i.e. inside the warm-up regime, and `sabah serve` itself did not come up on
  this artifact (its `gguf-py` rejects the quant type, then the warm-RAM planner
  fails to pin host memory). See [`docs/CONTEXT_VRAM.md`](CONTEXT_VRAM.md).
- `--validate` (float64 self-check, byte verification) is much slower and is
  meant for verification, not use.
- The 77 GB expert bank does not fit in the validation machine's RAM; the
  storage-backed path is a correctness path, not a performance claim.
