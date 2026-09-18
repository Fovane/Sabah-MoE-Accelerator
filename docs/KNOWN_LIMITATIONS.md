# Known limitations

- Only `qwen4exp` is accepted by the accelerator.
- The CUDA expert/block path is validated on one RTX 4050 Laptop GPU.
- Full-model attention/PLE/KV/tokenizer/sampling integration is incomplete.
- 64-token greedy equivalence with llama.cpp is not met (first divergence at
  token 38, see the RC4 report); 16-token equivalence is.
- On the development machine (6 GB GPU) Sabah is measured 3.3× slower than
  stock llama.cpp: the expert tier is 1 GiB and every miss is synchronous.
- `GGML_OP_OFFLOAD_MIN_BATCH=1` is required for Sabah to execute decode
  steps; without it llama.cpp runs small batches on the CPU.
- The RC4 numerical criterion is derived from one prompt (59 + 64 tokens);
  its layer-3 factor (1.25× control) is a judgement, not a statistical bound.
- Multi-GPU H2D, sharding, and contention remain `UNVALIDATED`.
- The 77 GB expert bank does not fit in the development machine's RAM; its
  storage-backed path is a correctness/development path, not a performance
  claim.
- No end-to-end Sabah speedup is claimed; the measured one is 0.30×.

