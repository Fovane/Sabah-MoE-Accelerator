# Known limitations

- Only `qwen4exp` is accepted by the accelerator.
- The CUDA expert/block path is validated on one RTX 4050 Laptop GPU.
- Full-model attention/PLE/KV/tokenizer/sampling integration is incomplete.
- The local API is currently reference-mode llama.cpp only.
- Logits and greedy-token equivalence against the reference are not yet
  demonstrated.
- Multi-GPU H2D, sharding, and contention remain `UNVALIDATED`.
- The 77 GB expert bank does not fit in the development machine's RAM; its
  storage-backed path is a correctness/development path, not a performance
  claim.
- No end-to-end Sabah speedup is claimed.

