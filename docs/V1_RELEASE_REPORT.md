# Sabah Accelerator v1 release report

## Release verdict

```text
v1.0.0 NOT READY
Release candidate produced instead: v0.9.0-rc4 (see RC4_NUMERICAL_EQUIVALENCE_REPORT.md)
```

The exact expert/block runtime is real and tested. Gate F's first real
full-model Sabah token now passes through the reproducible native
`MUL_MAT_ID` boundary, but the multi-token numerical ladder shows measured
CPU-reference versus CUDA expert arithmetic differences, so shipping `v1.0.0`
would overstate the result. Detailed RC3 evidence is in
[`RC3_CORRECTNESS_REPORT.md`](RC3_CORRECTNESS_REPORT.md).

The llama.cpp archaeology and implemented native seam are recorded in
[`LLAMA_CPP_INTEGRATION_MAP.md`](LLAMA_CPP_INTEGRATION_MAP.md). The selected
seam is the existing `GGML_OP_MUL_MAT_ID` backend/scheduler boundary. Sabah
now receives the real host GGUF tensor, authoritative IDs and device hidden
states there, uses a native LRU and exact quantized expert execution, then
returns to the normal graph.

Starting repository commit: `1bc838c67d11c48a8740306ea4a036dea965ddf9`.

RC3 package: `dist/sabah_moe_accelerator-0.9.0rc3-py3-none-any.whl`.
SHA-256: `6d54780040ee9a8dff78ffb19da72e5a51f87dc99d9de5e56b14d2f28366e611`.

## Gates

| Gate | Result | Evidence |
|---|---|---|
| A — build | PASS | CUDA 13.3 + MSVC build of `sabah_rt.dll` and `mgpu_qualify.exe` |
| B — real model inspection | PASS | Qwen3.8-Flash-Next UD-Q4_K_XL: 48 blocks, 512 experts, 4 shards, 111.335 GB; layout verified |
| C — real expert runtime | PASS | Real GGUF expert bytes loaded and executed on RTX 4050 |
| D — hot-tier cache | PASS | Hit/miss, LRU eviction, reload and residency independence exercised |
| E — expert/MoE correctness | PASS | Four qtypes `max|diff|=0`; block relative L2 `6.03e-7` / `6.02e-7`; wrong-expert discrimination PASS |
| F — end-to-end model | FIRST TOKEN PASS / MULTI-TOKEN BLOCKED | Clean native bridge reproduces the first token; all-block multi-token comparison exposes CPU-reference vs CUDA expert arithmetic drift |
| G — server | PARTIAL | OpenAI-compatible localhost proxy works in explicitly labelled reference mode |
| H — user flow | PARTIAL | inspect/qualify/plan/selftest/benchmark/serve reference flow documented |
| I — claim hygiene | PASS | measured/projected/speedup-unavailable states are separated |

## Current-machine measurements

Machine qualification on 2026-09-18:

- RTX 4050 Laptop GPU, 6.44 GB nominal / 4.75 GB usable
- 29.8 GB RAM / 25.5 GB usable
- simultaneous H2D: 12.73 GB/s; one-device contention factor: 1.00
- storage read at model path: 2.43 GB/s

Real block A/B on block 0, 64 measured tokens, real routing trace:

| Hot tier | Hit rate | ms/token | GPU wait | Host stage | Fetch |
|---:|---:|---:|---:|---:|---:|
| 16 slots | 0.0797 | 11.804 | 0.243 ms | 7.981 ms | 28.27 MB/token |
| 64 slots | 0.2766 | 5.126 | 0.197 ms | 2.946 ms | 22.22 MB/token |
| 512 slots | 1.0000 | 1.312 | 0.000 ms | 0.000 ms | 0 MB/token |

The 512-slot row is the one-block kernel cost. It is not full-model
tokens/second. The 16-slot streaming penalty against that row was measured at
10.492 ms/token.

## Correctness

The CUDA self-test used real expert bytes from blocks 0 and 2:

- Q4_K, Q5_1, Q5_K, and Q8_0 decode matched gguf-py with zero reported max
  absolute difference.
- GPU block output relative L2 was `6.030e-7` for block 0 and `6.015e-7`
  for block 2.
- Replacing one routed expert changed output by `4.680e-1` and `6.657e-1`,
  proving the test detects a wrong expert.
- Forced eviction/refetch returned bit-identical output.

The first-token result is recorded in
[`results/full_model_first_token.json`](../results/full_model_first_token.json):
reference and Sabah both sampled token ID `1596` (`We`). The result also
records native cache counters. Router-weight, intermediate-state, MoE
aggregate and logit error dumps are explicitly marked unavailable rather than
being inferred from final text.

## Projection and unvalidated work

No end-to-end Sabah speedup is claimed. Multi-GPU H2D, sharding, and any
3×3060 result remain `UNVALIDATED`. Existing planner values are projections
from measured single-device inputs and historical traces.

The same target GGUF was also loaded and run for a one-token deterministic
reference smoke test through llama.cpp; its SHA-256 is recorded in
`results/reference_baseline_single_token.json`. That run is a validity smoke
test, not a representative throughput baseline, and no Sabah speedup is
derived from it.

## Known limitations

See [`KNOWN_LIMITATIONS.md`](KNOWN_LIMITATIONS.md). In particular, the current
30 GB-class machine cannot hold the 77.018 GB expert bank in RAM, so the
storage-backed path is primarily for correctness and runtime validation.

## Next work

1. Establish a numerically equivalent reference backend (or a documented
   semantic tolerance) and close the all-block router/intermediate/logit ladder.
2. Replace the reference-mode API backend with the integrated Sabah backend.
3. Re-run identical full-model reference/Sabah benchmarks before considering
   `v1.0.0`.
