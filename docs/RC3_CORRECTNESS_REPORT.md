# v0.9.0-rc3 correctness gate

> **Superseded by RC4.** The drift reported here had a structural cause: a
> down-projection input-row bug in the native `MUL_MAT_ID` path. In addition,
> the RC3 "Sabah" run executed Sabah only for the one prompt of 32 or more
> tokens. See `RC4_NUMERICAL_EQUIVALENCE_REPORT.md`.

## Verdict

`v1.0.0` remains blocked. `v0.9.0-rc3` is the honest release candidate.

The native integration boundary is reproducible and real: a clean checkout of
llama.cpp at `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`, with the patch in
`patches/llama.cpp/0001-sabah-native-mul-mat-id.patch`, built `llama-cli` with
CUDA `sm_89` and produced the same deterministic first token (`1596`, `We`) as
the reference run.

## Measured integrated trace

The same llama.cpp graph, GGUF, and three prompts were run through the
reference and Sabah paths. The recorder covered all 48 MoE blocks, 78 total
prefill/decode rows, router IDs, and router weights.

- Every block file was present and byte-length matched.
- The first prompt’s final token matched (`32`); the second prompt diverged
  (`reference=32`, `Sabah=76`) under the CPU-reference comparison.
- The first router-ID mismatch appeared after the first prompt boundary in the
  multi-token run; the isolated Turkish prompt diverged from block 1 onward.
- Block-0 gate/up intermediate differences were up to `2.85e-2`; the block-0
  routed aggregate reached `2.47e-1` max absolute difference in the captured
  tensor stream.
- Native Sabah cache counters for the three-prompt run were:
  `hits=33780`, `misses=13740`, `evictions=12798`,
  `bytes_fetched=14397132800`, `resident_bytes=1072844800`.

These are not silently relabelled as PASS. The reference uses llama.cpp’s
`--cpu-moe` quantized CPU path, while Sabah uses the native CUDA expert kernel;
their accumulation/backend arithmetic is not yet a proven equivalence class.
The structural guarantee still holds: the graph’s authoritative IDs are
passed through unchanged, the original GGUF expert bytes are used, and no
expert prediction/substitution occurs.

Evidence directories:

- `results/correctness_reference/` and `results/correctness_sabah/`
- `results/correctness_deep_reference/` and `results/correctness_deep_sabah/`
- `results/correctness_block0_reference/` and `results/correctness_block0_sabah/`
- `results/correctness_ops_reference/` and `results/correctness_ops_sabah/`

## Release gate status

| Gate | Result |
|---|---|
| Clean llama.cpp integration build | PASS |
| Native Sabah `MUL_MAT_ID` boundary | PASS |
| Router IDs across all blocks vs CPU reference | BLOCKED |
| Router weights / intermediate aggregate equality | BLOCKED |
| Multi-token greedy equivalence | BLOCKED |
| End-to-end speedup claim | NOT RUN / NOT CLAIMED |
| v1.0 release | NOT READY |

The next correctness task is to establish a numerically equivalent reference
backend (or a documented tolerance/semantic contract) and then rerun the
ladder before adding `backend=sabah` to the API.
