# Changelog

## 1.0.0

First release validated against a preregistered correctness contract
(`docs/V1_CORRECTNESS_CONTRACT.md`, frozen at `ca0a222` before any
validation prompt was run). Results: `docs/V1_VALIDATION_REPORT.md`,
`results/v1_validation/`.

- All hard gates pass: reproducible build; structural exactness (coverage,
  ids, bytes); float64 op-level ≤ 2e-6 (max 9.25e-7 over 27,972 samples);
  wrong-expert sentinels 12/12; residency invariance; 1/2/4-way concurrency;
  prompts across the ubatch boundary; full-graph non-inferiority (UB95
  R = 1.033 ≤ 1.25); greedy-flip non-inferiority (UB95 D = 0.0089 ≤ 0.03).
- **Slower than stock llama.cpp on the validation machine: 0.254× decode
  throughput (MEASURED).** v1.0 validates correctness, not speed.
- Runtime: in-runtime float64 self-check, exact status counters, byte
  verification, test-only sentinel builds, zero-row expert ops handled.
- Server: `--validate`, `--parallel`; no fixed backend timeout. Validation
  attempt 1 failed three API gates on a 600 s proxy timeout; fixed and rerun
  as the contract requires.
- Tooling: `tools/v1/` (driver, float64 oracle, API harness, evaluator,
  build check); `llama-sabah-diag` multi-prompt / multi-sequence modes.

## 0.9.0-rc4

Root cause of the RC3 drift found and fixed. Details and evidence:
`docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md`, `results/rc4_numerical/`.

- **Fixed a structural bug:** the native `MUL_MAT_ID` path fed every selected
  expert's down projection with expert slot 0's activation. Runtime ABI is now
  `sabah_rt_mul_mat_id_v2` (per-slot input rows); v1 is removed so a mismatched
  patch and library cannot load together.
- **Fixed:** decode crashed under CUDA-graph capture; CUDA graphs are now
  disabled for Sabah-routed `MUL_MAT_ID`.
- **Fixed:** ggml-cuda operator fusion could consume the host expert tensor
  directly (illegal memory access); fusion now never spans a Sabah-routed op.
- **Fixed:** a slot fetched earlier in a call could be evicted before that
  call's kernel ran when the tier was smaller than the call's working set.
- Found that RC3's "Sabah" run executed Sabah only for prompts of 32 or more
  tokens; decode ran on the CPU. `GGML_OP_OFFLOAD_MIN_BATCH=1` is now required
  and set by `sabah serve`.
- Added byte verification, coverage counters and a live status file; `/health`
  reports the runtime's own counters.
- Added `llama-sabah-diag` (patch 0002) and the `tools/rc4/` harness.
- Measured: Sabah's `MUL_MAT_ID` matches float64 math to 7.8e-7 (worst of 144
  ops); llama.cpp's CPU and CUDA backends are at 5e-3 to 3e-2 because they
  quantize activations to 8 bits. 16-token greedy equivalence with native CUDA
  passes; 64-token diverges at token 38, a contested step whose flip rate
  matches llama.cpp's own CPU-vs-CUDA variation.
- Measured speedup on the development machine: **0.30× vs stock llama.cpp**
  (Sabah is slower). v1.0.0 is not released.

## 0.9.0-rc3

- Added a reproducible llama.cpp integration patch pinned to upstream commit
  `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`.
- Verified a clean CUDA llama.cpp checkout builds `llama-cli` and reaches the
  native Sabah `MUL_MAT_ID` bridge on the real Qwen3.8-Flash-Next GGUF.
- Added integrated router/weight/intermediate trace comparison artifacts.
- Kept v1.0 closed: CPU-reference versus CUDA expert arithmetic diverges on
  multi-token prompts and can change later router IDs; no speedup or API
  backend claim is made.

## 0.9.0-rc1

- Added explicit expert identities and exact range validation.
- Hardened the hot tier with serialized admission and a correctness fence
  before overwriting a slot that may still be read by the compute stream.
- Added byte-weighted hit telemetry and duplicate-request accounting.
- Added `sabah benchmark` with reference provenance and honest speedup gating.
- Added `sabah serve` as a localhost-only OpenAI-compatible reference-mode
  proxy; it refuses implicit fallback.
- Added packaging metadata, quickstart, architecture, benchmark, supported
  model, and limitation documentation.
