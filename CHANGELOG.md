# Changelog

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
