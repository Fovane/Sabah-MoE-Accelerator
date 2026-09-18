# Changelog

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

