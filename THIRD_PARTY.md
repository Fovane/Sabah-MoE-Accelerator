# THIRD_PARTY

Everything that is not first-party code in this repository, with the licence it
carries and where it comes from. Nothing here is bundled: the llama.cpp checkout
and the model are fetched by the user, per `QUICKSTART.md`.

## llama.cpp (the runtime this forks)

* What: MIT-licensed. `third_party/llama.cpp` is a checkout of
  `ggml-org/llama.cpp` at the pinned commit recorded in each patch header under
  `patches/llama.cpp/` (`96ffdc4` at the time of writing).
* Our changes: `patches/llama.cpp/0001..0005`. They add the Sabah bridge, the
  diag capture tool, the segmented CUDA-graph path, the property-check
  tightening and the (disabled by default) layer-group prefill experiment.
* Attribution: every file under `patches/llama.cpp/` is a diff against upstream
  and keeps its copyright header. `LICENSE` in this repository covers the Sabah
  code only; upstream MIT text stays with upstream.
* Files touched by the patches: `ggml/src/ggml-backend.cpp`,
  `ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml/src/ggml-cuda/CMakeLists.txt`,
  `ggml/src/ggml-cuda/sabah-seg.{h,cpp}`, `src/llama-*.{h,cpp}`,
  `src/models/qwen4exp.cpp`, `common/common.{h,cpp}`,
  `tools/sabah-diag/sabah-diag.cpp`.

## GGML / ggml-cuda

Part of llama.cpp, same licence as above. The quant kernels used by the tile
work (`tools/tile_exact.cu`) are derived from upstream `ggml-cuda` kernels, and
`docs/` records which hunk of upstream each corresponds to.

## The model: Qwen3.8-Flash-Next (ISTA-DASLab), IQ3_S

* What: the weights the benchmark numbers in `results/phase0/REPORT.md` were
  measured with. They are **not** in this repository.
* File: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf`
  (plus shard 2), 83.6 GB, and its `iq3_s`+`iq4_nl` quantisation was produced by
  the ISTA-DASLab tooling. The original weights carry the Qwen licence; the
  quantised files added here carry whatever the quantiser declares. The
  licence text and the required attribution belong to the ISTA-DASLab release,
  and are **not** reproduced here because they are not ours to license.
* What this project adds to them: nothing. The runtime reads the GGUF and
  executes it exactly, which is the correctness contract the gate proves
  (byte-for-byte, 2000 served blocks checked per run).

## NVIDIA CUDA and the driver

* CUDA toolkit (12.8 at the time of the numbers) and the display driver
  (32.0.16.1742 / nvidia-smi 617.42) are used under their own licences; the
  build line that produced the measured binary is recorded in
  `docs/BENCHMARK_METHOD.md`.
* Nothing here redistributes CUDA headers, driver binaries or the CUDA runtime.

## Python dependencies

Read from `pyproject.toml`: only the standard library plus
`nvidia-ml-py`/`psutil`-class helpers where they are declared there. No
runtime dependency is bundled.

## Not in this repository

Model weights, CUDA toolchains, the Windows driver, and the per-run
`logits.f32` byte dumps (their SHA-256 is recorded in
`results/phase0/REPORT.md`; the dump itself is git-ignored).
