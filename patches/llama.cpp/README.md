# Reproducible llama.cpp integration

This directory contains the exact upstream base and the native Sabah bridge
patch used by the clean-build gate. The dirty development checkout at
`D:\llama-glm53` is not part of the release artifact and must not be used as
the source of truth.

1. Clone `https://github.com/ggml-org/llama.cpp.git`.
2. Checkout commit `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`.
3. Apply `0001-sabah-native-mul-mat-id.patch`.
4. Build with CUDA enabled for the target GPU.

The patch is opt-in through `SABAH_LLAMA=1` and leaves the normal llama.cpp
path unchanged. It preserves the authoritative graph IDs and hidden state,
keeps the host GGUF expert bank out of a full device-shaped copy, and invokes
Sabah only at the existing `GGML_OP_MUL_MAT_ID` backend boundary.

The runtime ABI is loaded from `SABAH_RT_LIB` (or the platform default) and
requires `sabah_rt_init`, `sabah_rt_mul_mat_id`, `sabah_rt_last_error`, and
`sabah_rt_get_metrics`.
