# Supported models

## Verified

`qwen4exp` is the only architecture accepted by the Sabah accelerator today.
The validated artifact is Qwen3.8-Flash-Next UD-Q4_K_XL with contiguous,
quantization-block-aligned expert slices.

The inspector checks the GGUF metadata, tensor names, expert axis, qtype,
per-expert byte size, shard and offset. An artifact that fails these checks is
refused; Sabah does not reinterpret arbitrary MoE layouts.

## Not supported

Other GGUF architectures and any artifact with non-contiguous or misaligned
expert slices. Unsupported artifacts may be used with an external reference
runtime, but they are not accelerated by Sabah.

