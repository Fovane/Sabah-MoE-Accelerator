# Sabah architecture

Sabah preserves the authoritative router decision. If the router selects
expert `E`, the exact stored expert `E` is fetched and executed.

```text
GGUF inspector
    -> exact byte-range profile
    -> mmap/file or RAM backing store
    -> per-device VRAM hot tier
    -> async H2D miss
    -> quantized expert CUDA kernel
```

The backing store is read-only. Expert identity is `(block, expert)` and each
role (`gate`, `up`, `down`) carries its own shard, offset, byte count, and
qtype. The hot tier uses fixed-size per-block slots and LRU replacement.

The current CUDA path is a block executor. It is not yet wired into the full
attention, recurrent/PLE, KV-cache, tokenizer, and sampling graph. The
full-model seam is the local llama.cpp reference engine used by `serve
--allow-reference` while that integration is completed.

