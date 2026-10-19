# Phase 0: where the steady-state token goes

Measured on the reference rig, on the current binary, before any runtime change.
Everything below is native timing unless a row says nsys; the nsys rows are used
only for the GPU/CPU split and never for a throughput number.

- GPU: RTX 5060 Ti 16 GB `sm_120`, 16,283 MiB total, driver 617.42
- Host: 15.38 GiB RAM, 8 threads
- Model: `Qwen3.8-Flash-Next` IQ3_S (83.6 GB, 2 shards), 48 blocks, 512 experts,
  top-10, expert bank 50.29 GB
- Binary: `third_party/llama.cpp/build/bin/llama-sabah-diag.exe`
- Runtime: `sabah/runtime/cuda/sabah_rt.dll`
- Flags: `-dev CUDA0 -ngl 99 --cpu-moe -ot "\.per_layer_token_embd\.weight=CPU" -t 8 -c 512 -n 256 -p Hello`
- Env: `SABAH_LLAMA=1`, `GGML_OP_OFFLOAD_MIN_BATCH=1`,
  `SABAH_LLAMA_DIRECT_IO=1`, `SABAH_LLAMA_IO_DEPTH=8`, `SABAH_LLAMA_TRACE=1`,
  `SABAH_LLAMA_HOT_BYTES=9.8e9`, `SABAH_LLAMA_PREFETCH=0`, no status file

## 1. The regimes (native, `results/phase0/baseline-n256`)

| regime | n | median | tok/s | notes |
|---|---|---|---|---|
| cold fill (53-token prompt) | - | **11.27 s** | - | 12.68 GB of expert bytes, tier starts empty |
| warm-up | 128 | 71.86 ms | 13.9 | tier still discovering the working set |
| steady state | 127 | **58.59 ms** | **17.07** | 135.7 reads/token = 91.8 MB/token |
| decodes only (1..255) | 255 | 89.4 ms | 11.19 | |
| end to end | 256 | - | 7.49 | includes the fill |

Peak VRAM 15,417 MiB (plateau 15,367, 916 MiB free). Process peak working set
5,168 MiB. Available host RAM min 5,395 MiB. Resident slab 9.799 GB.
Gate on the same configuration: `verify 2000/0`, `id reuse 768/0`,
`mapped_fallbacks 0`, `overflow 448`, exit 0.

The steady state is reproducible: `control-c512-f16-n256` (run hours later, same
flags plus explicit `-ctk f16 -ctv f16`) gives 58.34 ms.

## 2. GPU busy is 74% of the token and the host is the other 26%

Nsight Systems warm replay (`results/phase0/warm_replay`, two items of 8 decode
steps, KV cleared between items, no disk in item 2). Token span under tracing is
inflated to 206-294 ms, so only the kernel durations are used:

| per steady token | ms | share of token | kernels |
|---|---|---|---|
| Sabah `gate`+`up` | 17.1-18.0 | 29-31% | 96 |
| dense matmul (llama.cpp) | 12.4-12.8 | 21% | 752-770 |
| Sabah `down` | 9.6-10.0 | 16-17% | 48 |
| elementwise / router | 2.45-2.51 | 4.3% | 2,494-2,556 |
| PLE / embedding | 0.42 | 0.7% | 93-98 |
| SSM | 0.23 | 0.4% | 70 |
| attention | 0.11-0.12 | 0.2% | 55-60 |
| other | 0.04-0.05 | - | 54-58 |
| **GPU busy** | **43.3-43.7** | **74%** | **3,662-3,756** |
| not busy (host) | 15.2 | 26% | - |

Sabah's 144 expert calls are 27.1-28.0 ms of the 43.4 ms (62%), reading
~1.01 GB/token at ~37 GB/s effective. The dense matmuls are llama.cpp's own and
cost 12.4-12.8 ms. Everything else on the GPU is 3.2 ms.

The trace also shows ~80 `cudaStreamSynchronize` per token totalling 2.6-2.8 ms,
and 6.4 ms idle gaps between the three expert kernels of one layer whenever a
miss occurs - the host round trip, visible.

## 3. The 15.2 ms host gap is launch submission (Phase 0d)

`tools/launch_latency.cu` (CUDA 13.4, `sm_120`, WDDM), 4,000 samples per case:

| case | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| empty launch + sync, one at a time | 10.10 | 7.20 | 23.5 | 31.7 us |
| empty launch, batched 16 per sync (/launch) | 6.06 | 5.83 | 7.04 | 12.1 us |
| **empty launch submission only, GPU idle** | **5.69** | **5.61** | 6.18 | 7.77 us |
| **empty launch submission, GPU busy** | **3.49** | **3.42** | 3.75 | 4.58 us |
| 64 B D2H blocking copy (includes sync) | 8.78 | 6.20 | 22.4 | 34.8 us |
| busy kernel (23 us) launch + sync | 16.12 | 15.40 | 16.3 | 33.8 us |
| 48x (empty + D2H + sync), per token | 806 us | 791 | 881 | 1,030 us |
| 48x (busy 23 us + D2H + sync), per token | 1,038 us | 1,006 | 1,156 | 1,409 us |

Schedule policy and thread priority make no measurable difference: a second pass
with `cudaDeviceScheduleSpin` and `THREAD_PRIORITY_HIGHEST` gives 3.61 us/launch
busy and 812 us per 48-step token, i.e. within noise of the default.

The account of the gap, using the measured 3.42-3.49 us submission cost:

| item | per token |
|---|---|
| 3,756 kernel submissions x 3.42 us | 12.85 ms |
| 48 router-id drains (D2H + sync) | 0.30 ms |
| residual (memcpy submissions, graph walk) | ~2.0 ms |
| **total** | **~15.2 ms = the measured gap** |

Two consequences follow, and they set the phase order:

1. **The host is not overlapping with the GPU, it is adding to it.** 58.6 =
   43.4 + 15.2, not `max(43.4, 15.2)`. The cause is the per-layer router drain:
   it empties the queue 48 times per token, and after each drain the host must
   re-fill the layer's ~78 launches while the GPU waits. At 3.4 us/launch that
   is ~265 us of exposed submission per layer, x48 = ~12.7 ms.
2. **Launch count, not launch cost, is the problem.** The elementwise/router
   group is 2,556 launches (68% of all launches) for 2.5 ms of GPU work. Those
   launches cost 8.7 ms of host time to submit. Any work that reduces the op
   count also removes host time.

## 4. Context and VRAM: 128K is affordable in KV, but not at a 9.8 GB slab

Measured with `-c 131072`, `-n 256`, same flags, KV types via `-ctk/-ctv`:

| item | f16 | q8_0 | q4_0 |
|---|---|---|---|
| KV cache at 131,072 cells | 384 MiB | 204 MiB | 108 MiB |
| llama.cpp compute buffer | 964 MiB | 976 MiB | 976 MiB |

Only 12 of 48 blocks keep a cache (`full_attention_interval=4`), it is K-only
(V is empty), 2 KV heads x 128 elements. The context itself is cheap.

But the token time at 128K is not, and the cause is the VRAM ceiling rather than
the context. Same prompt, same 131.6 misses/token, same kernel set:

| `-c` | slab | steady median | tok/s | VRAM plateau | free of 16,283 |
|---|---|---|---|---|---|
| 512 | 9.8e9 | 58.34 ms | 17.14 | 15,367 | 916 |
| 8192 | 9.8e9 | 58.39 ms | 17.13 | 15,475 | 808 |
| 32768 | 9.8e9 | 60.04 ms | 16.65 | 15,982 | 301 |
| 131072 | 6.0e9 | **58.23 ms** | **17.17** | 15,377 | 906 |
| 131072 | 7.0e9 | 63.29 ms | 15.80 | 15,991 | 292 |
| 131072 | 8.0e9 | 71.54 ms | 13.98 | 15,994 | 289 |
| 131072 | 9.8e9 | 88.43 ms | 11.31 | 16,000 | 283 |
| 131072 | 9.8e9 + `-ub 64` | 79.29 ms | 12.61 | 15,988 | 295 |

A short nsys capture at 128K (slab 9.8e9) shows GPU busy 92.3 ms/token, with
`k_mul_mat_id` at 465 us/call against ~188 us at 512 context: the same kernel,
the same bytes, 2.5x slower when the card is within ~300 MiB of its ceiling.

The 128K run at a 6.0 GB slab reads 149.5 objects/token instead of 131.6
(+14% bytes) and is still full speed: the extra reads overlap, the VRAM pressure
does not.

Mechanism is not established beyond this: what is measured is that throughput is
flat until the plateau passes ~15.4 GB and then degrades smoothly, and that the
plateau saturates at ~16.0 GB for every slab at or above 7.0e9. Whether that is
WDDM eviction, driver bookkeeping at the ceiling, or the runtime's over-budget
allocations is not separated here.

## 5. Decision

GPU busy is 74%, so **Phase 2 (kernel efficiency) is the primary lever**, per the
plan's own rule. But the plan's rule prices only the GPU side; this measurement
shows a second, equally mechanical lever:

- Phase 1 (remove the exposed host submission): 58.6 -> ~43.4 ms = 23.0 tok/s.
  This is the ceiling of any host-side fix, and it needs the per-layer drain to
  stop emptying the queue.
- Phase 2 (kernel efficiency): 43.4 -> ~25 ms of GPU work (Sabah 27.5 -> 12,
  dense 12.6 -> 8.5) would give ~25-28 ms/token = 36-40 tok/s **only if** the
  host gap is also gone. With the host gap still additive, the same kernel work
  lands at ~40 ms = 25 tok/s.

So the 30-35 tok/s target needs both, in this order: Phase 2 first (it is 74% of
the token and has the clearer headroom - 37 GB/s effective against a 448 GB/s
class part), then Phase 1 (the queue must stay deep or the kernel win is hidden).

For 128K: run the slab at 6.0e9 with `-c 131072`, or accept 11.3 tok/s at 9.8e9.
The KV type is worth 180-276 MiB and is not the deciding factor.

## 6. Not verified

- The launch-bound model is inferred from the microbenchmark plus the native gap,
  not from a native instrumented count of submissions. It fits to ~2 ms.
- The hit-rate curve at a 6.0 GB slab on a real chat/coding trace was not
  measured; the 128K rows above use the benchmark prompt, whose working set is
  friendlier than a shifting conversation.
- The 128K + 6.0e9 configuration was run to 255 decodes, not to a full 128K
  prefill. Prefill at 131,072 cells was not exercised.
- The VRAM-ceiling mechanism (section 4) is an observation, not a diagnosis.
- No kernel change has been made; Phase 2 is still to be implemented and gated.
