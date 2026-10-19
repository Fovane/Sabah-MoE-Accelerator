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

The mechanism is now measured. Windows exposes both halves of the footprint
(`\GPU Adapter Memory(*)\Dedicated Usage` and `\Shared Usage`, plus the same
counters per process); sampling them through the run (`tools/gpu_mem_probe.ps1`)
shows that once dedicated usage pins at the ceiling, every further slab byte is
backed by **shared system memory**, and the slowdown follows it:

| `-c` | slab | our dedicated | our shared | steady |
|---|---|---|---|---|
| 512 | 9.8e9 | 14,894 MiB | 110 MiB | 58.3 ms |
| 131072 | 6.0e9 | 14,916 MiB | 304 MiB | 58.2 ms |
| 131072 | 7.0e9 | 15,479 MiB | 750 MiB | 63.3 ms |
| 131072 | 8.0e9 | 15,471 MiB | 1,824 MiB | 71.5 ms |
| 131072 | 9.8e9 | 15,469 MiB | 3,623 MiB | 88.4 ms |

Dedicated usage saturates at ~15.47 GB (adapter total 15.99 GB of 16,283 MiB);
the allocated-but-not-resident remainder of the slab is placed in shared memory
and read over PCIe. That is the 188 -> 465 us `k_mul_mat_id`. It is WDDM
oversubscription, not driver bookkeeping: the per-process `shared` counter
carries exactly the excess slab budget (9.8e9 -> 6.0e9 is 3.62 GB, and shared
falls by 3.32 GB).

Other processes hold 728-731 MiB of dedicated and ~17 MiB of shared (28 GPU
instances, all small; the desktop baseline is ~540 MiB). Closing them would raise
the dedicated ceiling by ~0.73 GB, which buys ~0.7 GB of slab at 128K, i.e. it
moves the safe slab from 6.0e9 to ~6.7e9. It cannot remove the 3.3 GB of shared
memory while the slab stays at 9.8e9.

So the safe rule for 128K is: the **whole footprint** (model 3,983 + slab +
compute 976 + KV 384 + context) must fit in ~15.4 GB, not the slab alone. At
`-c 131072` that is a slab of ~6.0e9; the table above is the measured penalty for
exceeding it.

## 5. Decision and projections

GPU busy is 74%, so **Phase 2 (kernel efficiency) is the primary lever**, per the
plan's own rule. But the plan's rule prices only the GPU side; this measurement
shows a second, equally mechanical lever. Arithmetic on the measured numbers
(43.4 GPU = 27.3 Sabah + 12.6 dense + 3.5 other; 15.2 host; 58.6 total):

| step | GPU | host | ms/token | tok/s |
|---|---|---|---|---|
| today | 43.40 | 15.20 | 58.60 | 17.07 |
| (a) host submission removed, queue stays deep | 43.40 | ~1.0 | 44.40 | 22.52 |
| (b) Sabah gate+up+down halved (27.3 -> 13.65), host still additive | 29.75 | 15.20 | 44.95 | 22.24 |
| (a)+(b) | 29.75 | ~1.0 | 30.75 | 32.52 |
| (a)+(b)+dense at memory peak (12.6 -> 9.3) | 26.45 | ~1.0 | 27.45 | 36.43 |

So halving the Sabah kernels is worth 22.2 tok/s on its own only if the host gap
stays additive, and 32.5 tok/s once the queue stays deep. The host fix and the
kernel fix are multiplicative, not alternatives, and the order matters: (a) is
cheap relative to its 5.5 tok/s and it un-hides (b).

Dense matmul (llama.cpp's own): the non-expert matmul weights on CUDA0 are
**3,979 MiB** (measured from the GGUF; the token embedding is a row gather and is
excluded), i.e. 4.17 GB read per token. In 12.6 ms that is **331 GB/s, 74% of the
card's 448 GB/s**. It is bandwidth-bound already; the remaining 26% is worth
3.3 ms, and the real lever there is more positions per weight read (MTP or
self-speculation), not a faster kernel. That is not in the current plan.

Next, in order: (a) per-layer graph segments to remove the exposed submission,
(b) the `down` kernel increment in parallel, (c) device-side residency with the
drain kept behind an env var.

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
- The VRAM-ceiling mechanism is measured at the counter level (section 4:
  oversubscription into shared system memory), not inside the driver.
- No kernel change has been made; Phase 2 is still to be implemented and gated.
