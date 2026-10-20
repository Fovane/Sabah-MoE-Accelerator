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

## 7. Regression triage (Oct 20): the 88.9 ms episode was a low-clock state

**This section replaces an earlier conclusion in this file that blamed WDDM time
slicing among concurrent GPU processes. That was wrong, and the measurements
below are what refute it.**

Same DLL, same context, same slab (9.8e9), same flags as Phase 0. For part of the
night the token ran at 88.9 ms; it now runs at 60.5 ms and is reproducible.

### 7.1 What changed: `ids` per call, nothing else

`SABAH_PHASE` splits each of the 144 calls per token into phases. Comparing three
states of the same binary (driver 32.0.16.1742, `-c 512`, 9.8e9):

| state | ids us/call | fetch us/call | wait us/call | launch us/call | steady ms |
|---|---|---|---|---|---|
| Phase 0 (23:19, same boot) | 243 | 477 | 468 | - | 58.59 |
| the 88.9 ms episode (00:41) | 442 | 479 | 469 | - | 88.47 |
| now (12:00) | 256 | 474 | 464 | - | 61.06 |

`fetch` and `wait` are identical in all three. Only `ids` moved. 144 calls × 186 us
= 26.8 ms/token, which is the whole regression (88.5 − 61.7). `ids` is the router-id
download *and its stream synchronise*, so it absorbs the time the GPU needs to
finish the work queued before it.

### 7.2 The clock state, measured (not inferred)

`nvidia-smi --query-gpu` sampled every 400 ms during a decode; min/median/max over
the steady window. Max SM clock on this board is 3090 MHz, max memory 14001 MHz.

| state | SM MHz | mem MHz | P-state | power | temp | steady ms |
|---|---|---|---|---|---|---|
| idle | 427-547 | 405 | P8 | 8.4 W | 30 C | - |
| the 88.9 ms episode (dmon) | 1342-1590 | 7001 | - | 17-35 W | 45 C | 88.47 |
| now, unpinned | 2730/2842/2850 | 13801 | P1/P3 | 43.5 W | 36 C | 61.07 |
| pinned 2800-3000 | 2812/2827/2842 | 13801 | P1 | 71.9 W | 50 C | 60.67 |
| pinned 1400-1500 | 1410/1485/1492 | 7001 | P3 | 21.5 W | 38 C | 101.72 |

The episode ran at roughly half the SM clock and exactly half the memory clock.
No throttle reason fires (`clocks_throttle_reasons.active` is constant
`0x400` = reliability, not power or thermal; power was 35 W of a 180 W limit,
temperature 45 C of ~90 C).

### 7.3 Pinning the clock old-style, both directions (reversible)

`nvidia-smi -lgc 2800,3000` / `-lmc 13000,14001`, three steady runs each, then
`-rgc` / `-rmc` (restored; idle returned to 442 MHz / P8):

| condition | run0 | run1 | run2 | median ms | tok/s |
|---|---|---|---|---|---|
| unpinned, now | 61.06 | 60.49 | 60.72 | 60.72 | 16.47 |
| pinned near boost | 60.67 | 60.77 | 60.50 | 60.67 | 16.48 |
| pinned at the episode's clock | 101.72 | - | - | 101.72 | 9.83 |
| unpinned, after restore | 60.38 | - | - | 60.38 | 16.56 |

Pinning low reproduces the slowdown and then some (101.7 vs 88.5, because 1485 MHz
is a little lower than the episode's ~1.5 GHz average). Pinning high changes
nothing because the machine is already at boost.

### 7.4 Kernel time vs gap time (CUDA events, since nsys cannot see this DLL)

nsys 2026.3.2 captures kernels for a trivial CUDA binary (80 kernel rows), but for
the diag it records 9 runtime calls and zero kernels: the runtime is loaded with
`LoadLibrary` and CUPTI never sees its launches. (`C:\Temp\nsys\ctx128.sqlite`,
cited as a Phase 0 kernel trace, is in fact a *non-Sabah* run: 121,843 kernels with
no `k_gate_up`/`k_down`, so it was never a Sabah kernel baseline either.)

So `sabah_rt.cu` now has an env-gated probe, `SABAH_LLAMA_EVENTS=1`: each
`k_mul_mat_id` launch is bracketed by a CUDA event pair, and the pairs are read at
the next router drain, which already proves they completed. `host_ms` is the
matching submission time. Times are per decode token (144 calls) at 2842 MHz unless
stated; driver 32.0.16.1742:

| condition | gate/up us | down us | kernel ms/tok | launch ms/tok | steady ms |
|---|---|---|---|---|---|
| 2842 MHz | 189.6 | 207.3 | 28.16 | 2.12 | 61.94 |
| 1485 MHz (pinned) | - | - | 53.04 (368.3 us/call) | 1.96 | 101.83 |

Kernel time scales with the clock (194.8 → 368.3 us/call = ×1.89 against a clock
ratio of 1.91) and host submission does not move (14.7 → 13.6 us/call). That is the
decisive split: **the kernels stretched, the gaps did not.** The `ids` phase grew
because it waits on those longer kernels, not because the host got slower.

### 7.5 What did *not* cause it

Per-process GPU engine utilisation during the current decode
(`\GPU Engine(*)\Utilization Percentage`, steady window, 0.5 s samples):

| process | engine | mean % | max % |
|---|---|---|---|
| llama-sabah (the workload) | 3D | 56.4 | 81.4 |
| llama-sabah | copy | 2.2 | 4.3 |
| OpenCode | 3D | 0.7 | 0.8 |
| System | copy | 0.4 | 0.7 |
| dwm | - | 0 | 0 |

**Nothing else is using the GPU.** dwm is at zero, so the display path is not
involved. (The CPU does have an AMD Radeon iGPU, so moving the monitor to the iGPU
header is possible, but there is nothing to move away from.)

Detached benchmark (runs without this session or the UI; `tools/a5_detached_bench.ps1`,
60 s countdown before the second condition), driver 32.0.16.1742, clocks 2812-2850:

| condition | run0 | run1 | run2 | median ms |
|---|---|---|---|---|
| as-is | 61.23 | 60.85 | 60.99 | 60.99 |
| after 60 s with apps closed | 60.63 | 60.51 | 60.70 | 60.63 |
| clocks pinned high | 60.85 | 60.85 | 60.77 | 60.85 |

Closing applications and pinning the clock both change nothing once the machine is
at boost, which is consistent with 7.5 and 7.2.

### 7.6 Conclusion and the consequence for measurement

- The 88.9 ms episode was the GPU sitting at ~1.45 GHz / 7 GHz instead of 2.84 GHz /
  13.8 GHz. Kernel wall-clock roughly doubles; the host side is unchanged.
- Nothing in the code, the slab, VRAM, the launch path or another process explains
  it; the clock state came and went on its own within one boot session
  (Phase 0 58.6 ms at 23:19, episode 88.5 ms at 00:41, 60.5 ms at 12:00).
- Phase 0's 58.6 ms is therefore not wrong, and neither number is a regression. Any
  benchmark that does not record (or pin) the clock is not reproducible on this
  machine; every table above carries the clock state for that reason.
- Environment in which all of the above was measured: driver 32.0.16.1742
  (9/30/2026), nvidia-smi 617.42, CUDA UMD 13.4, Windows 11 Pro 26300.9711 (26H2),
  KB5127694/KB5121794/KB5127648 (10/14) + KB5127647 (10/13), last boot 10/19 22:37,
  power plan Balanced, HAGS key absent (off), no application clocks set.

## 8. Bit-exact tiling of `k_mul_mat_id` (Oct 20)

`k_mul_mat_id` gives one row to a 128-thread block. With `nsub = k/32 = 80`
(gate/up) only 80 of the 128 threads hold a partial; with `nsub = 20` (down) only
20 do, and the block then costs two `__syncthreads` to add four warp sums of which
three are zero.

The reduction order that must be preserved is, exactly:

1. virtual lane `v` holds `partial_v = 0; for (s = v; s < nsub; s += 128) partial_v += sub32(...)`
   — with `nsub <= 128` this is a single `sub32` or nothing, i.e. `+0.0f`;
2. each virtual warp reduces with `__shfl_down_sync` offsets 16, 8, 4, 2, 1;
3. `r = 0.0f; r += w0; r += w1; r += w2; r += w3` in that serial order;
4. `block_sum` returns `sh[4]`, so the value is warp-sum-ordered, not a tree over blocks.

`sub32` is used unchanged, so its internal FMA contraction is identical by
construction.

### 8.1 Standalone proof (`tools/tile_exact.cu`)

Both reductions are run on identical partials and compared as raw 32-bit words:
20,000 rows × `nsub ∈ {1, 2, 19, 20, 21, 31, 32, 33, 63, 64, 65, 79, 80}`, with
±0.0, denormals, ±inf, NaN and `FLT_MAX` mixed in.

```
nsub= 20 rows=20000  bitwise mismatches=0
nsub= 80 rows=20000  bitwise mismatches=0
TOTAL mismatches=0  (nsub20=0 nsub80=0)
```

**0 mismatches in 260,000 rows.** The earlier 8-lane tile was rejected for 24/256
argmax flips; those flips were rounding from a genuinely different reduction order,
not a bug in that kernel, and this construction does not have them.

### 8.2 Two shapes, both bit-exact

- `SABAH_LLAMA_TILE2=1` — 4 rows per 128-thread block, one physical warp per row,
  the four virtual warps done as four sequential passes with the same tree.
  Warp-uniform early exit for virtual warps entirely past `nsub` (a per-lane break
  would have diverged out of `__shfl_down_sync` with a full mask — that bug was
  caught before it was ever run).
- `SABAH_LLAMA_TILE3=1` — the tree is the contract, but the 32-element dots that
  feed it are independent, so they are computed 8 lanes per row (ten per thread,
  which is where the latency hiding comes from) and their exact bits are staged in
  shared memory by virtual-lane index. A full warp then replays the original tree
  and the original serial warp-sum. 16 rows per 128-thread block. Guarded to
  `nsub <= 96`.

Both are off by default (`git diff` on the timing path is empty unless the env var
is set).

### 8.3 Integration: same tokens, measured speed

`-c 512`, 9.8e9, `-n 256`, driver 32.0.16.1742, clocks 2812-2850 MHz, N=3:

| config | fill s | warm-up ms | steady min/med/max ms | tok/s (med) |
|---|---|---|---|---|
| default | 11.19 / 11.20 / 11.45 | 72.8 / 73.2 / 73.4 | 60.36 / 60.56 / 60.92 | 16.51 |
| TILE3 | 10.75 / 10.81 / 10.81 | 57.98 / 58.25 / 58.71 | 44.05 / 44.21 / 44.35 | 22.62 |

Per-kernel, `SABAH_LLAMA_EVENTS=1`, same clocks:

| config | gate/up us/call | gate/up ms/tok | down us/call | down ms/tok | kernel ms/tok | steady ms |
|---|---|---|---|---|---|---|
| default | 189.6 | 18.21 | 207.3 | 9.95 | 28.16 | 61.94 |
| TILE2 | 186.4 | 17.90 | 190.1 | 9.13 | 27.03 | 60.59 |
| TILE3 | 79.7 | 7.65 | 73.0 | 3.51 | 11.16 | 45.76 |

TILE2's 4% is what the reduction alone is worth; TILE3's 2.5× comes from giving
each thread ten independent dots instead of one, which is the same numbers in a
different schedule.

Bit-exactness of the integration, byte-comparing whole files:

| output | bytes | default vs TILE2 | default vs TILE3 |
|---|---|---|---|
| `fed.i32` | 1,024 | identical | identical |
| `argmax.i32` | 1,024 | identical | identical |
| `logits.f32` | 254,279,680 | identical | identical |

`sabah gate` PASSes on both paths (2000/0 verify, 768/0 id reuse, 0 mapped
fallbacks).

### 8.4 Where the time is now, and the graph re-estimate

Measured per token at 44.21 ms: Sabah 11.16 + dense 12.4-12.8 + other ~3.5
(section 2's split) = 27.1-27.5 ms of GPU, leaving ~16.7-17.1 ms of host/queue
exposure — the same ~15.2 ms host gap Phase 0 measured, now a quarter of the token
instead of a tenth of it.

| step | ms/token | tok/s | basis |
|---|---|---|---|
| Phase 0 (58.6 ms) | 58.60 | 17.07 | measured, section 2 |
| today, TILE3 | 44.21 | 22.62 | measured, 8.3 |
| + CUDA graphs removing the ~12.7 ms host submission | ~31.5 | ~31.7 | 44.21 − 12.7 (Phase 0d's measured host saving), GPU work unchanged |
| + dense at memory peak (12.6 → 9.3) | ~28.2 | ~35.5 | section 5(b), unchanged |

So the graph work, deferred in the last round when the host gap was 21% of a 58.6 ms
token, is now worth ~29% of a 44.2 ms token: it is the difference between 22.6 and
~31.7 tok/s, i.e. it is what stands between TILE3 and the 30-35 tok/s target. It is
still not started, per the instruction to defer it and only re-estimate here.

### 8.5 Not attempted, and why

- The slab/VRAm cliff at `-c 131072` (section 4) is untouched; all of the above is
  `-c 512`.
- No routing change: ids are still downloaded per layer (48 drains/token), the
  reuse group is still 3, and no expert is pruned, substituted or skipped.
- 30-35 tok/s is not reachable by further `k_mul_mat_id` reshaping alone: the MoE
  kernels are 11.16 ms of a 44.21 ms token, so even a free MoE kernel leaves
  ~33 ms and ~30 tok/s only with the host gap removed.

## 9. Clock hygiene: the low state is reproducible (Oct 20)

Section 7 named the low-clock state as the cause of the 88.9 ms episode but could
not reproduce it on demand. It reproduces now, with a 1 s trace across
idle -> decode -> idle (`tools/clock_trace.py`, `results/phase0/clock-trace/`).

Decode, 256 tokens, `-c 512`, 6.0e9, TILE3 on, **unpinned**:

| t s | SM MHz | mem MHz | P | W | note |
|---|---|---|---|---|---|
| 0-15 | 412-472 | 405 | P8 | 8.3 | idle |
| 16-21 | 2617-2857 | 13801 | P1 | 17-32 | ramp |
| 22-29 | 1470 -> 825 | **7001** | **P3** | 20-21 | **downclocked mid-decode** |
| 30-31 | 2392-2422 | 7001 | P3 | 32 | recovering |
| 32-40 | 2400-2527 | 13801 | P1 | 34-38 | partial |
| 41-51 | 2820-2842 | 13801 | P1 | 76-86 | full boost |

The downclock starts ~7 s into a decode that is itself making 13% utilisation:
the host is waiting on expert fetches, the driver sees a low duty cycle, drops
the memory clock to 7001 and the SM to 825, the kernels stretch, and the low
duty cycle persists a little longer. It is a closed loop with the workload, which
is why it looks random between runs and why closing applications and pinning the
process made no difference in A.5.

Pinning results, same trace shape:

| pin | SM min/med/max | mem min/med/max | low samples | verdict |
|---|---|---|---|---|
| none | 412 / 1335 / 2857 | 405 / 7001 / 13801 | 43 / 72 | invalid |
| `-lgc 2800,3090` | 2805 / 2805 / 2842 | **810 / 810 / 13801** | 0 / 59 | **not sufficient** |
| `-lgc 2800,3090 -lmc 13000,14001` | 2805 / 2812 / 2842 | 13801 / 13801 / 13801 | 0 / 53 | holds |

So pinning the SM floor alone is not enough on this card: the memory clock still
fell to 7001 MHz for 7 s (P3), which slows every memory-bound kernel. Both floors
are needed. Commands, restore commands and the idle cost (P1/2805 MHz/~26 W
instead of P8/430 MHz/~8.4 W) are in `docs/CLOCK_HYGIENE.md`.

Machine state during all of the above: power plan **Balanced**, HAGS key
**absent (off)**, display **attached and active**, driver **617.42**, power limit
**180 W** (default 180, max 200), no `PowerMizer*` override. The throttle bitmask
is `0x400` (reliability) in every state, pinned or not, so it is not the trigger.
The driver was not modified.

Every benchmark record now carries `clocks.whole`, `clocks.steady` (SM/mem
P-state/power/throttle/display), `clocks.valid` and `machine`, from
`sabah/tools/clocks.py` (0.5 s sampler). A run is **invalid** when the steady
median is below 2400 MHz SM or 10000 MHz memory; `tools/phase0_measure.py` and
`sabah bench` then exit 4 (`--allow-low-clock` keeps the exit code while still
recording the run as invalid). All tables in sections 9-12 were taken pinned,
and every row below reports the clock that goes with it.

## 10. TILE3 is the default path (Oct 20)

The guard covers the real model: `tools/expert_shape_guard.py` reads the GGUF and
finds 144 expert tensors (48 blocks x 3 roles). Gate/up have k = 2560 -> 80
sub-blocks, down has k = 640 -> 20, with 512 experts, so every tensor is inside
the 96-sub-block limit of the staged tree (`SB_T3_V`). No tensor is near it.

Default is TILE3. `SABAH_LLAMA_TILE3=0` (also `n`/`f`) restores the original
kernel, and `SABAH_LLAMA_TILE3_MAXNSUB=<n>` lowers the guard, which is how the
fallback branch is exercised on a model that is otherwise inside it.

Byte-exactness, whole-file comparison, 1024 tokens per run, pinned clocks
(2805-2857 MHz SM, 13801 MHz memory), `-n 1024`, 6.0e9 slab:

| `-c` | prompt | TILE3 on warm/steady ms | TILE3 off warm/steady ms | reads | `fed`/`argmax`/`logits` |
|---|---|---|---|---|---|
| 2048 | "Hello" | 55.3 / 175.01 | 70.8 / 183.00 | 179.1 GB | identical (1,017,118,720 B) |
| 2048 | merge-lists | 283.5 / 46.18 | 286.1 / 61.07 | 135.7 GB | identical |
| 131072 | "Hello" | 50.0 / 176.02 | 55.4 / 183.16 | 179.1 GB | identical |
| 131072 | merge-lists | 295.0 / 47.39 | 291.9 / 59.99 | 135.7 GB | identical |

`fed.i32` and `argmax.i32` are 4,096 bytes (1024 tokens) and match byte for byte;
`logits.f32` is 1,017,118,720 bytes and matches byte for byte in all four pairs.
The two prompts produce the same read pattern at both contexts (179.1 GB and
135.7 GB exactly), i.e. context length does not change routing here.

What the two prompts say about TILE3: on the merge-lists prompt the tier stays
hot and the token is compute/latency bound, so TILE3 is worth **-24.4%** at
`-c 2048` and **-21.0%** at `-c 131072`. On "Hello" the steady token is I/O
bound (175-183 ms for 175 MB/token, ~1.0 GB/s of expert fetch), so the same
kernel change is worth only -3.9% to -4.6%. TILE3 removes kernel time; it does
not remove fetch time.

Fallback test (`tools/tile3_fallback_test.py`, 128 tokens, `-c 1024`,
`SABAH_LLAMA_EVENTS=1`):

| run | guard | per-call gate/up | per-call down | vs `tile3` | vs reference |
|---|---|---|---|---|---|
| `tile3` | maxnsub 96 (staged tree) | 79 us | 73 us | - | bit-exact |
| `guard16` | maxnsub 16 (**falls back**) | 188 us | 208 us | bit-exact | bit-exact |
| `off` | `TILE3=0` (reference) | - | - | bit-exact | - |

The runtime line from the guard16 run confirms the branch: `SABAH_EV tile2=0
tile3=1 maxnsub=16`, with per-call times back at the reference kernel's 188/208
us. All three runs write byte-identical `fed`/`argmax`/`logits` (127,139,840 B).

`sabah gate` with TILE3 as the default (`-c 4096`, 9.8e9, 2000 tokens): **PASS**,
2000/0 served bytes matched, 192,000/0 reuse checks matched, 0 mapped fallbacks,
0 overflow-affecting checks (5,990 documented over-budget allocations), clocks
valid at 2805-2857/13801, steady 78.4 ms (12.76 tok/s). The 44.2 ms figure in
section 8 was `-n 256`; at 2000 tokens the same slab settles at 78 ms because
the miss rate rises as more distinct experts are touched. Section 11 measures
where that plateau comes from and section 12 re-runs the slab sweep pinned.



