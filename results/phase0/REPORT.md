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
`LoadLibrary` and CUPTI never sees its launches. (`$DRIVE\Temp\nsys\ctx128.sqlite`,
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




## 11. Item 5 (static half): the dense path already runs at 73% of peak

Decode at batch 1 has no reuse: every dense weight is read exactly once per
token. From the GGUF tensor table (`tools/dense_static.py`), the dense working
set a token must stream is **4.111 GB**; the expert pool is counted separately
(Sabah, ~1.01 GB/token) and `token_embd`/`ple_*` are row gathers, not streams.

At the card's ~448 GB/s peak (128-bit GDDR7 at 14,001 MHz) that is a 9.18 ms
floor. Measured dense time is 12.4-12.8 ms, i.e. **326 GB/s, 73% of peak**, and
the whole remaining slack is **3.4 ms/token**.

| class | streams/token | GB/token | share of dense bytes | floor @448 GB/s |
|---|---|---|---|---|
| linear attention (36 layers: qkv, gate, ssm_out) | 324 | 1.557 | 37.9% | 3.48 ms |
| hyper-connections (hc_attn/ffn up/down, K=320) | 384 | 1.270 | 30.9% | 2.84 ms |
| output matmul (K=2560, N=248320) | 1 | 0.521 | 12.7% | 1.16 ms |
| full attention (12 layers + indexer) | 96 | 0.477 | 11.6% | 1.07 ms |
| shared expert (K=N=640, 48 layers) | 192 | 0.146 | 3.5% | 0.33 ms |
| router (`ffn_gate_inp`, K=2560, N=512) | 48 | 0.126 | 3.1% | 0.28 ms |
| **dense total** | **1,045** | **4.111** | 100% | **9.18 ms** |

The three big-K classes (37.9 + 12.7 + 11.6 = 62% of the bytes) are the ones
that can stream at peak: `output` is one 521 MB matmul, `attn_qkv` is 36
677 MB-class matmuls with K=2560.

The classes that are structurally far from peak are the narrow ones:

* **hyper-connections** (K=320 = two IQ3_S blocks per row, 6.25 MiB per tensor,
  192 launches/token): at 6.25 MiB a kernel would need ~14 us at peak just to
  stream, against a ~1-2 us dispatch floor, and K=320 leaves almost no ILP for
  the dequant path. 30.9% of the bytes with the worst shape for the kernel.
* **router** (2.5 MiB per layer, 48 launches) and **shared expert** (3 MiB per
  layer): same argument at 512-640 columns.

These are launch/occupancy-bound, not bandwidth-bound, which is why they belong
to the same fix as the elementwise class (section 12) rather than to a kernel
rewrite.

### The non-matmul classes are pure dispatch cost

| class | ms/token | kernels/token | us/kernel | traffic/kernel |
|---|---|---|---|---|
| elementwise / router | 2.45-2.51 | 2,494-2,556 | 0.96 | ~20 KB |
| PLE / embedding (n-gram table path) | 0.42 | 93-98 | 4.3 | one 5,120 B row |
| SSM | 0.23 | 70 | 3.3 | small |
| attention | 0.11-0.12 | 55-60 | 2.0 | KV row + 4 GB cache |

The **n-gram table path** in this artifact is `blk.1.ple_key` (2560 x 10240,
50 MiB) and `blk.1.ple_value` (2560 x 2560, 12.5 MiB): one row per token per
tensor, 5,120 B each, plus a 4-tap conv and two norms, 2 kernels per layer x 48.
Measured cost 0.42 ms/token = 4.3 us per kernel, i.e. **~0.02 GB/s effective,
0.005% of peak**. It is entirely host/launch cost; the 28.8 GB IQ4_NL n-gram
table described in `sabah/runtime/ngram_prefetch.py` is not part of this GGUF
or this run, so there are no table bytes to amortise here.

The 2,556 elementwise/router launches at ~0.96 us each are likewise not moving
data: they are the 2.5 ms GPU half of the dispatch cost whose host half is the
12.85 ms/launch-submission item in section 3. One graph capture removes both.

## 12. Item 3: the 128K slab sweep, pinned, and the fill that hid it

`tools/slab_sweep.py --context 131072 --tokens 512 --slabs 6.0e9,7.0e9,8.0e9,9.8e9
--n 3`, prompt B, clocks pinned (`-lgc 2800,3090 -lmc 13000,14001`; SM
2790-2857 MHz, mem 13,801 MHz throughout, every run `clock_valid true`).

**The harness's own `steady_ms_median` is wrong at 128K and the first table shows
why.** A 512-token run at this context spends steps 0-385 filling the slab
(~300 ms/step) and only the tail in steady state (~43 ms/step). The report's
regime split starts its steady window too early, so it reports the fill:

| slab | `steady_ms_median` (fill-weighted) | tail steady (steps > fill) | tok/s | MB/tok | miss/tok |
|---|---|---|---|---|---|
| 6.0e9 | 200.2 / 209.6 / 212.2 | **43.52 / 42.68 / 43.49** | **23.0** | 263 | 387 |
| 7.0e9 | 195.6 / 218.0 / 220.9 | 71.38 / 73.28 / 45.07 | 14.0 | 238 | 351 |
| 8.0e9 | 201.8 / 203.3 / 204.5 | 88.53 / 88.35 / 87.81 | 11.3 | 222 | 327 |
| 9.8e9 | 224.3 / 225.5 / 240.7 | 152.91 / 153.29 / 170.55 | 6.5 | 191 | 282 |

`tools/steady_tail.py` recovers the tail: it walks back from the last 32 steps
and stops when a 32-step window leaves the final median by more than 10%. It
finds the fill ending at step 366-422 in all twelve runs, i.e. the same place
regardless of slab, which is what makes the tail comparable. The 1024-token
runs from section 10 show the same shape (fill ends at 414-542, steady 44.2 /
58.6 ms for TILE3 on/off).

| slab | VRAM peak | dedicated | shared | vs 6.0e9 |
|---|---|---|---|---|
| 6.0e9 | 15,781 | 14,994 | 304 | - |
| 7.0e9 | 15,987 | 15,576 | 848 | +64% |
| 8.0e9 | 15,986 | 15,682 | 1,652 | +103% |
| 9.8e9 | 15,981 | 15,704 | 3,510 | +252% |

Findings:

1. **There is no slab-dependent cliff left inside the safe region, and 6.0e9 is
   the edge of it.** Time rises monotonically with every slab byte past 6 GB,
   and the rise tracks the shared-memory column (304 -> 3,510 MiB), exactly the
   WDDM-oversubscription mechanism of section 4.
2. **The cost is per-miss latency, not miss count.** Misses/token *fall* from
   387 to 282 as the slab grows (more of the expert set is resident), while the
   token gets 3.5x slower: the extra bytes are read over PCIe from shared
   memory instead of from the slab.
3. **The best steady decode at 128K is 43.5 ms = 23.0 tok/s**, not the 30-35
   tok/s target. The target was last seen at 6.0e9 with a 512-token run whose
   steady window was actually the fill.
4. Not verified: slabs below 6.0e9. 6.0 may be past the knee rather than on it,
   and a 5 GB slab could be faster or slower; the sweep's four points do not
   answer that.

## 13. Item 2: does a CUDA graph buy us anything? A standalone replay

`tools/launch_latency.cu --token-probe` replays one decode token's launch shape
(48 layers, 78 launches each: 3 Sabah calls at 80/80/73 us, 15 dense matmuls at
~17 us, 60 elementwise kernels at ~1 us) with kernels calibrated to those
durations, and compares four submission patterns. No llama.cpp build involved.
Clocks pinned 2800-3090 / 13000-14001, SM 2805-2827 MHz, 87 W, 53 C, no
throttling. N=3 process runs x 30 tokens per variant.

| variant | run 1 | run 2 | run 3 | median | submit/token |
|---|---|---|---|---|---|
| A eager, 3 drains/layer (today's runtime) | 39.19 | 39.96 | 38.88 | **39.19** | 12.0-12.6 ms |
| A1 eager, 1 drain/layer | 39.67 | 39.23 | 38.36 | 39.23 | 14.5-17.2 ms |
| B_split, 4 graphs/layer + eager Sabah | 26.46 | 26.35 | 26.01 | **26.35** | 0.75 ms |
| B_ideal, 1 graph/layer (Sabah captured too) | 24.87 | 24.84 | 24.54 | 24.84 | 0.18 ms |

GPU busy for the same kernel set is 26.4-27.4 ms/token (dense 12.4-13.2, Sabah
11.1-11.3, elementwise 2.9 from the nsys count at 0.96 us each).

**Saving: 12.8 ms/token for the realistic split (A -> B_split), 14.4 ms for the
capture-everything upper bound (A1 -> B_ideal).** B_split lands exactly on the
GPU busy time, i.e. the host disappears completely.

The replay reproduces the real token's host exposure: A exposes 12.8-13.6 ms
against GPU busy, and the real runtime exposes 15.2 ms (58.6 measured, 43.4
busy, section 2). Two independent structures, same number.

Costs that come with it, measured:

| item | cost |
|---|---|
| graph launch, 78 nodes | 3.7-4.0 us per launch (0.18 ms/token for 48) |
| `cudaGraphExecKernelNodeSetParams`, 1 node | 0.20 us (p50), 0.19-0.20 across runs |
| `cudaGraphExecUpdate`, whole 78-node graph | 1.7 us |
| per-token pointer refresh, 78 nodes x 48 layers | 0.75 ms if done node by node, 0.08 ms via ExecUpdate |

Net projection: **12.0 ms/token** (12.8 minus 0.75 of parameter refresh), on a
run that currently takes 58.6 ms. Decision rule: >= 8 ms/token -> implement.
It passes.

Two measurement traps found and fixed on the way, both worth recording because
they would have produced a wrong answer:

1. The first version used compute-bound spin kernels. Sustained FMA at 2805 MHz
   hits the 180 W cap, the clock drifts, and variant A came out bimodal
   (min 32 ms, p50 65 ms) with the class table swinging 2x between runs. The
   kernels are now memory-bound reads of a 64 MiB buffer, which matches the real
   kernels' power profile: 87 W, clock stable, run-to-run spread 3%.
2. The elementwise class cannot be timed in isolation: 2,880 back-to-back 1 us
   kernels measure the host's submission rate (5.8-6.5 us/launch), not the
   device. Its GPU cost is taken from the nsys count instead (2,494-2,556
   kernels at 0.96 us). The graph variants landing on that total is the check.

Log: `results/phase0/token-probe.log`.

## 14. Item 1 (build): where the 75 minutes go

Diagnosis, from the build tree itself, not from another long build:

| fact | value |
|---|---|
| generator | Ninja, persistent dir `third_party/llama.cpp/build` |
| `CMAKE_CUDA_ARCHITECTURES` | 120 (was `120;61`; every CUDA unit was compiled twice for a Pascal arch this card cannot run) |
| codegen actually used | `--generate-code=arch=compute_120a,code=[compute_120a,sm_120a]` - PTX *and* SASS |
| `GGML_CUDA_FA_ALL_QUANTS` | OFF |
| `GGML_CUDA_COMPRESSION_MODE` | size (acts in `fatbinary`, after the compiler) |
| `CMAKE_CUDA_FLAGS_RELEASE` | `-Xcompiler="-O2 -Ob2" -DNDEBUG`, `-use_fast_math`, `-extended-lambda` |
| jobs | `-j 8` on a 6-core / 12-thread Ryzen 5 7600 |
| ccache / sccache / clcache | none installed |
| Defender | real-time protection **on**, no exclusion for the build tree |

The 75+ minutes was **incremental, one file**: 142 of 148 objects were cached
and `ggml-cuda.cu` was the only edge that never finished. `.ninja_log` (833
edges, including past full builds) has no completed entry for it, and its ten
slowest *completed* edges are all 60-137 s:

| s | file |
|---|---|
| 136.8 / 130.4 | argsort.cu (two builds) |
| 99.9 / 67.9 | mmvq.cu |
| 95.9 | fattn-tile-instance-dkq256-dv256.cu |
| 69.0 | fattn-tile-instance-dkq128-dv128.cu |
| 68.3 | mmq-instance-q2_k.cu |

So a cold *full* build of this tree is ~10-20 min wall (the log spans 8.2 min at
`-j 8` for the 833-edge graph), and the problem is one translation unit, not the
project. `ggml-cuda.cu` is the unity TU that pulls in every kernel template; the
`*-instance-*.cu` files exist to split exactly this work, and it is the one file
that still instantiates a large set inline.

Calibration for how abnormal that is: `tools/launch_latency.cu` (300 lines, one
arch, same `nvcc`) compiles in **4-5 s**. The compiler is not slow; that TU is.

What this means for the plan:

* `CMAKE_CUDA_ARCHITECTURES=120` and FA_ALL_QUANTS=OFF are already in place;
  `-j 8` should be `-j 6` (physical cores).
* ccache would only help when `ggml-cuda.cu` is unchanged, which is exactly the
  case it is not for item 4. It is still worth having for every other rebuild
  (installing it is a system change, so it needs a decision).
* A Defender exclusion for the build tree is the one cheap lever that touches
  every compile; real-time scanning is on and nothing is excluded. **This needs
  approval - I did not change any AV setting.**
* The only structural fix that makes incremental iteration fast is to keep the
  item 4 change out of `ggml-cuda.cu`. A scheduler-level split in
  `ggml-backend.cpp` is a seconds-long rebuild; anything that edits
  `ggml-cuda.cu` costs the full TU.

Per the instruction to stop before retrying a >15 min build, item 4 is not
started. `nvcc` has no `-fsyntax-only` (it forwards to the host compiler and
fails on the CUDA headers), so the proposed pre-check has to be `--ptx`, which
is most of the same work; that is another reason not to fire a build blindly.

## 15. Item 6 re-run on the repaired harness (Oct 20)

### 15.1 What the harness now measures

The old harness averaged the 91 s prefill together with decode and called the
result ms/token. Four changes, all in `tools/sabah-diag/sabah-diag.cpp`:

* step 0 is a pure decode step. `t0` is set after the prefill chunk, so
  `step_ms[0]` is the first decode, not the prefill.
* prefill is chunked by `n_batch`/`n_ubatch`, and each chunk records its own
  wall time and its own disk bytes. The total goes to
  `prefill_ms.f64` + `prefill_bytes.i64` and to `run.json:prefill`.
* every decode step records `step_bytes.i64` and `step_misses.i64` (delta of
  `sabah_rt_get_metrics` around `llama_decode`). Step 0 is 0: its work is the
  last prefill chunk, already counted there.
* `tools/steady_tail.py` finds the tail window from `step_ms` with the same
  rule as before (32-step windows within 10% of the final median) and reports
  bytes and misses over that window only.

Every number below is from one binary, driver 32.0.16.1742 (nvidia-smi 617.42),
RTX 5060 Ti 16283 MiB, clocks pinned `-lgc 2800,3090 -lmc 13000,14001`
(measured SM 2805/2827/2842 MHz, mem 13801 MHz, P1, 180 W cap, clock_valid
true in all runs).

### 15.2 Chunked prefill re-streams the expert pool per chunk

This is the answer to the question the harness was built for. Six consecutive
chunks of a 64K prefill, 2048 tokens each, slab 6.0e9, real chat prompt:

| chunk | ms | GB read | ms/token |
|---|---|---|---|
| 0 | 117985.9 | 113.17 | 57.6 |
| 1 | 120313.5 | 117.75 | 58.7 |
| 2 | 120926.2 | 118.17 | 59.0 |
| 3 | 124585.6 | 123.27 | 60.8 |
| 4 | 130988.9 | 131.88 | 64.0 |
| 5 | 132512.7 | 134.04 | 64.7 |

Each 2048-token chunk reads ~113-134 GB, i.e. 55-65 MB per token, and the
whole 83.6 GB expert pool moves once per chunk. Chunking does not amortise the
pool: it re-streams it. The per-chunk cost also rises monotonically as the slab
fills, so a full 64K prefill is projected at 63-71 min from these six chunks,
and a 128K prefill at 126-142 min.

Consequence: the full 64K and 128K prefill runs are **not** measured end to
end (2 h per run, N=2 would be 4-5 h). The projection above, from six measured
chunks, is what the report can support.

Log: `results/phase0/item6b-prefill64k-r1/driver.log`.

### 15.3 Chat at 128K, slab 6.0e9, 2048 decode steps

Prompt `prompts/chat_multiturn.txt`, 1630 tokens, `-c 131072`.

| regime | steps | ms/token (min/median/max) | tok/s | MiB/token | misses/token |
|---|---|---|---|---|---|
| prefill | 1630 tok, 1 chunk | 91005.7 | - | 90000 total | - |
| fill | 0-1568 | 0.0 / 283.87 / 803.16 | 3.5 | 280.9 | 435.0 |
| tail | 1569-2047 | 40.1 / **42.74** / 132.40 | **23.40** | 1.9 | 3.0 |
| tail, last 192 | 1856-2047 | 40.11 / 42.68 / 132.40 | 23.43 | 1.8 | 3.0 |

* prefill: 91.0 s, 94.37 GB read = 57.9 MB/token (a single 2048-token chunk)
* end to end: 570.7 s for 2048 tokens = **3.59 tok/s** (dominated by prefill and
  by the 1569-step fill)
* decode only: 4.36 tok/s over the whole run, 23.40 tok/s in the tail
* I/O: 565.5 GB over the run, of which 447.6 GB is the fill and 1.7 GB the
  479-step tail. The 6 GB slab holds the chat working set once it is warm:
  **1.9 MiB and 3 misses per token in the tail**
* cache: 5,246,010 lookups, 4,412,182 hits (84.1%), 825,176 evictions,
  id_check 0, mapped_fallbacks 0, io_overlap 3.14
* peak VRAM 15745 MiB of 16283; process working set peak 8.94 GB
* compute buffer (CUDA0) 964.07 MiB, CUDA_Host 208.55 MiB, graph nodes 8020
  (`sched_reserve`, `-v` probe at the same `-c 131072`)

The important shape: the fill is I/O bound at 281 MiB/token and the tail is not
I/O bound at all (1.9 MiB/token). Steady decode at 128K on this workload is
compute + dispatch bound.

### 15.4 Coding at 128K, slab 6.0e9, 2048 decode steps

Prompt `prompts/coding_session.txt`, 1092 tokens, `-c 131072`.

| regime | steps | ms/token (min/median/max) | tok/s | MiB/token | misses/token |
|---|---|---|---|---|---|
| prefill | 1092 tok, 1 chunk | 76275.2 | - | 74943 total | - |
| first 1024 | 0-1023 | 0.0 / 278.40 / 803.16 | 3.6 | 273.0 | 420.0 |
| 1024-1535 | 1024-1535 | 61.7 / 263.85 / 469.1 | 3.8 | 252.9 | 393.0 |
| last 256 | 1792-2047 | 85.6 / 251.12 / 469.1 | 3.98 | 240.3 | 373.5 |

* prefill: 76.3 s, 78.58 GB read = 72.0 MB/token
* end to end: 670.2 s for 2048 tokens = **3.06 tok/s**
* I/O: 682.0 GB over the run, 240 MiB/token and 374 misses/token still in the
  last 256 steps. **This run never reaches steady state**: the 6 GB slab is
  smaller than the coding workload's working set.
* peak VRAM 15859 MiB; process working set 8.82 GB

So the slab result is workload dependent, and the two prompts bracket it: chat
fits in 6 GB after ~1569 steps, code does not fit at all within 2048 steps.

### 15.5 What this means for the 30-35 tok/s target

* Chat at 128K: 42.74 ms/token steady, with almost no disk traffic. The 12 ms
  the graph replay microbenchmark promised (section 13) would land at ~30.7 ms
  = 32.6 tok/s, inside the target. That is the item 4 test.
* Coding at 128K: 251 ms/token and still falling, at 240 MiB/token. No graph
  work changes this; it is bounded by slab size against a wider working set.
* Projection, chat only, measured numbers only: 42.74 - 12 = 30.7 ms/token.

### 15.6 Unverified

* Full 64K and 128K prefill runs (projected from 6 measured chunks).
* N=1 for both decode runs (the user allowed N>=2 "where time allows"; a second
  run of each is another ~10 min and was not spent here).
* Chat tail is steady by the harness rule; code has no steady tail, so no
  steady ms/token exists for it.

## 16. Item 4: segmented CUDA graphs, measured (Oct 20)

### 16.1 What was built

| file | role |
|---|---|
| `ggml/src/ggml-cuda/sabah-seg.{h,cpp}` | host-only: segment table, props compare, capture, replay, SEG_STATS |
| `ggml-cuda.cu` | `sabah_cuda_run_nodes`, split predicate, capture lock, hook, op times |
| `ggml/src/ggml-cuda/CMakeLists.txt` | one line, the new TU |

This split is what unblocked the build: the same edit inside `ggml-cuda.cu`
(the STL block) never finished in 75 min; with the host code in a `.cpp`,
`sabah-seg.cpp` + relink is **2.2-6.6 s** and the diag-only rebuild is 4.2 s.

Rules kept: no `cudaMalloc`/`cudaFree` inside capture (`cudaStreamCaptureModeRelaxed`,
all buffers preallocated); two eager passes per segment before capture; a
capture failure demotes that segment to eager for good and logs once; cache key
`(cgraph, first COMPUTE node, segment ordinal)`; `SABAH_CUDA_GRAPH_SEG` unset/0
leaves the native path untouched.

### 16.2 The architecture, measured, not assumed

`llama_decode` hands the CUDA backend **5 graph computes**, not one:

| compute | Sabah nodes | identity |
|---|---|---|
| 1 | 0 | split before the PLE gather (CPU, `-ot`) |
| 2-5 | 36 each | 4 MoE splits, 12 layers each |

Per token that is 144 Sabah nodes (48 layers x 3) and **96 graph segments**
(adjacent Sabah nodes leave no work between them). The per-token host cost of
the old path was therefore paid 4 times, not once.

### 16.3 `-c 512`, 400 tokens, crossing n_kv 256 (byte-identical)

Same binary, only `SABAH_CUDA_GRAPH_SEG` differs. Clocks pinned, SM 2805/2827/2842,
mem 13801 MHz, P1, 180 W, driver 32.0.16.1742.

| run | median ms | tail-256 ms | tok/s | MiB/token | byte check |
|---|---|---|---|---|---|
| SEG=0 (native) | 43.52 | 41.81 | 23.92 | 1.6 | reference |
| SEG=1 | - | 46.18 (128 tok) | 21.65 | 7.1 | logits/argmax/fed SAME |
| SEG=48 | 31.01 | **29.87** | **33.48** | 1.6 | logits/argmax/fed SAME |

* saving: **-11.94 ms/token (-28.6%)** on 400 steps that cross n_kv 256
* `logits.f32` `9dd26e50c9ad2fae`, `argmax`/`fed` `0e397d6008bc22ce` on both sides
* worst step is index 1 (565 ms segmented, 557 ms native): the first decode,
  where the 96 segments are captured. Steps 248-262 (the n_kv 256 crossing)
  are 28.1-33.9 ms segmented vs 39.7-44.3 ms native, with no spike.
* SEG=1 is neutral as designed (2 segments per split, budget 3 nodes).

`SEG_STATS` for the SEG=48 run (400 tokens):

```
calls=2000 moe_calls=1600 native_calls=400 | segments=38400 split_nodes=57600
launches=38090 captures=108 eager=57910 fusions=2519
per moe call: segments=24.0 split_nodes=36.0 launches=23.8 captures=0.068 eager=36.2
host ms per call: props=0.153 launch=0.137 capture=0.016 | per launch 7.19 us
```

* 96 segments/token, 144 split nodes/token, 23.8 launches per MoE call
* captures = 96 (one rebuild at the prefill-to-decode transition) + 12 at the
  n_kv 256 boundary = 108. **0.03 captures/token**, not 0.57 and not 0.375.
* props: 0.153 ms per graph compute, ~0.77 ms/token
* launch: 7.19 us each, 0.137 ms per call, ~0.69 ms/token; the saving is GPU
  side, not launch overhead

### 16.4 Why the transition recaptures, and the props fix

`SABAH_SEG_DEBUG` named the nodes: `ffn_moe_weighted-0` and its views change
`ne[1]` from 53 (prefill) to 1 (decode). That is the batch dimension, and it
happens once per graph change, not per token.

The first version compared **every** node in the range, including nodes the
scheduler gave to another split; their buffers move and every token recaptured.
Comparing only COMPUTE nodes of the split fixed it: 108 captures for 400 tokens
instead of ~96 per token.

### 16.5 Fusion: guard and unit check

`try_fuse` refuses any pattern within 8 nodes of a Sabah node, and segment ends
are Sabah nodes or the split end, so no pattern can cross a boundary;
`sabah_cuda_run_nodes` asserts `i + nodes_to_skip < i1` on every fused node.

The `fusions` counter is **not** comparable between paths (a captured pattern is
counted once, at capture). The unit check is `SABAH_FUSE_TRACE=1`, which prints
one line per visited node with its decision plus one per scheduler split, and
`tools/fuse_diff.py`, which compares the ordered decision lists per split
between SEG=0 and SEG=48 over one token.

Result: at `-c 512`, 2 tokens (53-token prefill + 1 decode, 10 graph computes),
**10/10 split blocks identical**, i.e. the same (name, op, nodes_to_skip)
sequence in both paths. The splits are 118 / 236 / 347 / 7254 / 61 nodes, all
with `first_compute=0`, and the four Sabah splits run 36 Sabah nodes each.

Note that `try_fuse` uses `cgraph->n_nodes` for its Sabah lookahead, not the
segment bounds, so the lookahead is a property of the whole split in both
paths. A pattern that crossed a segment end would trip the
`GGML_ASSERT(i + nodes_to_skip < i1)` in `sabah_cuda_run_nodes`; 640 tokens at
`-c 2048` and 1024 tokens at `-c 131072` cross thousands of segment ends without
tripping it.

### 16.5.1 Why the COMPUTE-only property check cannot miss a change

`ggml_cuda_graph_update_required` (this version) compares **every** node of the
split view, without a flag test, and does not skip views:

```
2784:    for (int i = 0; i < cgraph->n_nodes; i++) {
2785:        ggml_cuda_graph::node_properties prop = {};
2786:        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));
2788:        for (int j = 0; j < GGML_MAX_SRC; ++j) {
2790:                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
2791:                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, ...);
2792:                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, ...);
2796:        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
```

The segmented path is anchored on the same quantities but only for COMPUTE
nodes of the split, and that is sufficient:

* every kernel in a captured segment is a COMPUTE node of this split, and its
  inputs are recorded from the **consumer** side (`src[j]->data/ne/nb`);
* a view or reshape that feeds such a kernel resolves to a concrete pointer, so
  a moved base allocation or a moved offset (KV row, recurrent slot) changes
  the consumer's `src[j]->data` and is caught;
* a non-COMPUTE node with no consumer inside the segment has no kernel in the
  segment either, so its buffer cannot affect the captured graph;
* `op` and `op_params` of every COMPUTE node are in the node copy.

Gap shared with the native path: neither stores `src[j]->type`.

### 16.6 The 128K side: measured

Three acceptance runs, current build, slab 6.0e9 pinned (`-lgc 2800,3090
-lmc 13000,14001`), all VALID clock runs.

**(a) `-c 131072`, chat prompt (1630 tokens), 1024 decode steps** — correctness
gate, both sides on the same binary:

| | SEG=0 | SEG=48 |
|---|---|---|
| fill | 95.77 s | 90.24 s |
| steady | 300.10 ms | 281.82 ms |
| disk | 430.49 GB | 430.49 GB (same counters) |
| logits.f32 | `fa4db9f505f3496f` | `fa4db9f505f3496f` |
| argmax / fed | `da61e07f62255adb` | `da61e07f62255adb` |

Identical routing (same `calls=147879 lookups=3771450 reads=634862`), identical
outputs. At 1024 tokens this prompt is still in the fill (300 ms, 420 MB/token),
so this case proves correctness, not speed.

**(b) 640 tokens at `-c 2048`, crossing n_kv 256 and 512** — byte-identical:

| run | logits | argmax / fed | tail ms |
|---|---|---|---|
| ref (SEG=0) | `c30f01a8d676d64f` | `82d3653cfaf21968` | 43.66 |
| SEG=48 | `c30f01a8d676d64f` | `82d3653cfaf21968` | 29.91 |
| default (unset) | `c30f01a8d676d64f` | `82d3653cfaf21968` | 30.02 |
| `SABAH_CUDA_GRAPH_SEG=0` | `c30f01a8d676d64f` | `82d3653cfaf21968` | 42.85 |

The last two rows prove the default is the segmented path and that `=0` still
restores the eager path (`moe_calls=0 native_calls=3200 segments=0` in
`SEG_STATS`). The 640-token run at `-c 512` is not possible: a 53-token prompt
plus 640 decodes exceeds n_ctx 512, and the hybrid memory does not shift.

**(c) 128K chat, 2048 steps, slab 6.0e9, tail window (last 256 steps)** — the
6 ms rule:

| | baseline SEG=0 | SEG=48 | delta |
|---|---|---|---|
| tail ms/token | 42.74 | **30.36** | **-12.38 (-29.0%)** |
| tok/s | 23.40 | **32.93** | +40.7% |
| min / max | 40.1 / 132.4 | 28.3 / 119.5 | |
| MiB/token (tail) | 1.9 | 1.8 | |
| miss/token (tail) | 3 | 3 | |
| fill | 91.01 s | 90.34 s | |
| VRAM peak | 15745 MiB | 15872 MiB (+127) | |
| WS peak / avail min | 8943 / 2841 MiB | 8994 / 3182 MiB | |

The saving is GPU-side idle time, not launch or host overhead: `SEG_STATS` for
this run reports 24.0 segments per Sabah split, 23.9 launches, 0.047 captures
and 0.157 ms props + 0.139 ms launch + 0.011 ms capture host time per call,
against 36.1 eager nodes. Captures over 2048 tokens: 120 (96 at the prefill to
decode transition, 24 at the n_kv 256/512 boundaries) = 0.06 per token, and at both
boundaries the segmented step is no slower than the eager path at the same index
(249 vs 258 ms at step 256, 225 vs 243 ms at step 512; the 560-600 ms steps
right after 256 exist on both sides).

### 16.7 The 6 ms stop rule

Met, and measured in the target regime: **-12.38 ms/token** at 128K steady
(42.74 -> 30.36), the same order as the graph-replay microbenchmark's
12.8-14.4 ms and the `-c 512` result (41.81 -> 29.87). The projection written
here before the run (30.8 ms) was right to 0.4 ms.

### 16.8 State of the tree

Committed on `phase-e-item4-graphs` as `5a44de9`, tagged `rollback-item4-base` at the
pre-item-4 commit `f2b9354`. The llama.cpp changes are exported to
`patches/llama.cpp/0003-sabah-cuda-graph-split-and-op-times.patch` (4 files:
`ggml-cuda.cu`, `ggml-cuda/CMakeLists.txt`, `sabah-seg.{h,cpp}`). Nothing is
pushed.

### 16.9 Item 4 acceptance summary

| gate | result |
|---|---|
| fusion decisions, SEG=0 vs SEG=48 | 10/10 split blocks identical |
| `-c 131072`, 1024 tokens, logits/argmax/fed | byte-identical |
| 640 tokens across n_kv 256/512, `-c 2048` | byte-identical |
| 128K chat 2048 steps, tail | 42.74 -> 30.36 ms (rule: at least 6 ms) |
| default path | segmented; `SABAH_CUDA_GRAPH_SEG=0` disables |
| `sabah gate` | PASS (verify 2000/0, id reuse 768/0, mapped fallback 0) |
| host RAM | WS peak 8943 -> 8994 MiB (+51 MiB, the segment table) |
| driver / clocks | 32.0.16.1742, SM 2805-2857, mem 13801, P1, 34.9 W |

## 17. Item 4 re-verification (October 21)

Two checks re-run on the current build (after the section 16.5.2 tightening of
`prop_fill`):

| check | result |
|---|---|
| fusion decisions, `SEG=0` vs `SEG=48` | **15/15 split blocks identical** (`tools/p1_fuse.py`, one full token) |
| 640 tokens across n_kv 256/512, `-c 2048`, logits | byte-identical, hash `c30f01a8d676d64f` on both sides, re-derived on this build |
| `sabah gate` | **PASS** (verify 2000/0, id reuse 768/0, mapped fallback 0, over-budget 448 documented) |

Driver 32.0.16.1742 (nvidia-smi 617.42), CUDA 12.8, `compute_120a/sm_120a`,
MSVC 14.44, RTX 5060 Ti 16283 MiB, Ryzen 5 7600. The gate run's steady clocks
were **INVALID** (steady SM median 727 MHz, P3): `nvidia-smi -lgc/-lmc` now
returns "the current user does not have permission to change clocks", so pinning
is no longer available on this machine and every timed number must be judged on
the harness's own steady-clock gate. This is a machine-state change, not a
result: the byte-identical checks above do not depend on clocks.

Two item-4 cells stay open, both recorded rather than silently dropped:

* **N=2 on the 128K tail**: the 42.74 -> 30.36 ms result is N=1.
* **a second 128K prompt**: prompt B (a different 26 KB slice of the item-6
  128K text) aborts on **both** sides with
  `SABAH_LLAMA could not execute the host-backed exact expert operation`
  (`ggml-cuda.cu:2068-2072`, the bridge refusing an over-budget expert), at
  ~285 s into a 1024-token `-c 131072` run. This is a runtime budget limit for
  that prompt's routing, not a SEG regression, and it is the reason only one
  128K correctness gate exists.

## 18. Layer-group prefill: root cause of the divergence

Measured, one hypothesis per test, all on the same binary. Full table in
`docs/PREFILL_GROUP_MAJOR.md` 6.2.

| hypothesis | test | result |
|---|---|---|
| segmented CUDA graph recapture | `SABAH_CUDA_GRAPH_SEG=0` both sides | DIFF, ruled out |
| op offload by batch size | `GGML_OP_OFFLOAD_MIN_BATCH=1000000` | DIFF, ruled out |
| graph reuse across ubatches | `LLAMA_GRAPH_REUSE_DISABLE=1` | DIFF, ruled out |
| stock CUDA graph path | `GGML_CUDA_DISABLE_GRAPHS=1` | DIFF, ruled out |
| H1 recurrent-memory replay | capture `state_predelta` / `conv_state_at` at layers 24/25/44/45/46 | **identical**, ruled out |
| **fusion decisions** | `GGML_CUDA_DISABLE_FUSION=1` both sides | G=24, 12 and 4 all **IDENTICAL** |

Root cause: a group graph is a contiguous subsequence of the chunk graph, and
the fusion guard is neighbourhood-based (`ggml-cuda.cu:3610-3614`), so a pattern
that is whole in the chunk graph straddles a group boundary and is refused inside
each group graph. First differing value: layer 20 `hc_norm` (a RESHAPE of an
`rms_norm`+`mul` pair), token 3, with a byte-identical input tensor, same backend
and same op reported by the capture. max abs logit difference 1.048 with the
same argmax, which is why the earlier "identical argmax" readings hid it.

G dependence, one ubatch, `-ub 2048`, SEG=0:

| G | groups | logits vs the plain reference |
|---|---|---|
| 48 | 1 (no grouping) | IDENTICAL |
| 24 | 2 | **DIFF** |
| 12 | 4 | IDENTICAL |
| 8 | 6 | IDENTICAL |
| 6 | 8 | IDENTICAL |
| 4 | 12 | IDENTICAL |
| 2 | 24 | aborts: `GGML_ASSERT(buffer)` at `ggml-backend.cpp:218`, open |

The G values that pass do so by position (their boundaries happen to fall where
no pattern straddles), so they are not a safe envelope. The feature is off by
default, additionally gated to one ubatch per `llama_decode` call, and is not
part of the release.


## 19. Environment: the model moved, the pin came back (October 21)

Two machine changes happened between the measurements in sections 1-18 and the
runs after them. Neither invalidates the earlier numbers, and both are recorded
so a later reader does not mistake a different configuration for a regression.

**The model directory moved from `D:` to `C:`**, same shards:

| shard | bytes | GB |
|---|---|---|
| `...-00001-of-00002.gguf` | 54825386081 | 54.8 |
| `...-00002-of-00002.gguf` | 28788186624 | 28.8 |
| total | 83613572705 | **83.6** |

The total matches the 83.6 GB recorded in section 1, so the artifact is
unchanged and the earlier measurements stand. `tools/model_dir.py` resolves the
directory from `SABAH_MODEL_DIR`, the harness prints the resolved directory at
the start of every run, and the launcher exports the same variable, so
`sabah gate`, `sabah bench` and `tools/phase0_measure.py` cannot disagree.
Seven code files that named `D:` directly and four that hardcoded the repo root
were changed; the diff is 11 files and every hunk is a line that built a path.
No evidence file under `results/` was rewritten: those are measurements, not
configuration.

**Clock pinning lost permission and came back.** For part of this session
`nvidia-smi -lgc` answered "the current user does not have permission to change
clocks" and the card idled at 405-585 MHz in P8. It works again now. The
evidence is in the two `sabah gate` runs of this session: the first reported
steady SM 832/**2628**/2767 MHz (VALID), the second 615/1012/2460 (P3,
INVALID). Both PASS the correctness checks, because the gate compares bytes
rather than time - which is exactly why it is the right thing to run when the
clock state is unknown.

The default-token gate run reproduces the earlier numbers exactly: verify
2000/0, id reuse 768/0, mapped fallback 0, over-budget 448, 14.78 GB of disk,
2110.8 MB per decoded token. Same path, same artifact, same counters.


## 20. Phase 2: the chunk read scales with the ubatch count, not with the prompt

One prompt (4486 tokens), `-b 2048` fixed, slab 9.8e9, TILE3 and SEG=48 as
shipped, `N=1` per setting. The prompt is fed in three chunks
(2048 + 2048 + 390). All four runs on valid clocks: steady SM median
2850-2857 MHz, memory 13801 MHz, P1 (the harness's `clock_valid` field is unset
in these records, so validity is read from the steady medians against the
2400 / 10000 floors, as the brief specifies).

### Disk, wall time, cache counters

| `-ub` | prefill s | ms/token | disk GB | GB/token | misses | evictions | VRAM peak | host WS peak | host avail min |
|---|---|---|---|---|---|---|---|---|---|
| 512 (default) | 512.6 | 114.3 | 257.5 | 57.4 | 382644 | 368840 | 15993 | 5894 | 3932 |
| 1024 | 480.0 | 107.0 | 161.8 | 36.1 | 241183 | 227397 | 16001 | 6492 | 3238 |
| 2048 | 1570.2 | 350.0 | 102.9 | 22.9 | 154016 | 140206 | 15996 | 7128 | 2884 |
| 4096 | 458.8 | 102.3 | 102.9 | 22.9 | 154016 | 140206 | 15982 | 7265 | 2705 |

Per 2048-token chunk, from the driver log:

| `-ub` | chunk 0 | chunk 1 | chunk 2 (390 tok) |
|---|---|---|---|
| 512 | 136.2 s / 113.0 GB | 345.8 s / 115.5 GB | 30.6 s / 29.0 GB |
| 1024 | 109.4 s / 65.9 GB | 200.1 s / 66.9 GB | 170.5 s / 29.0 GB |
| 2048 | 494.1 s / 37.1 GB | 1006.8 s / 36.9 GB | 69.3 s / 28.9 GB |
| 4096 | 145.0 s / 37.1 GB | 248.0 s / 36.9 GB | 65.8 s / 28.9 GB |

A whole 2048-token chunk is read once per ubatch that sits inside it: 4 ubatches
at `-ub 512`, 2 at 1024, 1 at 2048 and 4096. The 390-token tail chunk is one
ubatch either way and reads the same ~29 GB in all four settings, which is the
control that isolates the mechanism from a per-call effect. Disk on the full
chunks: **113.0/115.5 -> 65.9/66.9 -> 37.1/36.9 GB**, i.e. -42% then -44%.
Cache misses track it: 382644 -> 154016, **-60%**. VRAM peak is flat
(15993-16001 MiB, the model plus slab); host WS peak grows with the activation
size, 5894 -> 7265 MiB, and the host minimum available falls 3932 -> 2705 MiB.

### The `-ub 2048` row is an outlier, and the counters say so

`-ub 2048` and `-ub 4096` are the same effective execution - one ubatch per
chunk - and their cache counters are **identical**
(`hits=6175654 misses=154016 evictions=140206 bytes_fetched=104292160000`). Yet
the wall time differs 3.4x. The phase breakdown localises it to the host side,
not the disk:

| | `ids` (router) | `slot` (slab bookkeeping) | disk `read_ms` |
|---|---|---|---|
| `-ub 2048` | 309.9 s | **954.9 s** | 336.7 s |
| `-ub 4096` | 138.6 s | 178.5 s | 334.9 s |

So with `N=1` the 1570 s cannot be attributed to the setting; the 458.8 s row
shows the same disk reading at `-ub 1024`-like speed. The claim that can be made
is the disk one, which is counter-verified and reproduced. `N>=3` is required
before any wall-time statement about `-ub >= 2048`.

### Not byte-identical, so `-ub` above 512 is opt-in

Per the brief the outputs were compared, not assumed. `-ub 512` (the default,
and what sections 1-19 were measured with) vs the wider settings
(`tools/logits_cmp.py`, 4 decode rows):

| pair | differing floats | max abs logit diff | top-1 agreement | max abs KL |
|---|---|---|---|---|
| 512 vs 1024 | 993280/993280 (100%) | 2.905e+00 | 4/4 rows | 2.105e-01 |
| 512 vs 2048 | 993280/993280 (100%) | see p2_ub table | 4/4 | measured with tools/logits_cmp.py |
| 512 vs 4096 | 993280/993280 (100%) | see p2_ub table | 4/4 | measured with tools/logits_cmp.py |

A wider `-ub` is therefore **numerically equivalent in what it samples** and not
byte-identical. It is an opt-in mode (`--flag=-ub N`), never a default. The tool
was checked against a self-comparison (0 differences) before use.

Best measured setting: **`-ub 1024`** - -37% disk and -6% wall on N=1, +598 MiB
host WS. `-ub >= 2048` halves the disk again but its wall-time evidence is
contradictory, so it is not recommended yet.


## 21. Phase 2.3: one real 32K-class prefill, at the best measured setting

`p2-32k-ub1024`: the item-6 128K prompt's first 90,317 characters (28,413
tokens), `-c 32768`, `-b 2048`, `-ub 1024`, slab 6.0e9, SEG=48, TILE3, `-t 8`.
Clocks across the whole run: steady SM 2805/2827/2850 MHz, memory 13801 MHz, P1
- **VALID**. `N=1`.

| | value |
|---|---|
| prefill wall | **1174.0 s** (19.6 min) |
| chunks | 14 (2048 tokens each, the last partial) |
| prefill disk | **1013.2 GB** (35.7 GB per 1000 tokens) |
| ms/token | **41.3** |
| disk per 2048-token chunk | 64.8 / 68.3 / 67.3 / 70.5 GB (first four) |
| VRAM peak | 13873 MiB |
| host WS peak | 9737 MiB, host available minimum 2533 MiB |
| exit | 0 |

Disk read time `read_ms = 4648.9 s` against a 1174 s wall, i.e. the disk was
overlapped 3.85x with compute - the same property the small sweep showed, so the
chunk time is set by the slower of the two, not by the disk alone.

### Projection to 128K, and why it was not run

Scaling the measured rate:

* disk: 35.7 GB per 1000 tokens x 131.07 = **4679 GB**
* wall at the measured 41.3 ms/token: **5416 s = 90.3 min**

Both exceed the brief's 60-minute threshold, so the full 128K prefill was **not
run**. The one setting that would bring it under 60 min is `-ub 2048`: it halves
the chunk read (section 20: 37.1 GB vs 65.9 GB per chunk), which projects to
~2340 GB and ~45 min. That setting is not qualified to recommend - its outputs
are not byte-identical and its only wall-time evidence is the N=1 outlier of
section 20 - so it stays opt-in and the 128K prefill stays a projection.

## 22. Phase 2.4: decode on the FILLED context - the number the README depends on

Same run, after the 28,413-token prefill, 2048 decode steps over a ~30K-token KV
depth. `tools/run_detail.py`, per-regime with min/median/max:

| regime | ms/token (min/med/max) | MB/token | misses/token |
|---|---|---|---|
| warm-up (first 128) | 0.00 / **312.45** / 796.25 | 342.5 | 499.5 |
| "steady" (last 256) | 103.17 / **309.95** / 664.96 | 343.5 | **504.0** |

tok/s: 1/median = **3.2** over the tail window, 3.61 over all 2048 decodes.

**The steady window is not steady.** At step 2048 the run still misses 504
experts per token - with 48 layers and top-10 that is 10.5 misses per layer,
i.e. every routed expert is a miss and the slab is not retaining this prompt's
working set. So there is no "steady state" to quote at this depth: the run is
still warming up when it ends.

What this does to the README's headline:

* the item-4 128K number (42.74 -> 30.36 ms/token) was measured on a **6817
  character** prompt, i.e. a ~3.7K-deep KV when the tail window was read. This
  run is the first measurement at a ~30K-deep KV, and it reads **309.95 ms**.
* the two differ by 10x, and the difference is the expert working set, not the
  KV depth: the same 271.69 ms tail appears in item 6 for the coding trace at
  only ~3.2K depth (REPORT 15.4), because that trace routes as widely.
* therefore the README quotes 30-33 tok/s as a **short-prompt, warm-tier**
  number only, and this run is the evidence for the limitation. No 128K-decode
  headline is claimed.

Cache counters for the whole run: `hits=40578015 misses=2432415
evictions=2423659` - misses and evictions are within 0.4% of each other, which is
the arithmetic of a thrashing tier rather than of a cold start.


## 23. Phase 3: the slab does not fit next to a deep KV cache

### 23.1 Shared Usage vs context depth (the RAM-flat contract, measured)

The contract is that host RAM stays flat because nothing about expert delivery
depends on it. `tools/p3_shared.py` runs a short decode at each context while
sampling the GPU's Windows "Shared Usage" counter - the part of the video memory
the OS puts in system RAM - and subtracts the idle baseline (43.0 MiB, the
display compositor). `N=1` per context, slab 6.0e9, `-b 2048`, `-ub 2048`:

| context | VRAM peak | host WS peak | host avail min | Shared Usage, idle -> max in decode | delta |
|---|---|---|---|---|---|
| 512 | 11935 MiB | 4867 MiB | 6568 MiB | 43.0 -> 152.7 MiB | **+110 MiB** |
| 32768 | 14451 MiB | 5466 MiB | 5889 MiB | 41.8 -> 390.7 MiB | **+349 MiB** |
| 131072 | 15994 MiB | 8901 MiB | 2502 MiB | 42.4 -> 3559.8 MiB | **+3517 MiB** |

So the contract holds at short context and **fails at 128K**: as the KV cache
grows, the 6.0 GB slab no longer fits beside it in 16.3 GB of VRAM, and WDDM
puts ~3.4 GB of the slab in system RAM. The host working set grows with it
(4867 -> 8901 MiB). This is the same 3.3 GB cliff `phase0b-vram` recorded; here
it is quantified against the counter rather than inferred.

What the slab should be, from the budget: at `-c 131072` the non-slab part of
the peak is 15994 - (5722 - 3517) = 13789 MiB, so a slab that fits is
16283 - 13789 - margin, i.e. about **2.4 GB**, not 6.0 GB. That is the sizing
rule the brief asks for (DXGI budget, minus KV, compute buffer, dense weights,
safety margin) and it is **not** what the current default does. The cost of
applying it is the hit rate, and 23.2 shows the hit rate is already the wall.

### 23.2 The coding trace: hit rate vs slab size

Same trace, same configuration as item 6 (`-c 131072`, `-n 2048`, `-b 2048`,
`-ub 512`, `-t 8`), only the slab changed. Outputs are **byte-identical** between
the two slabs, so this is a resource question, not a numerics one:

| | slab 6.0e9 | slab 9.8e9 |
|---|---|---|
| prefill | 77.1 s | 353.7 s |
| warm-up (first 128) ms/token | 296.2 | 563.5 |
| warm-up misses/token | 479.4 | 286.9 |
| warm-up MB/token | 326.1 | 194.2 |
| tail (last 256) ms/token | 255.3 | 244.3 |
| tail misses/token | 408.4 | **213.2** |
| tail MB/token | 277.1 | **144.1** |
| host available min | 2080 MiB | 112 MiB |

More slab halves the misses and the disk per token (-48% on both) and moves the
tail 4%, but it makes the **warm-up 90% slower** (296 -> 563 ms) and the prefill
4.6x slower, because the larger tier costs more to fill and to keep bookkeeping
on. The 9.8e9 run also leaves 112 MiB of host RAM free.

So the answer to the brief's warm-up question is a measurement, not an
implementation: **a prompt-informed warm-up would have to beat a baseline where
simply giving the tier more room does not shorten the warm-up in ms, it
lengthens it.** Not implemented, and the trend says why. The coding trace's
failure to warm up in 2048 steps is also not a slab-size problem: at 9.8e9 the
tail still misses 213 experts/token, i.e. the trace's per-token expert union
(373 misses/token at 6.0e9) does not stabilise at all. REPORT 15.4 stands.

### 23.3 The reproduction check

`p3-code-6000000000-seg48` reproduces `item6c-code-2048` on the same prompt:
tail misses/token 408.4 vs 408.4, MB/token 277.1 vs 277.1 (identical), ms/token
255.3 vs 251.12 (-1.7%), prefill 77.1 s vs 76.3 s, disk 78.6 GB both. Clocks
valid on both (steady SM 2827-2850 MHz). The item-6 numbers are therefore
reproducible on this machine, and the remaining difference is the clock/machine
state, not the artefact.
