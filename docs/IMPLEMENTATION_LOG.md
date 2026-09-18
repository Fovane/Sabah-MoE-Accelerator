# Sabah v5 — Implementation Log

## Session 1 — Phase A (core portability)

Development machine: 1× RTX 4050 Laptop 6 GB, 29.8 GB RAM, Windows 11,
CUDA 13.3, MSVC 14.44. **No multi-GPU machine available.**

### Built and verified

| # | thing | verification |
|---|---|---|
| 1 | `core/model_inspector` | run against the real 111.335 GB artifact; reproduces v3's byte-exact numbers |
| 2 | `hardware/profile` (versioned, v2) | measured this machine end to end |
| 3 | `hardware/mgpu_qualify.cu` + `--json-out` | rebuilt, emits schema-1 JSON, smoke-tested |
| 4 | `hardware/expert_pipe_bench.cu` | carried over from v4, still builds and runs |
| 5 | `planner/planner` | 9 synthetic machine classes, all invariants hold |
| 6 | `tools/cli` | `doctor`, `inspect`, `qualify`, `plan`, `status` all run |
| 7 | `tests/test_planner` | passes, and caught a real estimator bug |

### Bugs found during bring-up (all fixed)

1. **RAM reported as 0 GB.** `wmic ComputerSystem get TotalPhysicalMemory`
   returns nothing under this shell. Switched to PowerShell with wmic as a
   fallback.
2. **Storage measured at 0.22 GB/s.** The path given to Sabah is shard
   *00001*, which for this artifact is a 10.9 MB metadata shard — the timing
   was dominated by `open()`. Now the largest shard in the set is chosen, and
   files under 64 MB are skipped entirely. Re-measured: 1.79–2.51 GB/s.
3. **Batch cost modelled as single-token cost.** The estimator applied the
   measured reuse factor `R_B` but never multiplied expert bytes by `B`, so a
   pass carrying 16 tokens was priced like one token. Projections were absurd
   (B=16 → 311–448 tok/s). Fixed: weights are shared across a pass, arithmetic
   is not. After the fix the sweep saturates at 32.7–47.1 tok/s and per-stream
   throughput falls 25.8 → 2.5, which is the expected fetch-bound shape.
   **This was caught only because the synthetic sweep was printed and read.**
4. `cudaEventCreate` shadowing in the `CK` macro, and `\n` escapes mangled by
   heredoc patching — both compile-level, fixed.

### Measured on this machine

```
RAM            29.8 GB total / 25.5 GB usable / ~14 GB/s streaming read
VRAM           6.44 GB total / 4.89 GB usable
H2D            solo 12.76 GB/s | simultaneous 12.79 | contention 1.00 (one device)
PCIe           gen4 x8 (idle state reads gen1 and ramps; trust the H2D figure)
storage        1.79-2.51 GB/s at the model path
```

### Planner verdict for this machine, on the real artifact

```
execution  : STORAGE_BACKED
expert tier: 0.00 GB
ram bank   : DOES NOT FIT (77.0 GB in 25.5 GB usable)
PROJECTED  : 1.6-2.4 tok/s   vs reference estimate 1.4 tok/s
```

The 4.83 GB fixed path plus ~0.94 GB of KV/workspace does not fit in 4.89 GB of
usable VRAM at context 8192, so there is **no** expert capacity on this GPU.
This is the correct answer, and the planner says so rather than forcing a plan.

---

## What still requires real multi-GPU hardware

None of these can be faked, and none of them are claimed:

1. **Simultaneous multi-GPU H2D aggregate and the contention factor.** The tool
   exists and is correct on one device; the number that matters is a
   three-device number. The whole ~46.5 tok/s projection rests on the three
   cards delivering roughly 32 GB/s *together*, and a board that gives
   11 GB/s each but 18 GB/s together changes the plan completely.
2. **Any measured speedup.** Everything in v5 is a projection.
3. **The GPU hot-tier runtime end to end** (Phase B), and therefore
   `gpu_wait_expert_ms`, the primary KPI.
4. **The M1 forced-resident vs streamed A/B**, which isolates the streaming
   penalty from kernel and graph overhead.
5. **Fixed-path sharding (M2)**, whose inter-GPU activation traffic has never
   been priced in any Sabah phase.

## Not yet built (honest gaps against the v5 brief)

- `runtime/` — RAM bank, GPU hot tier, H2D scheduler, expert executor, KV manager
- `server/` — the OpenAI-compatible endpoint, streaming, sessions
- `adapters/` — no third-party application has been tested, so none is claimed
  compatible
- correctness harness against the reference path (router IDs, expert outputs,
  logits, greedy tokens)
- `sabah benchmark` / `sabah-bench`, including the forced-resident A/B
- workload profiles beyond the global one
- online popularity refresh with hysteresis
- GUI launcher

The planner deliberately ships before the runtime: it is what decides whether a
given machine should run the runtime at all, and on this machine the answer is
no.

---

## Session 2 - Phase B (the GPU hot-tier runtime)

Goal, as set: `real GGUF -> RAM expert bank -> VRAM hot tier -> exact GPU
expert execution -> correct output`. That chain now runs end to end on the
6 GB development GPU, on a subset of blocks.

### Built

| # | thing | verification |
|---|---|---|
| 1 | `runtime/cuda/sabah_rt.cu` | Q4_K/Q5_K/Q5_1/Q8_0 decode-and-matvec, copy stream, events |
| 2 | `runtime/rt.py` | ctypes binding, actionable error when the library is absent |
| 3 | `runtime/expert_bank.py` | mmap or pinned-RAM; slice offsets byte-compared to gguf-py |
| 4 | `runtime/hot_tier.py` | 4 states, LRU, per-block slot pools, telemetry |
| 5 | `runtime/executor.py` | one MoE block + an independent numpy/gguf-py reference |
| 6 | `tests/test_quant_kernels.py` | bit-exact vs gguf-py on real bytes |
| 7 | `tests/test_moe_block.py` | exactness + discrimination + residency independence |
| 8 | `tools/bench_moe.py` | measured per-block throughput, M1 A/B |
| 9 | `tools/calib_check.py` | static placement vs LRU, on the same trace |
| 10 | CLI `selftest` / `bench` / `calibrate` | all run |

### The exactness argument, in three tests

1. **Decode.** All four expert quant types produce `max|diff| = 0.000e+00`
   against `gguf.quants.dequantize`, on bytes taken out of the user's own
   artifact. The device test recovers each element by dotting the sub-block
   with a unit vector, so it exercises `sub32_*` - the code the matvec kernels
   call - rather than a parallel implementation that could be right while the
   kernels are wrong.
2. **Execution.** A full block agrees with a numpy reference to 6.0e-07
   relative L2. The same comparison with one routed expert swapped gives
   4.7e-01. A test that cannot detect a wrong expert cannot certify a right
   one, so that second number is part of the pass criterion.
3. **Residency independence.** Same block, same ids, run resident and again
   after forced eviction and refetch: bit-identical. This is what makes the
   cache legitimate rather than merely fast.

### Measured (block 0, real trace, 1000 tokens)

```
slots  hit      ms/tok   gpu stall  host stage   fetched
   16  0.0765    4.646      0.223       3.759    28.4 MB    mmap bank
   64  0.3091    3.377      0.203       2.569    21.2 MB
  256  0.8410    1.520      0.061       0.551     4.9 MB
  512  1.0000    1.274      0.000       0.000         0     <- kernel only

   16  0.0765    2.879      1.407       0.000    28.4 MB    pinned RAM bank
   64  0.3091    2.186      0.905       0.000    21.2 MB
  256  0.8410    1.330      0.061       0.000     4.9 MB
```

### Two findings

**1. With an mmap bank the bottleneck is the host, not PCIe.** 81% of the time
at 16 slots (3.759 of 4.646 ms) is the CPU copying page-cached bytes into
pinned staging; the GPU waits 0.223 ms. This was measured directly, not
inferred by subtraction - the staging copy is timed in
`hot_tier._fetch_async`. Staging throughput works out to 7.5 GB/s, which for a
single-threaded read+write is already at the RAM roof.

The fix is not to thread the copy but to remove it: a pinned RAM bank is
DMA-readable in place. `ExpertBank(mode="ram", pinned=True)` plus the
`_direct` path in the hot tier deletes the copy and buys 1.6x at low
residency. With it gone the accounting closes - 1.274 ms kernel + 1.407 ms
measured stall vs 2.879 ms observed - and since 28.4 MB at the measured
12.76 GB/s would cost 2.22 ms, about 37% of the transfer is genuinely hidden
behind compute.

This also sharpens what the planner's `GPU_HOT_TIER` vs `STORAGE_BACKED`
distinction is worth: it is not just a hit-rate difference, it is whether the
staging copy exists at all.

**2. The planner was calibrated on the wrong policy.** v4's hit curve measured
a STATIC hot set. The runtime seeds by popularity and then runs LRU. Measured
on the same trace, same byte weighting, same capacity budget, LRU beats an
*oracle* static placement by 0.13-0.19 and the deployable one by 0.17-0.24.
The planner was under-predicting its own runtime by ~0.10 hit and refusing
machines it could serve.

Fixed: `HIT_CURVES['qwen4exp']` now carries `lru` (the operating point) and
`static_deployable` (the floor). The generalisation-gap subtraction was
removed from the LRU path - that curve is already measured on tokens the
warm-up window never saw, so subtracting it again would double-count. The low
end of every hot-tier projection is now the same model re-evaluated at the
static hit rate, rather than a flat percentage band.

### Bugs found during bring-up

1. **PTX JIT refused by the driver.** Default nvcc codegen for CUDA 13.3
   produced PTX the installed driver would not compile. Fixed by building for
   the detected architecture (`-arch=sm_89` here); the build instructions now
   derive it from `nvidia-smi` instead of hard-coding it.
2. **Measuring the stall created one.** `cudaEventElapsedTime` synchronises, so
   reading the compute-stream stall inside the loop serialised host and device.
   Replaced with a 512-deep ring of event pairs drained between runs.
3. **Staging reuse drained the whole copy stream.** Reusing a staging buffer
   called `cudaStreamSynchronize(copy)`, waiting on every in-flight transfer
   instead of that buffer's own. Each buffer now carries its own event. This
   alone took block 0 at 16 slots from 9.127 to 4.646 ms/token.
4. **Heredoc truncation.** Large source files written through shell heredocs
   were silently cut off mid-file, producing an unterminated heredoc rather
   than a syntax error. Switched to direct file writes for anything large.
5. Test entry points read `sys.argv[1]` even when called in-process, so the CLI
   subcommand name was interpreted as a model path.

## What is still not built

- the full-model pipeline: attention, the fixed path, sampling, KV management.
  Until that exists there is no end-to-end tokens/s number, measured or
  otherwise.
- a correctness harness against llama.cpp at the logits and greedy-token
  level. Block-level exactness is proven; whole-model equivalence is not.
- `server/` and `adapters/` - untouched, and therefore nothing is claimed
  compatible.
- multi-GPU anything. Still one 6 GB card here.
- online popularity refresh with hysteresis; GUI launcher.

## Next session

1. Wire the blocks together into a full forward pass, reusing llama.cpp for
   attention and the fixed path, so that logits can be diffed.
2. Correctness harness: router ids, expert outputs, logits, greedy tokens.
3. Only then the API server.
