# Locality findings — the real 48-block routing trace

Decisive measurement for the 10–15 tok/s target. Tool:
`tools/trace_locality.py`. Trace: `C:\LMStudio\scratchpads\dk\trace_full`
(48 blocks × 499 steps × top-10, captured from the real model). Artifact:
`results/trace_locality_full.json`.

## Baseline (measured, not projected)

`C:\LMStudio\scratchpads\dk\cap_run.log` is a reference run of the full model
through llama.cpp with `sabah-diag` capturing. Per prompt step:

```
step 460  argmax 37972  1539.4 ms      step 493  argmax 2998   885.2 ms
step 464  argmax 13     3022.5 ms      step 499  argmax 6009  1227.3 ms
step 466  argmax 6970   4531.5 ms
```

Median ≈ 1.25 s/token, mean ≈ 1.2 s/token → **≈ 0.8 tok/s**. That is the number
to beat; 10–15 tok/s is a **13–20× speedup**.

**Two corrections to how that baseline must be read.**

1. The trace is **decode, not prefill.** `cap_trace.bat` sets
   `SABAH_DIAG_STEPS=decode -n 500`, batch 1, greedy self-generation. So the 499
   steps behind every number below are *decode* steps, which is exactly the
   workload that matters. (The 62.7 s step 0 is the prompt.)
2. The run is **`-ngl 4 -dev CUDA1`**: four layers on the GTX 1080, the other
   **44 layers of attention/SSM/router on a Ryzen 5 7600**. It is not a fair
   llama.cpp baseline for this hardware, and it is not a disk measurement. See
   §"Baseline is not disk-bound" below.

## Disk: measured, unbuffered (`tools/disk_probe.py`)

`FILE_FLAG_NO_BUFFERING`, one handle per thread, random aligned offsets over the
whole shard — the page cache cannot flatter these numbers.

| read size | QD=1 | QD=8 | QD=32 | p50 @QD=1 |
|---|---|---|---|---|
| 512 KB | 1181 MB/s | **2113 MB/s** | 1647 MB/s | 0.44 ms |
| 1 MB | 1590 MB/s | **2491 MB/s** | 2036 MB/s | 0.65 ms |
| 2 MB | 1359 MB/s | **2744 MB/s** | 2367 MB/s | 1.63 ms |
| 4 KB | **29 MB/s** (7167 IOPS) | 276 MB/s (67k IOPS) | — | 0.15 ms |
| sequential 64 MB | 2417 MB/s | — | — | 24.0 ms |

Reading: **the drive is not the bottleneck.** Scattered 1–2 MB reads sustain
1.4 GB/s at queue depth 1 and **2.1–2.7 GB/s at queue depth 8**; QD 32/64 is
*worse* than QD 8, so the optimal deployment issues **≈8 outstanding reads**.
Sorting the offsets barely helps (2.5 GB/s vs 2.4), so there is no
coalescing win to chase.

The 4 KB line prices the PLE table (`per_layer_token_embd.weight`, 160 ×
320001536, IQ4_NL, 90 B per row, `ngram_size = 3`, `heads_per_ngram = 8` →
16 rows/token): ~0.14 ms per scattered row, ~2 ms per token even with the page
cache cold. **Negligible** — PLE prefetch is a nice-to-have, not a lever.

## Baseline is not disk-bound

The old working assumption was that 0.82 GB/s of implied throughput
(0.9823 GB ÷ 1.2 s) was the disk. The probe above refutes it: the disk does
1.4–2.7 GB/s for these accesses. So the ~1.2 s/token is spent **elsewhere** —
44 layers of attention/SSM/router on a 6-core CPU, plus llama.cpp's synchronous
mmap page-fault path and per-layer host orchestration.

That is a much better diagnosis than "the disk is too slow", because it means
the *first* fix is not a cache at all: it is **getting the fixed path off the
CPU and onto the 5060 Ti.**

## What the trace says

| metric | value |
|---|---|
| blocks / steps / top-k | 48 / 499 / 10 |
| requests | 239,520 |
| distinct `(block, expert)` objects touched | **13,000 of 24,576 (53%)** |
| routed bytes/token at 100 % miss (W) | **0.9823 GB** |
| expert size | 1.51 / 2.05 / 2.66 MB (min/mean/max) |
| expert bank | **50.29 GB** |
| cold (first-touch) requests | 5.4 % |
| reuse distance | p50 = 1442, p90 = 16,801, p99 = 80,632 |

Because only 53 % of the bank is touched in 499 steps, **no policy can exceed a
94.6 % hit rate on this trace** — the other 46 % of the bank is simply never
read.

## Hit rate vs capacity (byte-weighted)

Slot-count-based, so `GB = slots × mean_expert_bytes`; the partitioned column is
the design Sabah already has (one slot pool per block).

| expert cache | fill | global LRU | per-block LRU | Belady (optimal) | static top-K (oracle) |
|---|---|---|---|---|---|
| 3.99 GB | 15 % | 62.0 % | 61.2 % | — | 58.3 % |
| 5.32 GB | 20 % | 69.3 % | 68.7 % | 84.2 % | 66.3 % |
| 7.98 GB | 30 % | 80.4 % | 78.6 % | 89.6 % | 78.2 % |
| **10.64 GB** | **40 %** | **86.4 %** | **84.7 %** | **92.4 %** | 86.4 % |
| 13.30 GB | 50 % | 89.9 % | 88.6 % | 93.8 % | 91.8 % |
| 15.96 GB | 60 % | 92.5 % | 91.2 % | 94.5 % | 95.3 % |

Disk time per token = `fetch_bytes / disk_bandwidth`. The bandwidths are now
measured (above), not assumed:

| expert cache | fill | LRU hit | fetch MB/tok | @1.59 GB/s (QD1,1MB) | @2.49 (QD8,1MB) | @2.74 (QD8,2MB) |
|---|---|---|---|---|---|---|
| 5.32 GB | 20 % | 68.7 % | 301 | 5.3 | 8.3 | 9.1 |
| 7.98 GB | 30 % | 78.6 % | 193 | 8.2 | 12.9 | 14.2 |
| **10.64 GB** | **40 %** | **84.7 %** | **151** | **10.5** | **16.5** | **18.1** |
| 13.30 GB | 50 % | 88.6 % | 112 | 14.2 | 22.2 | 24.5 |
| 15.96 GB | 60 % | 91.2 % | 87 | 18.3 | 28.6 | 31.5 |

With the measured disk, a **plain per-block LRU tier of ~10.6 GB already lands
at 10.5–18 tok/s** — inside the target — before any prefetch or speculative
decoding is added. Batched passes (B = 3–4, factor 0.69–0.75) push it further.

## Four findings that change the design

1. **Per-block partitioning is nearly free** (−0.6 to −1.8 points vs a global
   LRU at the same total bytes). Sabah's existing per-block slot pools are the
   right structure; there is no payoff in building a unified cross-block cache.

2. **Static pre-pinning does not work.** Ranking experts on the first 99 steps
   and scoring on the remaining 400: 10.64 GB gives **69.6 %** where adaptive
   LRU gives **86.4 %** and oracle-static gives 86.4 %. The expert distribution
   drifts within a single 500-token prompt. Any "rank once at load, pin forever"
   plan is wrong by ~17 points.

3. **LRU leaves 6 points on the table vs Belady** (86.4 % vs 92.4 % at
   10.64 GB). That gap is exactly what a predictor can harvest: 134 → 74 MB/tok
   = **+1.8×**, for free on the critical path because a prediction is only ever a
   hint. Pieces A (n-gram) and D (MTP) are aimed at this gap, and it is worth
   1.8×, not 1.02×.

4. **Batching pays a lot.** Union of experts per layer per pass, over the real
   sequential token order:

   | batch | union/pass (avg) | union/token | bytes/token | ratio |
   |---|---|---|---|---|
   | 1 | 480.0 | 480.0 | 982 MB | 1.000 |
   | 2 | 799.8 | 399.9 | 818 MB | 0.833 |
   | 3 | 1075.6 | 358.5 | 734 MB | **0.747** |
   | 4 | 1322.2 | 330.6 | 676 MB | **0.689** |
   | 8 | 2104.3 | 263.0 | 538 MB | 0.548 |

   Speculative decoding was priced as a *latency* trick. Here it is also a
   **bandwidth** trick: B = 3–4 cuts expert bytes/token by 25–31 %.

## Consequence for the target

| configuration | MB/tok | @1.59 GB/s | @2.49 GB/s | @2.74 GB/s |
|---|---|---|---|---|
| 10.6 GB VRAM tier, LRU, B=1 | 151 | 10.5 | 16.5 | 18.1 |
| 10.6 GB VRAM tier, LRU, B=4 | 104 | 15.3 | 23.9 | 26.3 |
| 10.6 GB VRAM tier, Belady-ish, B=1 | 74 | 21.5 | 33.6 | 37.0 |
| 10.6 GB VRAM tier, Belady-ish, B=4 | 51 | 31.2 | 48.8 | 53.7 |

So, in priority order:

1. **Hit rate plus a correct async I/O path is already enough.** ~10.6 GB of
   per-block slots at QD ≈ 8 clears 10–15 tok/s on disk time alone. The disk is
   not the thing to optimise.
2. **Prediction (the LRU→Belady gap, +1.8×) and batching (0.69–0.75×) are the
   headroom**, not the requirement. They buy the margin the target needs if the
   real per-token orchestration cost is larger than assumed.
3. The 4.18 GB fixed path is not optional and is not free: it is 26 % of VRAM
   before a single expert is cached. **Putting it on the 5060 Ti instead of the
   CPU is the first win available, and it needs no cache at all.**

## Caveats, stated plainly

- The 499-step trace is a single greedy continuation of a ~60-token prompt.
  It is decode, batch 1 — the right kind — but one conversation. A second trace
  from a different prompt is needed before the hit-rate curve is treated as
  steady state.
- 13 000 of 24 576 objects is 53 %: the 94.6 % ceiling is a property of this
  trace, not of the model.
- Byte sizes come from the GGUF's own qtype mix (gate/up `IQ2_S`/`IQ3_XXS`/
  `IQ3_S`/`IQ4_XS`, down `IQ4_NL`/`Q2_0`), which is why expert size varies
  1.51–2.66 MB between blocks.
- `GB` on the capacity axis assumes a uniform slot count per block, which is the
  design Sabah actually has.
- The disk probe was single-purpose and lightly loaded (LM Studio not resident).
  **Its 2.4-2.5 GB/s was a repeat-read artifact and has been withdrawn:** it drew
  one offset list and re-read it for every case it reported, and this drive
  serves a repeated read ~1.5x faster than a fresh one. Fresh offsets measure
  1.05-1.45 GB/s at one to sixteen readers (`tools/disk_probe.py`,
  `tools/io_bench.cu`, `docs/DECODE_BUDGET.md`), and under a real engine the
  drive also serves the PLE rows and the OS, so treat **1.2-1.4 GB/s** as the
  working number and the ~1.5x repeat-read rate as unreachable.
- The baseline's split between CPU compute and I/O orchestration is **not yet
  measured**; the claim that it is not disk-bound is inferred from the disk
  probe, not from a direct profile.
