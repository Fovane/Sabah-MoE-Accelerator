# Sabah-MoE-Accelerator

**A streaming MoE inference runtime for RAM-constrained machines.**

Sabah runs models far larger than the machine's VRAM + RAM combined. It does that
**without a RAM tier, without the page cache and without the page file**: the
expert path goes `disk -> pinned staging -> VRAM` with `FILE_FLAG_NO_BUFFERING`.
The measured claim is that **host RAM stays flat while the expert tier fits in
VRAM** - 4.78 GiB at a short context, `mapped_fallbacks=0`. At `-c 131072` the
tier stops fitting beside the KV cache and ~3.4 GB of it is served from system
RAM; that boundary is measured, not assumed (REPORT 23.1).

That is the point of the project. An 83.6 GB MoE model either fails to load on a
16 GB-VRAM / 15 GB-RAM machine, or it "runs" only by letting the OS page weights
through RAM  -  which requires a large page file and makes latency unpredictable.
Sabah is built for the machine that has **the VRAM but not the RAM**.

Measured end to end on the reference rig:

| | |
|---|---|
| model | `Qwen3.8-Flash-Next` IQ3_S, **83.6 GB** (2 shards, 48 blocks x 512 experts, top-10) |
| time to first token | **11.2 s** (the 53-token prompt is easy; the expert tier starts empty and fills 12.68 GB at 1.11 GB/s) |
| **decode, short context, SEG=48 (best measured)** | **30.36 ms/token = 32.93 tok/s** over the last-256 window, byte-identical outputs (REPORT 16.6, N=1, clocks valid) |
| decode, same, SEG=0 baseline | 42.74 ms = 23.40 tok/s - what the graph work bought |
| decode, short context, no SEG | **~67 ms/token = ~15 tok/s**, effectively disk-free (4.5 reads, 3.0 MB/token) |
| decode, 128K window, short prompt | 30.36 ms (32.93 tok/s), ~3.7K-deep KV (REPORT 16.6) |
| decode, **filled ~30K KV** | **280.45 ms/token = 3.57 tok/s**, 339 MB/token, 501 misses/token, still not steady at step 2048 (REPORT 21-22) |
| decode, first 128 tokens | 79-204 ms/token: the tier is still discovering its working set |
| 256-token answer end to end | 11.2 s + 24.6 s = **7.1 tok/s including the fill**, 10.4 over the decodes |
| **host RAM** | **flat at 4.78 GiB** at this context (53-token prompt), `mapped_fallbacks=0`. At `-c 131072` the 6 GB slab no longer fits beside the KV cache and **~3.4 GB of it runs from system RAM** - see the limitation below |
| peak VRAM | 15.36 GB of 16.31 GB |
| correctness | byte-exact against the GGUF ranges (`verify_ok=64191 verify_fail=0`) |

The shape matters here, so it is worth stating plainly: this runtime is fast once
warm and slow to get warm, because the tier (9.8 GB) is smaller than the working
set of a fresh prompt (12.68 GB of cold fill) and has to learn it. One long answer
runs at 15 tok/s; many short ones pay ~11 s each before the first token. The full
breakdown, per regime, is in [`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md).

Numbers describe the **shipped streaming path** (`sabah/runtime/cuda/sabah_rt.cu`
+ the llama.cpp bridge). The separate Python block-level harness in
`sabah/runtime/*.py` is research scaffolding and is described
[at the end](#python-block-harness-separate-from-the-shipped-path).

---

## Design rules

1. **RAM is not a tier.** Nothing about expert delivery may depend on host memory
   growing. Host RAM is allowed to hold only fixed-size pinned staging buffers
   and loader bookkeeping.
2. **Routing semantics are never changed.** Sabah decides *when* an expert's bytes
   reach the GPU and *where* they are kept. Which experts run, and the input rows
   they are applied to, is exactly what llama.cpp computed. The bridge exists only
   to intercept execution, not to alter it.
3. **Every optimisation must be provable.** Either byte-exact verification against
   the GGUF ranges, or a preserved invariant (for example: a prefetcher is only
   allowed to move bytes in time, so the tier's `hits`/`misses`/`evictions` must
   come out identical).

Rule 3 is why every number in this document comes with the check that proves it.

## How it works

### The tier: VRAM is hot, the disk is cold

There is no intermediate RAM tier. The bank of expert weights is a 50.29 GB pool
of 24576 objects (512 experts x 48 blocks) and it never fits; a VRAM slab
(`SABAH_LLAMA_HOT_BYTES`, 9.8 GB on this rig) holds whatever is currently hot and
everything else is read on demand from the GGUF.

A decode token asks for 480 experts x 3 tensors = 1440 tensor reads (~0.97 GB of
expert bytes). With the 9.8 GB slab, **81.8%** of those hit and **262 reads
(174.7 MB) go to the disk**.

### Direct I/O with pinned staging

Misses are issued as aligned unbuffered reads:

- `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED`, 4096-byte alignment, so the
  read lands in pinned host memory instead of the OS page cache. The page cache is
  deliberately bypassed  -  it is what would make host RAM grow.
- a double-buffered pinned staging pool (~43 MB) with a wave-boundary event gate;
  `SABAH_LLAMA_IO_DEPTH` (default 8) reads in flight per wave.
- `SABAH_LLAMA_DIRECT_IO=0` switches back to the mapping, for A/B comparison only.

### The llama.cpp bridge

Sabah is not a fork and not a standalone loader; it hooks one operator. Patch
`0001` makes the scheduler keep the original host GGUF expert tensor in the graph
for `MUL_MAT_ID` and hands that op to `sabah_rt_mul_mat_id_v2` together with the
graph's own ids, input and output tensors. Nothing else about llama.cpp changes
and the integration is **opt-in via `SABAH_LLAMA=1`**. See
[`patches/llama.cpp/README.md`](patches/llama.cpp/README.md).

Two placement details are required and easy to get wrong:

- `GGML_OP_OFFLOAD_MIN_BATCH=1`. With `--cpu-moe`, llama.cpp only sends an expert
  op to CUDA when its batch is at least 32 tokens (the default), so every
  single-token decode would silently run on the CPU and never reach Sabah.
- `-ot "\.per_layer_token_embd\.weight=CPU"`. `-ngl 99` alone pushes the 28.8 GB
  per-layer token embedding into VRAM. It is dense and read sequentially, so the
  CPU with the file mapping is the right home for it.

### What was optimised, and what it bought

Everything here came from measuring the token, not from guessing. **Every tok/s
figure in this table is the warm-up regime** (the tier still filling, 6-token
runs) - not the steady state, and not comparable with the headline table above,
whose figures are steady-decode. Baseline to current in that regime is
**4.37 -> 4.84 tok/s (+11%)**, with no change to the outputs.

| change | cost before | after | tok/s (warm-up) |
|---|---|---|---|
| baseline (per-call stream drain, linear erase) | `ids` 87 ms | | 4.37 |
| reuse the router ids across a layer's 3 expert tensors | 87 ms | ~23 ms | 4.75 |
| single-scan residency memo per call | | | 4.79 |
| swap-and-pop eviction instead of `vector::erase` | `slot` 30 ms | ~11.5 ms | **4.84** |

**Router id reuse** is the interesting one. The three expert tensors of a MoE
layer (`gate`, `up`, `down`) read the *same* `selected_experts` tensor, so two of
the three `cudaStreamSynchronize` per layer are redundant. The bridge passes
`dst->src[2]`  -  the ids *tensor address*  -  so the runtime can recognise "same
tensor as the previous call" and skip both the download and the drain. Groups are
capped at `SB_ID_GROUP_MAX=4` calls, and that cap doubles as the width of the
eviction-protection window, which is what makes it safe to free a slot while a
kernel enqueued before the last drain may still be reading it.

Removing those drains exposed a latent race worth recording: the pointer table is
**one** shared device buffer that every call rewrites, and the upload used to run
on the copy stream. The drains had been supplying that ordering implicitly. The
upload now runs on the compute stream. (A run with the racy order reported
9.90 tok/s and 23% fewer bytes; those counters are not reproducible and that run
is not trusted.)

### Prefetch is off by default, on purpose

A staging-readahead prefetcher exists (`SABAH_LLAMA_PREFETCH=1`) and is correct:
it reserves no tier space and leaves `hits`/`misses`/`evictions` identical to a
prefetch-free run. It is off because it does not pay  -  adding reads in flight
raised the drive's effective rate (1.3 -> 1.6 GB/s) but not the wall clock, and
40% of the readahead was superseded before anyone asked for it. It becomes worth
revisiting when the disk term drops below the bookkeeping term.

## The measured decode budget

A generation has three regimes and they must not be averaged:

| regime | cost | what limits it |
|---|---|---|
| **time to first token** (53-token prompt) | **11.2 s** | the tier's cold fill: 12.68 GB at 1.11 GB/s |
| **warm-up** (first ~128 decodes) | 70.6 MB/token of disk, 79-204 ms/token | the tier still discovering its working set |
| **steady state** (~15 tok/s, ~67 ms/token) | 3.0 MB/token, 4.5 reads/token | the no-disk floor - the disk is off the critical path |

End to end a 256-token answer is 11.2 s + 24.6 s = **7.1 tok/s including the
fill**, 10.4 tok/s over the decodes alone, 15 tok/s once warm. Full method, phase
attribution and the per-regime lever table: [`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md).

### 128K context, and the segmented CUDA graph (item 4)

At 128K the steady decode is disk-bound again, and the segmented path removes
GPU-side idle time rather than launch overhead:

| | baseline (SEG=0) | SEG=48 (default) |
|---|---|---|
| 128K chat, 2048 steps, tail window (last 256), ms/token | 42.74 | **30.36** |
| tok/s | 23.40 | **32.93** |
| VRAM peak | 15745 MiB | 15872 MiB |
| outputs | reference | byte-identical |

That is N=1, at slab 6.0e9, with the tail window read at a **~3.7K-deep KV** - a
6817-character prompt, not a filled 128K context. It is **not** a
filled-128K-context number, and the coding trace does not warm up in 2048 steps
at that slab - see the limitations below. The saving is measured, the mechanism
is in [`docs/CUDA_GRAPH_SPLIT.md`](docs/CUDA_GRAPH_SPLIT.md).

### Decode on a deep KV: what the same code does at 30K tokens of context

The first measurement at real depth, same build and settings (`-c 32768`,
`-b 2048`, `-ub 1024`, slab 6.0e9, 2048 decode steps after a 28,413-token
prefill, clocks valid, `N=1`):

| regime | ms/token (min / median / max) | MB/token | misses/token |
|---|---|---|---|
| warm-up (first 128) | 0.0 / **312.5** / 796.3 | 342 | 500 |
| last 256 ("steady") | 103.2 / **310.0** / 665.0 | 344 | **504** |

tok/s over that tail: **3.2**. The last-256 window is **not steady**: at step
2048 the run still misses 504 experts per token, which with 48 layers and top-10
is 10.5 misses per layer - every routed expert is still a miss. So there is no
steady-state number to quote at this depth, and the 10x gap to the table above
is the expert working set rather than the KV depth: the coding trace shows the
same 272 ms tail at only ~3.2K depth because it routes as widely
(REPORT 15.4). Full evidence: REPORT sections 21-22.

### Why no policy fixes it: the offline bound

The routing trace for those 2048 steps was replayed offline
(`tools/slab_replay.py`, 2,949,120 accesses = 48 layers x 10 top-k x 3 roles),
so the best any replacement policy could do is measured, not argued:

| slab | LRU (what ships) | per-layer budgets | **Belady (offline optimum)** |
|---|---|---|---|
| 2.4 GB (spill-free) | 45.64% | 45.65% | **68.02%** |
| 6.0 GB | 67.64% | 67.31% | 84.25% |
| 8.0 GB | 75.48% | - | 88.57% |

The replay is trustworthy because its LRU at 6 GB predicts 466 misses/token
against the runtime's measured 501.

Each tier is a bytes-per-token budget (the regime is disk-bound: its ms/token is
within ~1.1x of the disk service time for its bytes alone):

| tier | tok/s | needs this hit rate |
|---|---|---|
| A | 10 | **88.0%** |
| B | 20 | 94.0% |
| C | 30 | 96.0% |

**Tier A needs 88%, and the offline optimum at the only spill-free slab reaches
68%.** The gap is in the trace - the distinct experts a ~30K-KV token routes to -
not in the eviction policy. A better policy buys at most 466 -> 227 misses at
today's slab, which is ~130 ms of disk against a 100 ms budget. So no tier is
reachable on a deep context on this hardware, and that is a bound rather than a
tuning result. What would move it: fewer bytes per distinct expert (a lower-bit
quantisation), or a drive faster than the measured 1.05 GB/s.

Removing the system-RAM spill was measured too: a 2.4 GB slab drops VRAM peak
12914 -> 9350 MiB and keeps the outputs byte-identical, but its bytes get worse
(840 misses / 570.6 MB per token against 501 / 339.5), so it trades system RAM
for disk and does not buy a tier. That run's clocks fell to 2017 MHz steady, so
its time is marked invalid and only its bytes are used.

**The short context is unaffected**: 30.36 ms/token and byte-identical, from
section 16.6, and it is the number the 30-33 tok/s headline refers to.


### Prefill with a wider ubatch (no new algorithm)

A 2048-token chunk read by four 512-token ubatches re-streams the expert pool
once per ubatch. With the chunk size fixed at 2048 and only the ubatch size
changed (same prompt, same build, byte-identical routing counters):

| `-ub` | disk bytes for one 2048-token chunk | vs `-ub 512` |
|---|---|---|
| 512 (4 ubatches) | **113.0 GB** | - |
| 1024 (2 ubatches) | **65.9 GB** | **-42%** |

Larger `-ub` moves the same work into one ubatch, so the pool is streamed once
per chunk instead of once per ubatch. The full sweep (one 4486-token prompt,
`-b 2048`, slab 9.8e9, same build, `N=1`):

| `-ub` | disk GB | per 2048-token chunk | ms/token | cache misses | host WS peak |
|---|---|---|---|---|---|
| 512 (default) | 257.5 | 113.0 / 115.5 GB | 114.3 | 382644 | 5894 MiB |
| 1024 | 161.8 | 65.9 / 66.9 GB | 107.0 | 241183 | 6492 MiB |
| 2048 | 102.9 | 37.1 / 36.9 GB | 350.0 | 154016 | 7128 MiB |
| 4096 | 102.9 | 37.1 / 36.9 GB | 102.3 | 154016 | 7265 MiB |

Disk halves with each doubling, because a chunk is read once per ubatch inside
it; the 390-token tail chunk reads the same ~29 GB in all four settings, which
is the control. The 2048 and 4096 rows are the same execution (identical cache
counters) and their 3.4x wall difference sits in the host `ids`/`slot` phases,
not the disk, so with `N=1` only the disk claim is made.

**A wider `-ub` is not byte-identical and stays opt-in.** `-ub 512` vs `-ub 1024`
over four decode rows: max abs logit diff 2.905, top-1 agreement 4/4 rows,
max abs KL 0.21. So `--flag=-ub 1024` is a *numerically equivalent* mode, never
a default; `-ub 512` is the default and is what every other number here was
measured with.

One real 32K-class prefill at the best setting (`-ub 1024`): 1174 s, 14 chunks,
1013 GB, 41.3 ms/token. Projected to 128K that is 4679 GB and **90.3 min**,
over the 60-minute threshold, so the full 128K prefill was **not** run - the one
setting that would fit (`-ub 2048`) is not qualified to recommend. Full sweep,
compute buffer, VRAM and per-setting detail: [`REPORT.md`](results/phase0/REPORT.md)
section 20-21.

The warm-up token, phase by phase (differencing `-n 32` minus `-n 8`, i.e. decode
steps 9..32  -  this is the number the README used to quote as *the* rate):

| phase | per token | notes |
|---|---|---|
| `fetch`  -  disk reads + H2D | **121.8 ms** | 164.5 MB/token over 247.4 misses, 82.8% marginal hit |
| `ids`  -  router download + drain | **29.3 ms** | 3 calls per layer |
| `slot`  -  residency bookkeeping | **9.3 ms** | 239 evictions/token, ~39 us each |
| `ptr` + `prime` | **2.8 ms** | |
| outside the bridge (graph, launch, host ops) | **27.7 ms** | residual: phases sum to 163.2 of 190.9 |
| **warm-up decode wall** | **190.9 ms** | median 205.7 ms |

The steady state is the floor and nothing else: measured independently by
replaying a token stream against a warm tier (which reads 0 bytes), **67.3
ms/token**. Two different measurements, same number.

### The disk is the wall for the fill and the warm-up, and it is saturated there

| | |
|---|---|
| drive | MSI M371 1TB NVMe (Gen3 class) |
| drive ceiling, fresh offsets, 676 KB random | **1.05-1.45 GB/s** (repeat-offset probes overstate it ~1.5x) |
| Sabah achieves in the demand path | **1.11-1.35 GB/s** |
| Sabah achieves in the steady state | ~0: 3 MB/token |

So the demand path is **at** the drive's rate, not below it  -  that is lever 2's
conclusion, and it is why raising IO depth or prefetching changes nothing: there
is no unused bandwidth to recover. The dependency chain is also genuinely one
layer deep (layer *L*+1's router needs layer *L*'s expert output), so the drive is
idle during the GPU work and vice versa; a faster drive helps by exactly
`bytes / rate`, which is why the 12.68 GB fill is the most valuable thing on this
list to attack.

## What this project does not claim

- **It reaches 15 tok/s only once warm.** The remaining gap is the fill, which is
  paid once per fresh working set: 12.68 GB at 1.11 GB/s. On a 6 GB/s drive the
  same fill is ~2.1 s and a 256-token answer goes from 7.1 to ~10.5 tok/s end to
  end. The steady state cannot be improved by any storage change at all - it is
  already the floor - and the floor is the model's own per-layer work at batch 1.
- **It is not faster than a machine with enough RAM.** With host RAM allowed to be
  a tier, the warm-up's ~110 reads/token could be served at 0.047 ms instead of
  0.45 ms per expert (measured, [`docs/SECOND_TIER.md`](docs/SECOND_TIER.md)),
  worth ~1.5x on the warm-up and nothing on the steady state. Sabah trades that
  away for a hard guarantee about RAM  -  a deliberate budget decision, not a
  performance claim.
- **Virtual memory does not help.** Windows "virtual memory" is the page file on
  the same NVMe, reached through the page-fault path instead of a 676 KB
  unbuffered read at QD 8  -  the same bandwidth, a worse access path, no
  controllable queue depth. It adds capacity, and the steady state does not need
  capacity.
- **The GTX 1080 is not used, and measurement says it should not be.** The old
  +18% ceiling assumed the path was cheap. It is not: a park plus a hit through
  the 1080 costs 0.454 ms per expert against the disk's 0.45 ms, with no P2P on
  consumer parts and a gen3 x4 link on the 1080. Break-even at a 100% hit rate and
  worse from there ([`docs/SECOND_TIER.md`](docs/SECOND_TIER.md)).
- **Prefill is the wall for long prompts.** One 2048-token chunk streams 113 GB
  because the expert pool is read once per ubatch in the chunk; the 128K prompt
  of item 6 needed ~130 min. A wider `-ub` removes the re-stream (see the table
  above), and the layer-group prefill that was meant to remove it entirely is
  **off by default and not fixed** ([`docs/PREFILL_GROUP_MAJOR.md`](docs/PREFILL_GROUP_MAJOR.md)
  section 6.2 records the root cause: the fused-kernel set changes at the split
  boundary).
- **The machine state can double ms/token.** The same decode measured 88.9 ms and
  58 ms on two runs (REPORT 9). Every timed number here was taken with the clock
  pin in force (`-lgc 2800,3090 -lmc 13000,14001`), and a run whose steady SM
  clock is below 2400 MHz is marked invalid by the harness and is not evidence.
  The pin needs admin; on this machine it was refused for part of one session
  and works again (REPORT 19).
- **Deep context is disk-bound, and no policy reaches a tier on this hardware.**
  The first measurement at ~30K KV depth reads 280.45 ms/token (3.57 tok/s) with
  339 MB/token of expert traffic and 501 misses/token; the last-256 window is
  *not* steady (every routed expert is still missing at step 2048). Replaying the
  same routing trace offline puts the Belady optimum at 68% hit rate on a
  spill-free slab, where tier A (10 tok/s) needs 88% (REPORT 25). The 30-33 tok/s
  figure above is a short-prompt, warm-tier number and does not extend to a deep
  KV with a wide-routing prompt. Never quote one number for both.
- **At 128K context the RAM-flat claim does not hold.** The *process* working set stays
  flat, but the driver moves slabs into system RAM when the slab no longer fits beside
  the KV cache and the compute buffer: sampling the GPU's Windows "Shared Usage"
  counter against its idle baseline gives **+110 MiB at `-c 512`, +349 MiB at
  `-c 32768` and +3517 MiB at `-c 131072`** (`tools/p3_shared.py`, REPORT 23.1).
  Size the slab for the context (the table in the reproduce section) and do not quote
  "host RAM is flat" for a 128K run with a 6 GB slab.
- **A bigger slab does not shorten the warm-up.** The coding trace at 6.0e9 vs
  9.8e9 (outputs byte-identical): double the slab halves the tail misses
  (408.4 -> 213.2/token) and the tail disk (277.1 -> 144.1 MB/token) and moves
  the tail 4%, but makes the warm-up 90% slower (296 -> 563 ms) and the prefill
  4.6x slower. A prompt-informed warm-up is not implemented; the measurement
  says why (REPORT 23.2).
- **Correctness is about bytes served, not model quality.** Byte-identical means the
  runtime serves exactly the GGUF's quantised weights (`verify_ok`, `id_check`, and
  the `SABAH_PREFILL_GROUP`/SEG toggles all check out byte-equal against the plain
  path). It does **not** prove the model is good: no perplexity, no accuracy
  benchmark and no A/B against another runtime is run here. The IQ3_S / IQ4_NL
  quantisation quality numbers are ISTA-DASLab's own measurements of their
  conversion, not ours, and Sabah can only be as good as the file it is handed. Run
  a perplexity or task-level comparison against a reference build before drawing
  any conclusion about output quality.
- **Layer-group prefill is not shipped.** Off by default, and additionally gated
  to a single ubatch per `llama_decode` call; see the design note.

## How to reproduce a measured number

One protocol, one command, and it refuses an invalid machine. It prints the
command for each regime and refuses when the clock is below the floor, another
process holds the GPU, or the GPU reports shared memory in use:

```bat
python -m sabah.tools.bench_protocol
```

```text
gpu: {'sm_mhz': 2835, 'mem_mhz': 13801, 'pstate': 'P1'}
processes on the gpu: none
verdict: CLEAN
machine is clean; run:
  python tools/phase0_measure.py --label <name> [--prompt-file PATH] --tokens 2048 --context 131072 ...
```

The pin commands it prints, and the warning that goes with them:

```bat
nvidia-smi -lgc 2800,3090        :: SM floor 2800, ceiling left free
nvidia-smi -lmc 13000,14001      :: memory floor 13000, ceiling left free
```

Both are reversible (`nvidia-smi -rgc` / `-rmc`) and need admin.

**The machine state can double ms/token, and this has happened here.** The same
decode measured **88.9 ms** and **58 ms** on two runs of the same binary: the first
sat in a low P-state with the SM clock at 585-900 MHz, the second at 2805 MHz
(REPORT 9). A 53% difference from no code change at all. So:

- pin the clocks before any timed run, and record `pstate`, `clocks.sm` and
  `clocks.mem` next to the number;
- the harness marks a run **invalid** when the steady SM clock is below 2400 MHz or
  the memory clock is below 10000 MHz, and an invalid run is not evidence - re-run it;
- if pinning is refused (it needs admin), the auto-boost may still be enough under
  load, but check the recorded steady clock before quoting the time;
- other processes holding a compute context are allowed (15 hold them on the
  reference rig, mostly system ones that never compute) but they are a flag, not a
  refusal - see `python -m sabah.tools.bench_protocol`, which prints the verdict
  and refuses on a low clock.

The regimes, and the command that produces each:

| regime | command shape |
|---|---|
| time to first token | `--tokens 32 --prompt Hello --context 131072` |
| warm-up (first ~128 decodes) | the same run, first 128 steps of `step_ms.f64` |
| steady / tail window | `--tokens 2048`, tail = last 256 steps of `step_ms.f64` |
| prefill | per-chunk rows of the driver log, `--flag=-b --flag=2048` |

Each run writes its own directory under `results/phase0/<label>/` with
`measurement.json` (clocks, the harness's validity verdict, per-chunk bytes and
ms, regime summary), `step_ms.f64`, `step_bytes.i64`, `step_misses.i64` and the
outputs used for the byte-identical checks.

## Hardware and target artifact

| component | value |
|---|---|
| GPU 0 | NVIDIA RTX 5060 Ti, 16 GB VRAM, `sm_120`  -  Sabah runs here |
| GPU 1 | NVIDIA GTX 1080, 8 GB VRAM, `sm_61`  -  not used |
| RAM | 15.38 GiB usable |
| disk | MSI M371 1TB NVMe (~1.05-1.45 GB/s random 676 KB reads, measured; see `docs/DECODE_BUDGET.md`) |

Target: `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, **IQ3_S** shard pair
(`...-00001-of-00002.gguf` 54.8 GB, `...-00002-of-00002.gguf` 28.8 GB).

## Build

**Windows, one click.** `install.bat` automates everything below: it finds Python
and MSVC, reads the GPUs' compute capability, creates `.venv`, installs the
package, builds the runtime DLL for the right architecture and runs
`sabah doctor`. Add `llama` to also fetch the pinned llama.cpp base, apply both
patches and build `llama-sabah-diag`:

```bat
install.bat            :: package + runtime DLL, architecture auto-detected
install.bat llama      :: also the bridge and the measurement driver
install.bat sm_89      :: force an architecture
```

It writes only inside the repository (`.venv\`, `sabah_rt.dll`,
`third_party\llama.cpp`), never to the registry, and it never opens a model. The
steps it automates, by hand:

Requirements: CUDA 12.8, MSVC 14.44 (VS 2022 Professional), CMake.

**1. The runtime DLL** (CUDA kernels, IQ quant types, slab allocator, direct I/O):

```powershell
cd sabah\runtime\cuda
.\build_dll.bat sm_120
```

**2. llama.cpp plus the bridge**  -  reproducible from the pinned upstream base:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b
git apply <repo>\patches\llama.cpp\0001-sabah-native-mul-mat-id.patch      # required
git apply <repo>\patches\llama.cpp\0002-sabah-diag-capture-tool.patch       # optional, diagnostics
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="61;120"
cmake --build build -j 8
```

## Run

The integration is inert unless `SABAH_LLAMA=1`. Same model, same flags as an
ordinary `--cpu-moe` run, plus the two placement requirements and the bridge:

```powershell
$env:SABAH_LLAMA        = "1"
$env:SABAH_RT_LIB       = "<repo>\sabah\runtime\cuda\sabah_rt.dll"
$env:SABAH_LLAMA_HOT_BYTES = "9.8e9"          # VRAM expert slab (default 2 GiB)
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"          # required, see above
$env:SABAH_LLAMA_TRACE  = "1"                 # cache / coverage / phase counters at exit

llama-sabah-diag.exe `
  -m $env:SABAH_MODEL_DIR\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf `
  -dev CUDA0 -ngl 99 --cpu-moe `
  -ot "\.per_layer_token_embd\.weight=CPU" `
  -t 8 -c 512 -n 32 -p "Hello"
```

`SABAH_MODEL_DIR` is the single source of truth for where the shards live - every
tool reads it, so no script in the repository carries a drive letter.

**Slab size by context** (REPORT 23: the slab has to fit beside the KV cache and
the compute buffer, or the driver puts the rest in system RAM):

| context | slab | why |
|---|---|---|
| `-c 512` .. `-c 32768` | 6.0e9 (9.8e9 is fine) | the KV cache is small; the shared-memory
spill is negligible (+110 MiB at `-c 512`, +349 MiB at `-c 32768`) |
| `-c 65536` | ~3.0e9 | the KV cache starts to dominate |
| `-c 131072` | ~2.4e9 | 6.0e9 spills ~3.4 GB into system RAM (Shared Usage +3517 MiB),
which makes every miss more expensive than the disk saving is worth |

`--hot-bytes 9.8e9` with `-c 512` is the *short-answer* setting; do not carry it to a
128K context.

`llama-sabah-diag` is the measurement driver from patch `0002`. The bridge lives
in `ggml-cuda`, so any llama.cpp binary loads it: `llama-cli` runs this model end
to end as well (verified  -  `EXITCODE=0`, zero mapped fallbacks, generated text,
~27 GB streamed for the same prompt), and so does `llama-server`, which is how a
client reaches this model over HTTP.

`-c 512` above is deliberate, not an oversight. The KV cache and the expert slab
are paid from the same 16.31 GB, and at `-c 32768` a 9.8 GB slab has ~0.4 GB of
headroom left: the run aborts with `SABAH_LLAMA could not execute the
host-backed exact expert operation` / `Sabah over-budget expert: out of memory`
once the tier's documented over-budget allocation has no VRAM to take. Both
sides of that trade, measured, are in
[`docs/CONTEXT_VRAM.md`](docs/CONTEXT_VRAM.md)  -  32K works with an 8.0 GB slab or
with `-ub 8`.

Two things to know when reproducing:

- **The prompt is 53 tokens, and the first token still costs ~11 s.** The chat
  template around `-p "Hello"` is 53 tokens (the diag writes the ids it used to
  `prompt_tokens.i32`; `prompt.txt` shows the text). The 11 s is the expert
  tier's **cold fill**: 12.68 GB at 1.11 GB/s, all of it before step 1. The
  `tokens=` field in the `SABAH_DIAG` line is a count of expert *rows*
  (`layers x tensors x batch`), not prompt tokens  -  for a 6-decode run it is
  already 8484, which is where an earlier version of this README misread it.
- **Quote decode per regime, not as one number.** `-n 1` is the fill, `-n 128`
  shows the warm-up, `-n 256` reaches the steady state (67 ms/token, disk-free);
  differencing `-n 32` and `-n 8` gives the warm-up marginal token
  (190.9 ms/token), which is *not* the sustained rate. `sabah run` reports all
  three.

### One-command harness

`sabah run` drives the same binary, reads the counters out of it, and reports the
three regimes (fill, warm-up, steady state); `sabah matrix` regenerates the
results matrix as JSON; `sabah gate` runs the correctness switches as pass/fail:

```powershell
python -m sabah.tools.cli run    <model.gguf> --tokens 256
python -m sabah.tools.cli matrix <model.gguf> --out results/matrix.json
python -m sabah.tools.cli gate   <model.gguf>
```

`sabah run` needs `SABAH_DIAG_EXE` (a built `llama-sabah-diag.exe`) and
`SABAH_RT_LIB`; `--diag` and `--rt` override both. The cache counters in every
report come from the runtime's exit snapshot, which is exact  -  so no timed run
sets `SABAH_LLAMA_STATUS_FILE` (the runtime rewrites that file after *every*
expert call, which costs ~0.3 s per token here and turns 14.6 tok/s into 1.9).

### Verification switches

These are the gate, not debugging aids. Run them before believing any number.

| variable | what it proves |
|---|---|
| `SABAH_LLAMA_VERIFY_BYTES=1\|fetch` | byte-compares served expert data (hits, or fetches) against the GGUF range |
| `SABAH_LLAMA_VERIFY_LIMIT` | caps those checks  -  the oracle reads the mapping and inflates host RAM |
| `SABAH_LLAMA_ID_CHECK=1` | forces the ids download even when reused and compares  -  proves the reuse byte for byte |
| `SABAH_LLAMA_TRACE=1` | prints cache, coverage and phase counters |
| `SABAH_LLAMA_IO_WORKERS=1` | reads each wave on the calling thread; `SABAH_IO` then reports `overlap=1.00` instead of ~3.8, with the same wall time |
| `SABAH_LLAMA_LAYER_MERGE=1` | one wave per layer instead of three; halves the wave count and costs ~13 ms/token (default off) |

Measured by `sabah gate` on the reference rig: `verify_ok=2000 verify_fail=0`,
`id_check_ok=768 id_check_fail=0`, exit code 0.

Two things about the run that prints those numbers. First, the `overflow` counter
it also prints (448) is **not** a correctness failure: `overflow` counts
over-budget expert allocations  -  when every eviction candidate is still
referenced by the call in flight, the tier allocates past its budget rather than
dangle a pointer  -  and the byte comparison above is what proves the routed result
stayed exact. The gate reports it and does not fail on it. Second, the exit code
is worth checking: every process that used the direct I/O wave pool used to end
in a fast-fail (`0xC0000409`) *after* the run and its trace had completed,
because the pool's worker threads were never joined. `sabah gate` was the first
thing here to check the exit code, and that is what found it; fixed in
`sabah/runtime/cuda/sabah_rt.cu`.

Third  -  read this before chasing a `differs from its GGUF bytes` failure. The
oracle reads the file through the OS mapping, i.e. the page cache; the runtime
reads it with `FILE_FLAG_NO_BUFFERING`, which the cache cannot answer. Those two
can disagree about the same range, and they did:

- 10 bytes of one 921600-byte expert, at offset 1,367,138,304 of the first shard,
  identical in three consecutive runs, and independent of the I/O path (same 10
  bytes at depth 1, at one worker, and at depth 2  -  so not a staging race);
- reproduced **outside the runtime and outside llama.cpp** with a plain buffered
  `read()` against an unbuffered `ReadFile` of the same range, while the mismatch
  line named the file, the offset and the byte positions;
- gone on the next run of the same binary, with no change to it.

So this failure means "the served bytes and the cached image of the file
disagree". It does not say which side is stale: a cached page can be older than
the disk, or newer than an unflushed write. What it does say is that the check
fired and that the case needs to be looked at  -  start with the two routes on that
range, not with the streaming path.

## Repository layout

```
sabah/runtime/cuda/sabah_rt.cu      the shipped streaming runtime: tier, direct I/O,
                                    pinned staging, kernels, IQ quant types
sabah/runtime/cuda/build_dll.bat    nvcc build (sm_120)
patches/llama.cpp/                  the pinned upstream base, the bridge patch, the diag patch
integration/llama.cpp/              reference copies of every llama.cpp file the bridge touches
tools/disk_probe.py                 direct-I/O rate/latency probe (produces the drive table)
tools/io_bench.cu                   C++ twin of it: one factor at a time (offsets, buffers,
                                    flags, locality, H2D copies, reader count)
tools/trace_locality.py             expert-reuse trace -> hit-rate curves
sabah/tools/bench_moe.py            per-block benchmark (research harness)
sabah/runtime/*.py                  Python research harness, see below
research_registry_v5.csv           experiment log: one row per hypothesis, the machine it
                                    ran on, the artefact and the verdict (PASS / FAIL)
docs/                               design notes, budgets, validation reports
results/                            raw JSON for every number quoted here; ~46 MB tracked.
                                    The raw per-token logit dump (logits.f32) is git-ignored;
                                    its SHA-256 is recorded in REPORT.md instead
```

## Python block harness (separate from the shipped path)

`sabah/runtime/*.py` is an earlier, independent research harness. It works on a
single block in isolation with synthetic traces, and its tables assume a **pinned
RAM warm tier**  -  the configuration the shipped runtime explicitly forbids. Its
`bench_moe` numbers are per-block and are not comparable to the end-to-end tok/s
above; the two must not be quoted together.

Use it for what it is good at: quant-kernel bit-exactness, planner hit-rate
curves, and prefetch-equivalence tests.

- `python -m pytest tests sabah\tests -q`
- `python sabah\tools\bench_moe.py --model $env:SABAH_MODEL --block 0 --tokens 500 --slots 16,32,64,128,256`

New IQ quant types implemented along the way, validated bit-exact against the
`gguf-py` reference: `IQ2_S`, `IQ3_S`, `IQ3_XXS`, `IQ4_XS`, `IQ4_NL`, `Q2_0`
(plus the existing `Q4_K`, `Q5_K`, `Q5_1`, `Q8_0`).

## Docs

| doc | contents |
|---|---|
| [`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md) | the three decode regimes (fill / warm-up / steady state), phase attribution, the no-disk floor, lever verdicts |
| [`docs/SECOND_TIER.md`](docs/SECOND_TIER.md) | lever 5 measured: what a second tier actually costs per expert on this rig |
| [`docs/LOCALITY_FINDINGS.md`](docs/LOCALITY_FINDINGS.md) | hit-rate and Belady measurements behind the slab size |
| [`docs/CONTEXT_VRAM.md`](docs/CONTEXT_VRAM.md) | context vs the expert slab: what 32K costs, the over-budget abort it can cause, and the working configurations found |
| [`patches/llama.cpp/README.md`](patches/llama.cpp/README.md) | bridge semantics, ABI, every environment variable |
| [`docs/ACCEL_LAYERS.md`](docs/ACCEL_LAYERS.md) | the A-D acceleration layers of the research harness |
| [`docs/V1_CORRECTNESS_CONTRACT.md`](docs/V1_CORRECTNESS_CONTRACT.md) | the correctness contract and its evidence |
| [`docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md`](docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md) | numerical-equivalence study |
| [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md), [`CHANGELOG.md`](CHANGELOG.md) | history |
| `docs/V1_*.md`, `docs/RC3_*.md` | point-in-time release/validation reports (historical) |

## What would actually move the needle

Ordered by measured leverage. **The regime decides which lever matters**: the fill
is paid once per fresh working set (11.2 s), the warm-up is the tier learning
(~9 GB), and the steady state is disk-free and only the floor can move it.

| # | lever | effect | cost/risk |
|---|---|---|---|
| 1 | **Gen4/Gen5 NVMe** | the *fill* is 12.68 GB at 1.11 GB/s = 11.2 s of the 35.8 s a 256-token answer takes; at 6 GB/s that is ~2.1 s and the answer goes 7.1 -> ~10.5 tok/s end to end. The steady state does not move (it is disk-free). Re-measure any candidate with fresh offsets (`tools/disk_probe.py`, `tools/io_bench.cu`) - the old 6 GB/s projection used a repeat-read ceiling | money. Still the largest single change for short answers |
| 2 | ~~**Close the exposure gap** (1.31 -> ~2.4 GB/s in the demand path)~~ **measured, no gap to close** | the 2.4 GB/s target was a repeat-read artifact of the old probe: with fresh offsets this drive reads ~1.05-1.45 GB/s and gains only ~15% from one reader to sixteen. The runtime reads at 1.11-1.46 GB/s | n/a. See `docs/DECODE_BUDGET.md` |
| 3 | ~~**Fewer bytes per token** - batch verification / speculative decoding~~ **measured, dead on this rig** | the "982 -> 676 MB/token" figure was residency demand, not miss traffic; measured bytes/position get *worse* with B (164.5 -> 169.6 MB). A 100%-acceptance pass is only 15% faster (161.3 vs 190.9 ms/position), so break-even needs 78-82% acceptance plus rollout, and this rig's `common_ngram_simple_draft` accepts **0.004 drafts/pass** on the model's own greedy stream (99.6% of passes accept nothing; n>=3 never): **2.8-7.7x slower** | high risk, measured net loss for speculation. **What survives: batching real concurrent requests.** The floor is `30.3 + 37/B` ms/position, so B=8 is 34.9 ms = 28.7 tok/s, and a server has independent requests without needing a drafter |
| 4 | ~~**The floor** (the steady state, measured)~~ **measured, and it moved twice** | The floor was 67.3 ms/token (disk removed) = `30.3` per-position work + `37/B` per-call cost. **Both halves have since moved on this rig**: a bit-exact quant kernel (TILE3) took the per-token cost 60.56 -> 44.21 ms, and segmented CUDA graphs removed GPU-side idle time to take the steady decode 42.74 -> **30.36 ms** (32.93 tok/s) at 128K, byte-identical. The floor is no longer fixed | done for this rig: kernel + graph, both byte-exact. Remaining: batching (lever 3) |
| 5 | ~~GTX 1080 as a second tier~~ **measured, net-negative** | a park + hit through the 1080 costs 0.454 ms per expert against the disk's 0.45 ms: break-even at a 100% hit rate, worse from there. No P2P on consumer parts; the 1080 sits at gen3 x4. A **host-RAM** tier is the one path that beats the disk (0.047 ms, 13.9 GB/s -> ~1.5x on the warm-up, nothing on the steady state) and it spends the RAM this project exists to avoid | n/a for the 1080. See `docs/SECOND_TIER.md` |
| 6 | Productisation: one-command launch, a harness that regenerates the whole results matrix as JSON, correctness as an automated gate | 0% speed, most of the project's remaining value | low. `sabah run`, `sabah matrix`, `sabah gate` |

## Related work, and why this is not compared to it

| project | what it is | how it differs here |
|---|---|---|
| [Glyd](https://github.com/antimatter15/glyd) | SSD-resident MoE inference on a
  consumer box, flash-attention and RAM-resident experts | different hardware class;
  **not measured by us** |
| [Strata](https://github.com/bdmendes/strata) | hierarchical SSD -> RAM -> VRAM
  caching for this model family | **not measured by us**; Strata's numbers are on
  their drives, which are not this drive |
| [GyroStrata](https://github.com/antimatter15/gyrostrata) | a fork/continuation of
  the Strata approach | **not measured by us** |
| [SGLang SSD Expert Pack](https://docs.sglang.ai/advanced_features/ssd_expert_pack.html) |
  SSD expert packing inside a serving stack | **not measured by us**; also a different
  integration model (server, many concurrent requests) |

Nothing in this README is a comparison against any of them. Every number here is
this rig, this drive (MSI SPATIUM M371, PCIe gen3 x4, measured 1.05-1.45 GB/s fresh
offsets) and this GPU (RTX 5060 Ti 16 GB). A faster drive or a different caching
policy would change every figure here.

## Status

**Proven:** end-to-end decode of an 83.6 GB MoE through patched llama.cpp, byte-exact
expert delivery, host RAM flat at 4.78 GiB with zero mapped fallbacks, and a
measured cost budget that attributes all three regimes of a generation to the
millisecond - including an independent measurement of the no-disk floor that
matches the steady-state decode rate.

**Done and measured:** the one-command harness (lever 6) - `sabah run` splits a
run into fill / warm-up / steady state, `sabah matrix` regenerates
`results/matrix.json`, `sabah gate` runs the correctness switches and passes on
the reference rig. Building it found and fixed a runtime bug: every process that
used the direct I/O wave pool fast-failed at exit (`0xC0000409`) because the
pool's threads were never joined. It also corrected the gate's misreading of the
`overflow` counter, and it caught a byte-level disagreement between the runtime
and the verification oracle that turned out to be a stale page cache rather than
a delivery error - the runtime now prints the file, offset and differing byte
positions when that check fires, and the README says how to tell the two apart.
The same release ships a one-click Windows installer: `install.bat` finds the
toolchain, builds the runtime for the detected GPU architecture and runs
`sabah doctor`, and `install.bat llama` does the same for the pinned llama.cpp
base and both patches.

**Not done:** a faster drive (lever 1). Levers 2, 3 and 5 are measured and closed
(the demand path is at the rate this drive serves it; speculative decoding is
2.8-7.7x slower with this drafter; the 1080 costs what the disk costs). Lever 4 is **done for
this rig**, and the numbers are REPORT 16: the bit-exact TILE3 kernel took the per-token cost
60.56 -> 44.21 ms, and segmented CUDA graphs took the steady decode 42.74 -> 30.36 ms
(32.93 tok/s) at 128K, byte-identical. What still moves it is batching real concurrent
requests (lever 3's surviving half), not another kernel.
Prefetching is implemented and correct but disabled; so is the merged layer fetch,
which halves the wave count and costs ~13 ms/token for it. The n-gram prefetch
layer of the research harness has a placeholder `index_fn`. The Python harness and
the shipped runtime have not been unified.

## License and third-party components

`LICENSE` (MIT) covers the Sabah code in this repository. The build also pieces
together third-party components, listed with their licences in
[`THIRD_PARTY.md`](THIRD_PARTY.md):

- **llama.cpp** - MIT, pinned at `96ffdc4`, with the Sabah patches under
  `patches/llama.cpp/` (each patch header records the pinned commit);
- **the model** - `ISTA-DASLab/Qwen3.8-Flash-Next` IQ3_S, not bundled; its licence
  and the quantisation quality numbers are the quantiser's, not ours;
- **CUDA / the NVIDIA driver** - used under their own licences, nothing
  redistributed here.
