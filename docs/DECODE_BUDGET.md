# Decode cost budget (measured)

Rig: RTX 5060 Ti 16 GB, 15.38 GiB host RAM, NVMe. Model
`Qwen3.8-Flash-Next` IQ3_S (83.6 GB, 2 shards), 48 blocks x 512 experts,
top-10, expert bank 50.29 GB / 24576 objects.
Tool: `llama-sabah-diag`, `--cpu-moe -ngl 99`, direct I/O on, IO depth 8.
`SABAH_LLAMA_HOT_BYTES=9.8e9`.

## What a generation actually costs: fill, warm-up, steady state

The number this project used to quote - "206 ms/token, 4.84 tok/s" - is the
marginal token of a generation that has just started, because it comes from
differencing short runs (`-n 32` minus `-n 8`, i.e. decode steps 9..32). A long
run prices the whole shape. Same prompt, same binary, only `-n` differs, and the
token stream is the model's own greedy tokens in every case:

| run | prefill | decode median | decode mean | decode sum | disk reads | disk GB | wall |
|---|---|---|---|---|---|---|---|
| `-n 1` (prefill only) | 11446 ms | - | - | - | 20087 | 12.68 | 20.1 s |
| `-n 128` | 11186 ms | 79.3 ms | 123.9 ms | 15.9 s | 34019 | 21.43 | 35.9 s |
| `-n 256` | 11156 ms | 70.1 ms | 96.2 ms | 24.6 s | 34598 | 21.81 | 46.1 s |

Differencing gives the three regimes exactly:

| regime | reads | bytes | per token | cost |
|---|---|---|---|---|
| **cold fill** (the 53-token prompt) | 20087 | 12.68 GB | - | **11.2-11.5 s, 1.11 GB/s** |
| **warm-up** (decode steps 1..127) | 13932 | 8.75 GB | 109.7 reads | 70.6 MB/token |
| **steady state** (steps 128..255) | 579 | 0.38 GB | **4.5 reads** | **3.0 MB/token** |

Step times by window (from a 257-step run) show the settling:

| steps | 1-32 | 33-64 | 65-96 | 97-128 | 129-160 | 161-192 | 193-224 | 225-256 |
|---|---|---|---|---|---|---|---|---|
| median ms | 203.9 | 122.4 | 72.2 | 71.5 | 68.3 | 68.4 | 67.5 | 66.9 |

So the machine does three different things, and averaging them into one tok/s
number hides all three:

* **Time to first token: ~11.2 s.** The prompt is 53 tokens; the 11 s is not
  prompt processing, it is the expert tier's **cold fill** - 12.68 GB of expert
  bytes at 1.11 GB/s. Nothing about the prompt is hard; the tier starts empty.
* **Warm-up: the first ~128 decodes.** The tier finishes discovering its working
  set (70.6 MB/token of disk), which is why the decode median is 204 ms at step
  32 and 79 ms at step 128.
* **Steady state: ~67 ms/token = ~15 tok/s, effectively disk-free** (4.5 reads,
  3.0 MB per token). After ~130 tokens the tier holds the working set and the
  disk is no longer on the critical path at all.

End to end, a 256-token answer costs 11.2 s + 24.6 s = 35.8 s, **7.1 tok/s
including the fill**, 10.4 tok/s over the decodes alone, and 15 tok/s once warm.
Which of those is "the" number depends entirely on whether the workload is one
long answer (steady state matters) or many short ones (the fill is paid every
time and dominates).

The same three regimes explain what each lever can buy, and the rest of this
document is per-regime:

| lever | cold fill | warm-up | steady state |
|---|---|---|---|
| a faster disk (lever 1) | **11.2 s -> ~3 s at 6 GB/s** | partially | nothing (already disk-free) |
| a bigger/second tier (lever 5) | cuts the bytes that miss | cuts the bytes that miss | nothing |
| batching (lever 3) | - | - | **the only thing that helps** |
| the floor itself (lever 4) | - | - | **it *is* the steady state** |

## Where a decode token goes during the warm-up

Isolated by differencing two runs of the same binary (`-n 32` minus `-n 8`,
so the prefill cancels out) minus the phase totals. **This prices decode steps
9..32, i.e. the warm-up, not the steady state** - see the section above for why
that distinction matters:

| phase | per token | notes |
|---|---|---|
| `fetch` disk reads + H2D | **121.8 ms** | 164.5 MB/token over 247.4 reads = 1.35 GB/s, at the drive's ceiling |
| `ids` router download + stream drain | **29.3 ms** | one drain per layer, 3 calls share it |
| `slot` residency bookkeeping | **9.3 ms** | 239 evictions/token, ~39 us each |
| `ptr` pointer table upload | **1.5 ms** | |
| `prime` residency snapshot | **1.3 ms** | |
| outside the bridge (graph, launch, host-side ops) | **27.7 ms** | residual: phases sum to 163.2 of 190.9 |
| **decode wall (sum over 24 tokens / 24)** | **190.9 ms** | decode *median* 205.7 ms |

By step 96 the same sum is **~70 ms**, and the disk portion of it is ~3 MB. The
tier is the whole story of the first two regimes; the floor is the whole story of
the third.

## The binding constraint is the disk, at ~1.2-1.5 GB/s - while it is on the path

The tier's cold fill must come off the disk, and the effective unbuffered rate in
this access pattern (random 676 KB blocks across an 83 GB file) is **~1.1-1.5
GB/s**, and **adding concurrency does not raise it**: this drive gains ~15% from
one reader to sixteen. The isolated probe's 2.5 GB/s at QD 8 is over *sequential*
1 MB reads, which is a different access pattern; its *random* 676 KB case quoted
2.4-2.5 GB/s only because it re-read one offset list, and drawn fresh that case is
worth ~1.1-1.45 GB/s (see "Lever 2" below).

So every byte the tier has not yet learned costs ~0.7 ms/MB, and a 12.68 GB cold
fill costs ~11 s before the model can answer anything.

## Raising the slab

| slab | hits/misses | decode bytes/run | tok/s | peak VRAM |
|---|---|---|---|---|
| 9.0 GB | 90653 / 28747 | 19.41 GB | 4.16 | 14566 MiB |
| 9.8 GB | 91285 / 28115 | 18.97 GB | 4.37 | 15356 MiB |

+0.8 GB buys ~2% fewer bytes. VRAM is essentially exhausted (16.31 GB card,
non-slab footprint ~5.4 GB).

## What was optimised, and what it bought

| change | cost before | after | tok/s |
|---|---|---|---|
| baseline (per-call drain, linear erase) | ids 87 ms | | 4.37 |
| router ids reused across a layer's 3 tensors | 87 ms | ~23 ms | 4.75 |
| single-scan residency memo per call | | | 4.79 |
| swap-and-pop eviction instead of `vector::erase` | slot 30 ms | ~11.5 ms | **4.84** |

The last two are the same two ideas the plan called for (per-layer id drain,
cheaper residency): together **4.37 -> 4.84 tok/s (+11%)**, still zero mapped
fallbacks, RAM flat at 4.78 GiB, peak VRAM 15.36 GB.

### Router id reuse

The three expert tensors of one MoE layer (`gate`, `up`, `down`) read the *same*
`selected_experts` tensor, so two of the three `cudaStreamSynchronize` per layer
are redundant. The bridge passes `dst->src[2]` (the ids ggml_tensor address) so
the runtime can tell "this is the same tensor as the call before" and skip the
copy and the drain. Groups are capped at `SB_ID_GROUP_MAX=4` calls, and that cap
is also the width of the eviction-protection window: entries from the last
`SB_ID_GROUP_MAX` epochs are never chosen as victims, which is what makes it safe
to free a slot while a kernel enqueued before the last drain may still read it.

*Proof:* `SABAH_LLAMA_ID_CHECK=1` forces every download anyway and compares the
reused bytes with the fresh ones: **`check_ok=3072 check_fail=0`**.
*Byte-exact:* with reuse active, `SABAH_LLAMA_VERIFY_BYTES=fetch
SABAH_LLAMA_VERIFY_LIMIT=500` gives **`verify_ok=500 verify_fail=0`**.

Removing the drains exposed a latent race: the pointer table is ONE device
buffer that every call rewrites, and the H2D used to go on the copy stream. The
per-call drain had been providing the ordering implicitly. The upload now runs on
the compute stream, so it is ordered after the kernel that is still reading the
same buffer. (A run with the racy order reported 9.90 tok/s and 23% fewer bytes;
those counters are not reproducible and that run must not be trusted.)

## Staging readahead (`SABAH_LLAMA_PREFETCH=1`, default off)

Shape: the prefetcher reads the *following* calls' experts into pinned staging and
leaves them there; the demand path copies them to their device slot and never
touches the disk. It reserves no tier space - an earlier shape that did reserve
made the demand path read its own reserved-but-unfilled slot a second time and
cost 2.8x the bytes (1.48 tok/s vs 4.16).

Two facts are learned rather than assumed:
* The call chain is keyed by the **expert tensor** address. Keying it by the
  router ids tensor was wrong: the graph allocator hands every layer the same
  address for that buffer, so the chain collapsed into one node claiming all 144
  tensors and the prefetcher pulled the whole bank (98% of bytes wasted).
* Whether the following call reuses the ids is learned by comparing the ids this
  call downloaded with the previous call's.

Correctness: the tier's `hits`/`misses`/`evictions` are **identical** to the
prefetch-free run, so the prefetch only moves bytes, never residency. Cost:
~8 x 2.7 MB extra pinned staging (~22 MB).

It does not pay: with the disk at ~1.3 GB/s and the demand path already able to
keep several reads in flight, moving bytes off the critical path changes nothing
(4.04-4.06 vs 4.16 at the time), and 40% of the readahead is superseded before
anyone asks for it (`stale=2313` of 5699). It becomes worth revisiting only if
the disk time drops below the bookkeeping time, which needs a bigger tier first.

## Lever 2, closed: the demand path is already at the drive's rate

The second lever in the plan was "close the exposure gap, 1.31 GB/s -> ~2.4 GB/s
in the demand path". The gap is not there. The ~2.4 GB/s target came from a
benchmark that - unintentionally, but unavoidably as it was written - re-read the
same offsets, and this drive serves a repeated read about 1.5x faster than a
fresh one. A fresh random 676 KB read workload is worth ~1.1-1.45 GB/s on this
hardware at any depth, and that is what the runtime already achieves.

**The drive, with fresh offsets every run** (`tools/io_bench.cu`: 676 KB reads -
the expert stride - 800 requests, whole repository file, pinned staging,
`FILE_FLAG_NO_BUFFERING`):

| readers in flight | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| fresh offsets, pass A | 1047 | 1136 | 1127 | 1196 | 1220 MB/s |
| fresh offsets, pass B | 1152 | 1293 | 1372 | 1457 | 1458 MB/s |
| **the same offsets re-read** | 1148 | 1617 | 1814 | 1809 | 1797 MB/s |

Rows two and three are back-to-back runs on the same drive; the only difference
is whether the offsets repeat. Any benchmark that draws one offset list and
reuses it - which is what `tools/disk_probe.py` did, for every queue depth it
reported - therefore quotes a number the demand path can never see, and that is
where the 2.4-2.5 GB/s in this document and in `LOCALITY_FINDINGS.md` came from.
`disk_probe.py` now redraws its offsets per case.

Nothing else moves the number: pageable vs `cudaHostAlloc` staging (the
allocation itself costs ~70 ms per process, which is the whole of the MB/s
difference), `FILE_FLAG_RANDOM_ACCESS` or not, and offsets clustered per wave
inside a moving 600 MB window - a layer's ten experts all live in one ~600 MB
tensor region, so this is the runtime's real locality - vs spread over the whole
file.

**What the runtime does with it.** Every configuration below reads the same
18.97 GB for the same run (`-n 32`, one shard, direct I/O):

| configuration | whole-run unbuffered time | decode median | `SABAH_IO overlap` |
|---|---|---|---|
| one overlapped file object, depth 1 / 8 / 32 | 1.42 / 1.35 / 1.35 GB/s | 205.9-207.7 ms | - |
| one synchronous lane behind each wave | 13713.9 ms | 205.8 ms | 1.00 |
| two lanes | 13663.7 ms | 203.1 ms | 1.78 |
| eight lanes | 13734.0 ms | 204.5 ms | 3.84 |
| eight lanes + merged layer fetch | 13657.0 ms | 218.0 ms | 4.22 |

`overlap` is the sum of the individual `ReadFile` durations divided by the wall
time the caller spent inside the waves, reported by `SABAH_IO`. At eight lanes
the reads really are four deep on average - the read calls sum to 52.8 s against
13.7 s of wave time - and the run is **not one millisecond faster** than the
one-lane run where they are strictly serial. That is the expected result now that
the drive's own curve is known: it gains ~15% from one reader to sixteen, so
~1.2-1.45 GB/s is what this access pattern is worth, and the runtime reads at
1.38-1.46 GB/s. It is at the top of the curve, not short of it.

The same reasoning retires the merged layer fetch (one wave per layer instead of
three, `SABAH_LLAMA_LAYER_MERGE`, default off): it halves the wave count exactly
as predicted, decode 99.9 -> 49.5 waves/token and prefill 5664 -> 4088, and the
time does not move by a millisecond, because the half of `fetch` that is not
transfer is not a per-wave charge on this drive. It is a net loss - concentrating
the layer's wait into one call leaves the GPU idle for a longer stretch, which
the next group's router drain then exposes - hence ~205 ms without it against
~218 ms with it. It stays in the code, defaulted off, because it is the shape
that would pay off on a wave deep enough for depth to matter.

**Current decode budget** (re-measured, same method: `-n 32` minus `-n 8`):

| phase | per token | notes |
|---|---|---|
| `fetch` disk reads + H2D | **121.8 ms** | 164.5 MB/token over 247.4 reads = 1.35 GB/s, at the drive's ceiling |
| `ids` router download + stream drain | **29.3 ms** | one drain per layer, 3 calls share it |
| `slot` residency bookkeeping | **9.3 ms** | 239 evictions/token, ~39 us each |
| `ptr` pointer table upload | **1.5 ms** | |
| `prime` residency snapshot | **1.3 ms** | |
| outside the bridge (graph, launch, host-side ops) | **27.7 ms** | residual: phases sum to 163.2 of 190.9 |
| **decode wall (sum over 24 tokens, / 24)** | **190.9 ms = 5.24 tok/s** | decode *median* is 205.7 ms; the tier keeps warming for ~50 tokens |

The same run's cache counters, which the trace line does not carry (`-n 32`
minus `-n 8` from the runtime's status JSON, a separate pass because
`SABAH_LLAMA_STATUS_FILE` rewrites a file on every call and inflates any timing
it touches):

| counter | per token | notes |
|---|---|---|
| calls | 144 | 48 layers x 3 tensors |
| lookups | 1440 | 10 experts x 144 |
| hits | 1192.6 | |
| misses | 247.4 | one 635 KB read each: 156.9 MB |
| evictions | 239.0 | the tier is full, so admissions evict almost 1:1 |
| overflow | 22.0 | eviction attempts that found only epoch-protected victims |
| resident | 9.13 GB of a 9.8 GB budget | cumulative hit rate 76.7%, marginal 82.8% |

Two passes of the identical configuration differ by ~5% in bytes/token (164.5 vs
156.9 MB); the miss *count* is stable (247.4 in both). Treat 5% as the noise
floor of this method.

Byte-exact after all of it: `SABAH_LLAMA_VERIFY_BYTES=1` gives
`verify_ok=64191 verify_fail=0` with the same 28114 reads, the same
18968409600 bytes and zero mapped fallbacks.

So the disk is 64% of the token and it is already reading at the rate this drive
serves this pattern. The remaining levers are "fewer bytes" (a bigger tier) and
the 67 ms of non-disk work, which is now measured rather than "unprofiled" (next
section). **Lever 1 stays real but its projection must be re-measured with fresh
offsets**: "174.7 MB at 6 GB/s = 29 ms" assumed the probe's repeat-read ceiling
scaled linearly onto a faster drive. With the floor below, the honest projection
is `157 MB / GB/s + 67 ms` instead of a byte count alone.

## The no-disk floor: what a perfect cache would still leave (lever 4, closed)

The diag can run several prompts in one process, clearing the KV between them but
**leaving the expert tier warm**. Feed the same forced token stream to every item
and the items after the first re-touch exactly the same `(tensor, expert)`
objects, so their disk demand collapses to nothing. Measured that way, with a
working set far under the tier (4-token prompt, 8 forced tokens, 6 items), items
2..6 read **0 bytes** and cost:

| positions per call (B) | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| pass wall (ms) | 67.3 | 99.1 | 156.8 | 278.8 |
| **per position (ms)** | **67.3** | **49.5** | **39.2** | **34.9** |

Fitted: `floor(B) ~ 30.3 + 37.0 / B` ms per position. The `30.3` is per-position
work (the expert arithmetic itself: ~1 GB of IQ3_S expert weights per token must
be read out of VRAM and dequantised); the `37 / B` is per-call cost, and it is
dominated by the per-layer router drain, not by residency: in the same shape at
B=1 the phase totals differenced across item counts give `ids ~50 ms/token` and
`ptr + prime ~1.8 ms/token`, with `slot ~0` (no evictions in a small working
set). The remaining ~15 ms is outside the phases: llama.cpp's graph walk, kernel
launches, and the per-step logits read the diag performs.

**This is the whole non-disk cost, and a long generation confirms it from the
other direction.** Two independent measurements land on the same number: this
replay, which removes every byte of disk traffic, costs 67.3 ms/token; and the
last 128 decodes of a 256-token generation, which read 3.0 MB/token, cost
67-68 ms/token. **The runtime's steady state is exactly its floor**, which also
means the steady-state rate cannot be improved by any storage change at all.

During the warm-up the two are additive: the measured 190.9 ms/token is
121.8 (disk) + 69.1 (everything else), agreeing to 1% with the replay. If the
runtime overlapped the drive with its own work there, the token would be
`max(121.8, 67.3) = 121.8 ms` instead of their sum. It does not, and that is
what the readahead section above is about.

Why it cannot be fixed by re-ordering:

* The dependence is real: layer L+1's router input is layer L's expert output, so
  the ids for L+1 cannot exist before L's kernels have run. One drain per layer
  is the floor for this design, and batching is the only thing that amortises it
  (`37/B` above, and `ids` 29.3 -> 11.9 ms/token from B=1 to B=8 in the
  verification probe).
* The drive is the bottleneck and its bandwidth is fully priced, so reads issued
  early are worth exactly what reads issued on demand are worth. Staging
  readahead therefore only pays with a near-perfect predictor, and the one that
  exists is 40% stale (`SABAH_LLAMA_PREFETCH`, section above), which is why it
  measured neutral-to-worse.

What this bounds, per regime (the numbers are per token of the regime named):

| change | warm-up token becomes | steady-state token becomes |
|---|---|---|
| today | 190.9 ms | 67 ms |
| perfect cache, zero disk traffic | 67.3 ms | 67 ms (already there) |
| a 6 GB/s drive | ~93 ms | 67 ms |
| a 3 GB/s drive | ~119 ms | 67 ms |
| any tier that removes 45% of the reads | ~124 ms | 67 ms |

And for the cold fill, which no batching or floor work touches:

| fresh tier | 12.68 GB cold fill at |
|---|---|
| today (1.11 GB/s measured) | 11.4 s |
| 3 GB/s | 4.2 s |
| 6 GB/s | 2.1 s |

So: no storage lever can change the steady-state decode rate - it is already the
floor - and the fill is paid once per fresh working set. Storage levers buy *time
to first token* and the warm-up, and the steady state needs the *demand or the
floor* to change, which is what levers 3 and the floor curve above are about.

## Lever 3, closed: batch verification and speculative decoding are dead here

The plan's "fewer bytes per token" was speculative decoding, on the theory that
offline analysis had shown the union of experts touched by a batch of 4 tokens to
be 676 MB/token against 982 MB for four sequential tokens - 31% less demand. That
number does not survive contact with the tier.

Measured cost of a *100%-acceptance* verification pass (the diag feeds the
baseline's own greedy tokens in groups of B as one `llama_decode`, so every
position is one a verifier would accept, and nothing is drafted or rolled back):

| B | pass wall | pass bytes | pass reads | ms/position | MB/position |
|---|---|---|---|---|---|
| 1 (sequential) | 190.9 | 164.5 | 247.4 | 190.9 | 164.5 |
| 2 | 339.8 | 330.9 | 497.6 | 169.9 | 165.4 |
| 4 | 648.8 | 669.7 | 1006.3 | 162.2 | 167.4 |
| 8 | 1290.8 | 1356.9 | 2037.7 | 161.3 | 169.6 |

Two things fall out. The best case is **15% faster per position** (B=8: 161.3 vs
190.9) and it comes entirely from amortising per-call overhead - `ids` 29.3 ->
11.9 ms/token, `waves` 96.5 -> 40.6 - not from fewer bytes. Bytes per position get
slightly *worse* with B (164.5 -> 169.6 MB), so the union figure was about cache
residency demand, not about what the tier actually misses.

Break-even, from the pass costs: with `P` passes per token and `A` the acceptance
rate, verification wins only if `A > (P * pass_cost / sequential) - 1`. At B=2
that is 78% of the 1 drafted token, at B=4 80% of 3, at B=8 82% of 7 - and that
ignores the re-decode a rejection costs, which pushes the real requirement to
85-90%.

The acceptance rate on this machine's own greedy stream, simulating upstream's
`common_ngram_simple_draft` exactly (n-gram size 2-6, 2-8 drafts per pass, over
256 tokens): **0.004 accepted drafts per pass at best, 99.6% of passes accept
nothing at all, and n >= 3 accepts literally never.** The prompt and the
continuation do not repeat token sequences, and that is what n-gram drafting
needs. Converting the measured pass cost to yielded tokens, including the
rejection re-decode:

| B | ms/token | vs sequential | MB/token | vs sequential |
|---|---|---|---|---|
| 2 | 528.7 | 2.77x slower | 493.4 | 3.00x |
| 4 | 836.5 | 4.38x slower | 830.9 | 5.05x |
| 8 | 1475.9 | 7.73x slower | 1515.4 | 9.21x |

Speculative decoding with this drafter costs 2.8-7.7x. It is not marginal, it is
not close, and the mechanism is not fixable by tuning n or m: the drafter's whole
premise is absent from the text this model is being asked to produce. A trained
drafter would change the arithmetic, but then B=8 has to accept 82% of seven
tokens, i.e. it needs to be right about the model's own greedy output almost
every time.

**What is not dead is batching itself.** The floor curve above says B=8 costs
34.9 ms/position against 67.3 at B=1 - 28.7 tok/s, and that is the number that
matters for the steady state, which no storage change can touch. Speculation is
one way to get a batch; *concurrent requests* are another, and a server does not
need to predict the model's own output to have them. `SABAH_DIAG_MULTISEQ` and
`n_parallel` exercise that path today. Two caveats before quoting 28.7 tok/s:
the probe's batch was one sequence's own tokens (so its expert sets were drawn
from a shared context, which is the friendlier case for the tier), and the
steady-state demand of N unrelated prompts is not measured here. The 34.9 ms
floor is solid; the disk part of a real concurrent batch is not.

## Can a bigger tier or "virtual RAM" get to 10-15 tok/s?

It already does, once warm: **15 tok/s is the steady state today**. What a bigger
tier buys is the *fill* and the *warm-up*, not the sustained rate. Byte
requirement: the warm-up demands ~833 MB/token of *distinct* expert bytes if
nothing is cached; cached bytes are free (they are already in VRAM); every
*uncached* byte costs 1/1.4 GB/s. The measured floor (67.3 ms) is **added** to the
disk time during the warm-up (it is not overlapped), so it appears in the wall
column:

| effective tier | hit rate | bytes/token in the warm-up | disk time | warm-up wall = disk + 67.3 ms | cold fill |
|---|---|---|---|---|---|
| 9.8 GB (today) | 82.8% | 110 reads/token (measured) | ~70 ms | ~140 ms (measured mean 124-96) | 11.4 s |
| 15.9 GB | 92.5% | ~48 reads/token | ~31 ms | ~98 ms | ~5 s |
| 27 GB | ~93% | ~45 reads/token | ~29 ms | ~96 ms | ~1.5 s |
| no disk at all | 100% | 0 | 0 | 67 ms (the floor) | 0 |

The hit-rate column is the measured LRU curve in
[`LOCALITY_FINDINGS.md`](LOCALITY_FINDINGS.md) (10.64 GB 86.4%, 15.96 GB 92.5%),
not an extrapolation.

- 15 tok/s is reachable today **once warm**, and nothing about the tier changes
  that: the floor is 67 ms/token and the steady state is already there.
- 10 tok/s over a *short* answer is the real target to aim at, and it is a
  **fill** problem: 12.68 GB at 1.11 GB/s = 11.4 s. A tier that is already warm
  across requests (a server, or a second tier in host RAM) is what removes it; so
  is a faster drive.
- 30 tok/s is reachable only by batching: `floor(B) = 30.3 + 37/B` puts B=8 at
  34.9 ms/position = 28.7 tok/s. Nothing else in this document moves the steady
  state.

**Virtual memory does not help.** Windows "virtual memory" is the page file, which
lives on the same NVMe, so bytes served from it cost the same ~1.3-1.6 GB/s and
arrive through the page-fault path instead of a 676 KB unbuffered read at QD 8 -
strictly worse, and it cannot be prefetched with a controlled queue depth. It can
only add *capacity*, and capacity is not the problem for the steady state: the
tier already holds the working set once warm.

The thing that *would* be fast is the OS page cache (RAM reads at ~10 GB/s), and
that is exactly what was ruled out: the expert path uses `FILE_FLAG_NO_BUFFERING`
precisely so host RAM stays free. Relaxing that constraint is the one change that
would move the needle on the *fill* - a host-RAM expert tier would serve the
warm-up's demand from RAM instead of disk, and the path is measured: **0.047 ms
per expert round trip through pinned host memory, 13.9 GB/s**
(`tools/second_tier_probe.cu`, see [`SECOND_TIER.md`](SECOND_TIER.md)). That is
9.6x cheaper than the disk's 0.45 ms, worth ~1.5x on the warm-up and nothing on
the steady state. It remains a decision about the RAM budget, not an optimisation,
and this project targets RAM-constrained machines.
