# Decode cost budget (measured)

Rig: RTX 5060 Ti 16 GB, 15.38 GiB host RAM, NVMe. Model
`Qwen3.8-Flash-Next` IQ3_S (83.6 GB, 2 shards), 48 blocks x 512 experts,
top-10, expert bank 50.29 GB / 24576 objects.
Tool: `llama-sabah-diag`, `--cpu-moe -ngl 99`, direct I/O on, IO depth 8.
`SABAH_LLAMA_HOT_BYTES=9.8e9`.

## Where a decode token goes

Isolated by differencing two runs of the same binary (`-n 32` minus `-n 8`,
so the prefill cancels out) minus the phase totals:

| phase | per token | notes |
|---|---|---|
| `fetch` disk reads + H2D | **133 ms** | 174.7 MB/token over 262 misses (81.8% hit) |
| `ids` router download + stream drain | **23 ms** | 3 calls per layer, one drain per call |
| `slot` residency bookkeeping | **11.5 ms** | |
| `ptr` pointer table upload | **1.8 ms** | |
| graph + launch overhead | ~37 ms | |
| **decode wall** | **206 ms/token = 4.84 tok/s** | |

Prefill is not a problem: 1596 tokens in ~11 s (~145 tok/s).

## The binding constraint is the disk, at ~1.2-1.5 GB/s

174.7 MB/token of expert bytes must come off the disk, and the effective
unbuffered rate in this access pattern (random 676 KB blocks across an 83 GB
file) is **~1.2-1.5 GB/s**, and **adding concurrency does not raise it**: this
drive gains ~15% from one reader to sixteen. The isolated probe's 2.5 GB/s at
QD 8 is over *sequential* 1 MB reads, which is a different access pattern; its
*random* 676 KB case quoted 2.4-2.5 GB/s only because it re-read one offset
list, and drawn fresh that case is worth ~1.1-1.45 GB/s (see "Lever 2" below).

So the token floor at this tier is ~120 ms of disk, whatever else is overlapped.

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
| `fetch` disk reads + H2D | **121 ms** | 166.6 MB/token over 262.4 reads, 99.9 waves; 1.4 GB/s |
| `ids` router download + stream drain | **28 ms** | one drain per layer, 3 calls |
| `slot` residency bookkeeping | **9.5 ms** | |
| `ptr` pointer table upload | **1.5 ms** | |
| `prime` residency snapshot | **1.4 ms** | |
| outside the bridge (graph, launch, host-side ops) | **~44 ms** | not affected by `-t 4..12` or `-fa`; unprofiled |
| **decode wall** | **205.7 ms = 4.86 tok/s** | |

Byte-exact after all of it: `SABAH_LLAMA_VERIFY_BYTES=1` gives
`verify_ok=64191 verify_fail=0` with the same 28114 reads, the same
18968409600 bytes and zero mapped fallbacks.

So the disk is 59% of the token and it is already reading at the rate this drive
serves this pattern. The remaining levers are "fewer bytes" (a bigger tier, or
batch verification, which raises the per-layer demand) and the ~44 ms that is
spent outside the bridge entirely. **Lever 1 stays real but its projection must
be re-measured with fresh offsets**: "174.7 MB at 6 GB/s = 29 ms" assumed the
probe's repeat-read ceiling scaled linearly onto a faster drive.

## Can a bigger tier or "virtual RAM" get to 10-15 tok/s?

Byte requirement: decode demands 833 MB/token of expert bytes if nothing is
cached. Cached bytes are free (they are already in VRAM); every *uncached* byte
costs 1/1.3 GB/s.

| effective tier | hit rate | disk bytes/token | disk time | best-case wall (perfect overlap of 73 ms of other work) |
|---|---|---|---|---|
| 9.8 GB (today) | 81.8% | 175 MB | 133 ms | 133 ms |
| 17.8 GB | ~88% | ~100 MB | ~77 ms | ~77 ms |
| 27 GB | ~93% | ~58 MB | ~45 ms | ~45 ms |

- 10 tok/s needs ~100 ms/token => a ~18 GB tier **and** essentially perfect
  overlap. Marginal, and the second GPU only has 8 GB (Pascal, ~320 GB/s).
- 15 tok/s needs ~67 ms/token => a ~34 GB tier. **Not reachable on this machine.**

**Virtual memory does not help.** Windows "virtual memory" is the page file, which
lives on the same NVMe, so bytes served from it cost the same ~1.3-1.6 GB/s and
arrive through the page-fault path instead of a 676 KB unbuffered read at QD 8 -
strictly worse, and it cannot be prefetched with a controlled queue depth. It can
only add *capacity*, and capacity is not the problem: the tier below VRAM is
already the SSD.

The thing that *would* be fast is the OS page cache (RAM reads at ~10 GB/s), and
that is exactly what was ruled out: the expert path uses `FILE_FLAG_NO_BUFFERING`
precisely so host RAM stays free. Relaxing that constraint is the one change that
would move the needle by a lot - a host-RAM expert tier would serve ~83% of
demand from RAM instead of disk. That is a decision about the RAM budget, not an
optimisation.
