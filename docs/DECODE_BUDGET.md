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

## The binding constraint is the disk, at ~1.3-1.6 GB/s

174.7 MB/token of expert bytes must come off the disk, and the effective
unbuffered rate in this access pattern (random 676 KB blocks across an 83 GB
file) is **~1.3 GB/s in the demand path and still ~1.6 GB/s when a prefetcher
adds 8 more reads in flight**. The isolated probe sees 2.5 GB/s at QD 8 over
*sequential* 1 MB reads; the real pattern cannot reach it, and **adding
concurrency does not raise it** (measured, see the readahead section).

So the token floor at this tier is ~133 ms of disk, whatever else is overlapped.

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
