# Benchmark method

There are two deliberately separate measurements:

1. `sabah bench` measures one real MoE block, real expert bytes, a routing
   trace, and hot-tier residency. Its resident-vs-streamed A/B isolates the
   streaming penalty. It is not full-model tokens/s.
2. `sabah benchmark` records a comparable full-model reference invocation and
   its provenance. It prints `Measured speedup unavailable` until a full-model
   Sabah invocation under the same artifact, prompt, context, decoding,
   concurrency, and generation length exists.

Every result should retain the Sabah commit, model hash, runtime command,
hardware, execution plan, and raw timings. `MEASURED`, `PROJECTED`,
`SIMULATED`, and `ORACLE` are not interchangeable labels.

## Any disk number must draw fresh offsets

This drive serves a repeated read about 1.5x faster than a fresh one, so a
probe that draws one offset list and re-reads it for every case reports a
throughput the model can never see. On the reference rig (676 KB reads, the
expert stride, whole file): fresh offsets 1187-1293 MB/s at qd 1-8, the same
list re-read 2318-2320 MB/s at qd 4-8. Every disk number in this repository
before `disk_probe.py` learned to redraw per case - including the 2.4 GB/s that
levers 1 and 2 were projected from - is the second column.

`tools/disk_probe.py` draws fresh offsets per case (`--reuse-offsets` brings the
old behaviour back for comparison); `tools/io_bench.cu` takes a seed and reports
the sum of the individual read calls next to the wall time, so overlap and
throughput can be read separately. `SABAH_IO` prints the same overlap ratio for
the real runtime.

## Decode cost is differenced, never read off a single run

`-n` is `n_predict` and step 0 of the diag's step loop is the *prefill*, so
`-n 1` yields zero decodes. Every per-token number in this repository comes from
running the same configuration at `-n 8` and `-n 32` and dividing the delta by
24: the prompt, the warm-up and the load all cancel, and the tier's own warming
(`~50 tokens`) is spread across both runs instead of being charged to one.

`SABAH_LLAMA_STATUS_FILE` must **not** be set during a timing run: the runtime
rewrites that file after every `MUL_MAT_ID` call (~2 ms each, 144 calls/token).
Counters that only exist in it (hits, misses, evictions, overflow, resident
bytes) come from a separate counters-only pass; the counters are exact, the
timings from that pass are junk.

## A decode number belongs to a regime: fill, warm-up, steady state

Differencing short runs prices the **warm-up** - the marginal token of a
generation that has just started - and that is not the sustained rate. Same rig,
same prompt, same binary, only `-n` differing (53-token templated prompt):

| run | disk reads | disk bytes | what it is |
|---|---|---|---|
| `-n 1` (prefill only) | 20087 | 12.68 GB | the **fill**: 12.68 GB before the first token, 11.3 s at 1.11 GB/s |
| `-n 128` minus `-n 1` | 13932 | 8.75 GB | the **warm-up**: 109.7 reads / 70.6 MB per token |
| `-n 256` minus `-n 128` | 579 | 0.38 GB | the **steady state**: 4.5 reads / 3.0 MB per token |

The fill reads more than the 9.8 GB tier holds because the tier is still evicting
while it fills; that is why it is 12.68 GB, not 9.8 GB. The steady state needs no
disk at all, which is why no storage number can be quoted as "the decode rate".

So quote every rate with its window, or quote all three. `sabah run` performs the
split itself from `step_ms.f64` (step 0 is the fill; `regimes(out_dir,
warm_tokens=128)` is the boundary), and `step_ms.f64` is written in the normal
path too, not only under `SABAH_DIAG_MANIFEST`.

## The verification oracle reads the cache; the runtime does not

`SABAH_LLAMA_VERIFY_BYTES` compares the served expert bytes against the GGUF
through the **memory mapping** (the page cache). The runtime's own reads use
`FILE_FLAG_NO_BUFFERING`, which bypasses the cache. The two are allowed to
disagree about the same range:

| observed | value |
|---|---|
| object | one 921600-byte expert, offset 1,367,138,304 of shard 1 |
| disagreement | 10 bytes, isolated, in a 170-byte window (`first=635221..635389`) |
| runs | the same 10 bytes in three consecutive runs |
| I/O depth 1 / one worker / depth 2 | the same 10 bytes (so not a staging race) |
| outside the runtime | a buffered `read()` vs an unbuffered `ReadFile` of the same range disagree identically |
| next run of the same binary | agrees again, no rebuild |

So a verification failure means "the served bytes and the cached image of the
file disagree"; it does not by itself say which side is stale - a cached page can
be older than the disk (a page that survived a rewrite) or newer than an
unflushed write. When it fires, compare the two routes on that range before
suspecting the streaming path. `SABAH_VERIFY_MISMATCH` prints the byte count, the
number of differing bytes, the first differing positions and the resolved file
and offset, and the host-visible error names them too.

## The no-disk floor: replay one stream against a warm tier

To measure what a decode costs with the disk removed, run several prompts in one
process with `SABAH_DIAG_MANIFEST` (the KV is cleared between items, the expert
tier is not) and give every item the *same* `SABAH_DIAG_FORCE_DIR/<id>/fed.i32`
token stream. Items after the first re-touch exactly the same `(tensor, expert)`
objects, so their disk demand collapses: with a working set far under the tier
they read literally 0 bytes, and their step times are the floor.

Two rules make it trustworthy:

* The working set of one item must fit the tier comfortably (a 4-token prompt and
  8-64 forced tokens; not the real prompt's 53 tokens plus a long stream), or the
  items thrash against each other and the "warm" items are not warm. Confirm with
  the trace: the whole-run `SABAH_DISK` byte count must equal one item's cold
  demand, i.e. it must not grow with the item count.
* In the normal path `step_ms[0]` is the prefill and must be dropped; in the
  batch-probe path (`SABAH_DIAG_BATCH_PROBE=B`) every `step_ms` entry is one
  pass, including the first. Dropping the first entry there silently deletes a
  measurement (it once turned the B=8 case into a column of zeros).

`measure_l4_floor2.ps1` in the working notes is the reference script: 6 items x
8 tokens at B = 1, 2, 4, 8, reporting per-position cost with the cold item
excluded.


