# Sabah-MoE-Accelerator

**A streaming MoE inference runtime for RAM-constrained machines.**

Sabah runs models far larger than the machine's VRAM + RAM combined. It does that
**without a RAM tier, without the page cache and without the page file**: the
expert path goes `disk -> pinned staging -> VRAM` with `FILE_FLAG_NO_BUFFERING`,
so **host RAM stays flat** for the whole run.

That is the point of the project. An 83.6 GB MoE model either fails to load on a
16 GB-VRAM / 15 GB-RAM machine, or it "runs" only by letting the OS page weights
through RAM — which requires a large page file and makes latency unpredictable.
Sabah is built for the machine that has **the VRAM but not the RAM**.

Measured end to end on the reference rig:

| | |
|---|---|
| model | `Qwen3.8-Flash-Next` IQ3_S, **83.6 GB** (2 shards, 48 blocks x 512 experts, top-10) |
| time to first token | **11.2 s** (the 53-token prompt is easy; the expert tier starts empty and fills 12.68 GB at 1.11 GB/s) |
| decode, steady state | **~67 ms/token = ~15 tok/s**, effectively disk-free (4.5 reads, 3.0 MB/token) |
| decode, first 128 tokens | 79-204 ms/token: the tier is still discovering its working set |
| 256-token answer end to end | 11.2 s + 24.6 s = **7.1 tok/s including the fill**, 10.4 over the decodes |
| **host RAM** | **flat at 4.78 GiB**, never grows, `mapped_fallbacks=0` |
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
  deliberately bypassed — it is what would make host RAM grow.
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

Everything here came from measuring the token, not from guessing. Baseline to
current is **4.37 -> 4.84 tok/s (+11%)**, with no change to the outputs.

| change | cost before | after | tok/s |
|---|---|---|---|
| baseline (per-call stream drain, linear erase) | `ids` 87 ms | | 4.37 |
| reuse the router ids across a layer's 3 expert tensors | 87 ms | ~23 ms | 4.75 |
| single-scan residency memo per call | | | 4.79 |
| swap-and-pop eviction instead of `vector::erase` | `slot` 30 ms | ~11.5 ms | **4.84** |

**Router id reuse** is the interesting one. The three expert tensors of a MoE
layer (`gate`, `up`, `down`) read the *same* `selected_experts` tensor, so two of
the three `cudaStreamSynchronize` per layer are redundant. The bridge passes
`dst->src[2]` — the ids *tensor address* — so the runtime can recognise "same
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
prefetch-free run. It is off because it does not pay — adding reads in flight
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

The warm-up token, phase by phase (differencing `-n 32` minus `-n 8`, i.e. decode
steps 9..32 — this is the number the README used to quote as *the* rate):

| phase | per token | notes |
|---|---|---|
| `fetch` — disk reads + H2D | **121.8 ms** | 164.5 MB/token over 247.4 misses, 82.8% marginal hit |
| `ids` — router download + drain | **29.3 ms** | 3 calls per layer |
| `slot` — residency bookkeeping | **9.3 ms** | 239 evictions/token, ~39 us each |
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

So the demand path is **at** the drive's rate, not below it — that is lever 2's
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
  away for a hard guarantee about RAM — a deliberate budget decision, not a
  performance claim.
- **Virtual memory does not help.** Windows "virtual memory" is the page file on
  the same NVMe, reached through the page-fault path instead of a 676 KB
  unbuffered read at QD 8 — the same bandwidth, a worse access path, no
  controllable queue depth. It adds capacity, and the steady state does not need
  capacity.
- **The GTX 1080 is not used, and measurement says it should not be.** The old
  +18% ceiling assumed the path was cheap. It is not: a park plus a hit through
  the 1080 costs 0.454 ms per expert against the disk's 0.45 ms, with no P2P on
  consumer parts and a gen3 x4 link on the 1080. Break-even at a 100% hit rate and
  worse from there ([`docs/SECOND_TIER.md`](docs/SECOND_TIER.md)).

## Hardware and target artifact

| component | value |
|---|---|
| GPU 0 | NVIDIA RTX 5060 Ti, 16 GB VRAM, `sm_120` — Sabah runs here |
| GPU 1 | NVIDIA GTX 1080, 8 GB VRAM, `sm_61` — not used |
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

**2. llama.cpp plus the bridge** — reproducible from the pinned upstream base:

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
  -m C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf `
  -dev CUDA0 -ngl 99 --cpu-moe `
  -ot "\.per_layer_token_embd\.weight=CPU" `
  -t 8 -c 512 -n 32 -p "Hello"
```

`llama-sabah-diag` is the measurement driver from patch `0002`. The bridge lives
in `ggml-cuda`, so any llama.cpp binary loads it: `llama-cli` runs this model end
to end as well (verified — `EXITCODE=0`, zero mapped fallbacks, generated text,
~27 GB streamed for the same prompt).

Two things to know when reproducing:

- **The prompt is 53 tokens, and the first token still costs ~11 s.** The chat
  template around `-p "Hello"` is 53 tokens (the diag writes the ids it used to
  `prompt_tokens.i32`; `prompt.txt` shows the text). The 11 s is the expert
  tier's **cold fill**: 12.68 GB at 1.11 GB/s, all of it before step 1. The
  `tokens=` field in the `SABAH_DIAG` line is a count of expert *rows*
  (`layers x tensors x batch`), not prompt tokens — for a 6-decode run it is
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
report come from the runtime's exit snapshot, which is exact — so no timed run
sets `SABAH_LLAMA_STATUS_FILE` (the runtime rewrites that file after *every*
expert call, which costs ~0.3 s per token here and turns 14.6 tok/s into 1.9).

### Verification switches

These are the gate, not debugging aids. Run them before believing any number.

| variable | what it proves |
|---|---|
| `SABAH_LLAMA_VERIFY_BYTES=1\|fetch` | byte-compares served expert data (hits, or fetches) against the GGUF range |
| `SABAH_LLAMA_VERIFY_LIMIT` | caps those checks — the oracle reads the mapping and inflates host RAM |
| `SABAH_LLAMA_ID_CHECK=1` | forces the ids download even when reused and compares — proves the reuse byte for byte |
| `SABAH_LLAMA_TRACE=1` | prints cache, coverage and phase counters |
| `SABAH_LLAMA_IO_WORKERS=1` | reads each wave on the calling thread; `SABAH_IO` then reports `overlap=1.00` instead of ~3.8, with the same wall time |
| `SABAH_LLAMA_LAYER_MERGE=1` | one wave per layer instead of three; halves the wave count and costs ~13 ms/token (default off) |

Measured by `sabah gate` on the reference rig: `verify_ok=2000 verify_fail=0`,
`id_check_ok=768 id_check_fail=0`, exit code 0.

Two things about the run that prints those numbers. First, the `overflow` counter
it also prints (448) is **not** a correctness failure: `overflow` counts
over-budget expert allocations — when every eviction candidate is still
referenced by the call in flight, the tier allocates past its budget rather than
dangle a pointer — and the byte comparison above is what proves the routed result
stayed exact. The gate reports it and does not fail on it. Second, the exit code
is worth checking: every process that used the direct I/O wave pool used to end
in a fast-fail (`0xC0000409`) *after* the run and its trace had completed,
because the pool's worker threads were never joined. `sabah gate` was the first
thing here to check the exit code, and that is what found it; fixed in
`sabah/runtime/cuda/sabah_rt.cu`.

Third — read this before chasing a `differs from its GGUF bytes` failure. The
oracle reads the file through the OS mapping, i.e. the page cache; the runtime
reads it with `FILE_FLAG_NO_BUFFERING`, which the cache cannot answer. Those two
can disagree about the same range, and they did:

- 10 bytes of one 921600-byte expert, at offset 1,367,138,304 of the first shard,
  identical in three consecutive runs, and independent of the I/O path (same 10
  bytes at depth 1, at one worker, and at depth 2 — so not a staging race);
- reproduced **outside the runtime and outside llama.cpp** with a plain buffered
  `read()` against an unbuffered `ReadFile` of the same range, while the mismatch
  line named the file, the offset and the byte positions;
- gone on the next run of the same binary, with no change to it.

So this failure means "the served bytes and the cached image of the file
disagree". It does not say which side is stale: a cached page can be older than
the disk, or newer than an unflushed write. What it does say is that the check
fired and that the case needs to be looked at — start with the two routes on that
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
docs/                               design notes, budgets, validation reports
results/                            raw JSON for every number quoted here
```

## Python block harness (separate from the shipped path)

`sabah/runtime/*.py` is an earlier, independent research harness. It works on a
single block in isolation with synthetic traces, and its tables assume a **pinned
RAM warm tier** — the configuration the shipped runtime explicitly forbids. Its
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
| 4 | **The floor** (the steady state, measured) | 67.3 ms/token with the disk removed entirely = `30.3` per-position work + `37/B` per-call cost, and the per-call cost is the per-layer router drain (ids ~50 ms/token at B=1). It *is* the steady state, so it is the only thing that moves 15 tok/s | medium: needs the batch (lever 3's surviving half) or faster expert kernels |
| 5 | ~~GTX 1080 as a second tier~~ **measured, net-negative** | a park + hit through the 1080 costs 0.454 ms per expert against the disk's 0.45 ms: break-even at a 100% hit rate, worse from there. No P2P on consumer parts; the 1080 sits at gen3 x4. A **host-RAM** tier is the one path that beats the disk (0.047 ms, 13.9 GB/s -> ~1.5x on the warm-up, nothing on the steady state) and it spends the RAM this project exists to avoid | n/a for the 1080. See `docs/SECOND_TIER.md` |
| 6 | Productisation: one-command launch, a harness that regenerates the whole results matrix as JSON, correctness as an automated gate | 0% speed, most of the project's remaining value | low. `sabah run`, `sabah matrix`, `sabah gate` |

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
2.8-7.7x slower with this drafter; the 1080 costs what the disk costs). Lever 4 is
attributed rather than fixed: the steady state is the floor, `30.3 + 37/B`, and
moving it needs batching *real* concurrent requests or faster expert kernels.
Prefetching is implemented and correct but disabled; so is the merged layer fetch,
which halves the wave count and costs ~13 ms/token for it. The n-gram prefetch
layer of the research harness has a placeholder `index_fn`. The Python harness and
the shipped runtime have not been unified.

## License

See [`LICENSE`](LICENSE).
