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
| decode | **4.84 tok/s** (206 ms/token) |
| prefill | ~145 tok/s (1596 tokens in ~11 s) |
| **host RAM** | **flat at 4.78 GiB**, never grows, `mapped_fallbacks=0` |
| peak VRAM | 15.36 GB of 16.31 GB |
| correctness | byte-exact against the GGUF ranges (`verify_ok=500 verify_fail=0`) |

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

Isolated by differencing two runs of the same binary (`-n 32` minus `-n 8`, so
prefill cancels out). Full method and attribution:
[`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md).

| phase | per token | notes |
|---|---|---|
| `fetch` — disk reads + H2D | **133 ms** | 174.7 MB/token over 262 misses, 81.8% hit |
| `ids` — router download + drain | **23 ms** | 3 calls per layer |
| `slot` — residency bookkeeping | **11.5 ms** | |
| `ptr` — pointer table upload | **1.8 ms** | |
| graph + launch overhead | ~37 ms | unprofiled |
| **decode wall** | **206 ms/token = 4.84 tok/s** | |

This budget is the honest state of the project: it says exactly what stands
between the runtime and its target, and it is the reason the target was re-scoped.

### The disk is the wall, and it is not yet saturated

| | |
|---|---|
| drive | MSI M371 1TB NVMe (Gen3 class) |
| drive ceiling measured | 2337 MB/s sequential, **2500 MB/s** random 1 MB @ QD 8 |
| per-request latency | **0.66 ms** (QD 1 random 1 MB is already 1568 MB/s) |
| Sabah achieves | **1310 MB/s** in the demand path |

So 174.7 MB/token should cost ~73 ms of pure transfer at the drive's own ceiling,
and it costs **133 ms**. The ~60 ms difference is **exposed latency**, not
saturation: the read for layer *L*'s experts cannot be issued until layer *L*'s
router has run, so the dependency chain is only about one layer deep and the
drive's queue is never filled. Raising IO depth cannot fix that, which is exactly
what the prefetcher experiment showed.

## What this project does not claim

- **It does not reach 10-15 tok/s on this machine, and it should not be described
  as if it does.** The original target was written before this budget existed.
  With the current algorithm and this drive the ceiling is
  `73 ms (perfect overlap) + 71.5 ms (everything else) = ~145 ms -> ~6.9 tok/s`.
  10 tok/s needs 100 ms/token, and 73 ms of that is disk at full efficiency.
- **It is not faster than a machine with enough RAM.** If host RAM were allowed to
  be a tier, ~83% of demand would be served from RAM at ~10 GB/s and the disk term
  would mostly disappear. Sabah trades that speed away for a hard guarantee about
  RAM — that is a deliberate budget decision, not a performance claim.
- **Virtual memory does not help.** Windows "virtual memory" is the page file on
  the same NVMe, reached through the page-fault path instead of a 676 KB
  unbuffered read at QD 8 — the same bandwidth, a worse access path, no
  controllable queue depth. It adds capacity, and capacity is not the constraint.
- **The GTX 1080 is not used.** `sabah_rt.dll` is built for `sm_120` only. A
  second tier on the 1080 would need cross-device copies over PCIe with no P2P on
  consumer parts; the projected ceiling is +18%, which has not been worth the risk.

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

- The chat template expands a trivial prompt to **~8.5k tokens** (`tokens=8484`
  for `-p "Hello"`), so a bare `llama-cli` run is prefill-dominated and its
  console summary is not the decode rate. It also does more work than the diag
  driver under identical flags (28.8 s vs 22.8 s wall, +35% bytes).
- **Quote decode from the diag driver.** Differencing `-n 32` and `-n 8` cancels
  the prefill and yields the 206 ms/token in the budget above.

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

Recorded results: `verify_ok=500 verify_fail=0`, `check_ok=3072 check_fail=0`.

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
| [`docs/DECODE_BUDGET.md`](docs/DECODE_BUDGET.md) | the full decode budget, disk analysis, tier projections |
| [`docs/LOCALITY_FINDINGS.md`](docs/LOCALITY_FINDINGS.md) | hit-rate and Belady measurements behind the slab size |
| [`patches/llama.cpp/README.md`](patches/llama.cpp/README.md) | bridge semantics, ABI, every environment variable |
| [`docs/ACCEL_LAYERS.md`](docs/ACCEL_LAYERS.md) | the A-D acceleration layers of the research harness |
| [`docs/V1_CORRECTNESS_CONTRACT.md`](docs/V1_CORRECTNESS_CONTRACT.md) | the correctness contract and its evidence |
| [`docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md`](docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md) | numerical-equivalence study |
| [`docs/IMPLEMENTATION_LOG.md`](docs/IMPLEMENTATION_LOG.md), [`CHANGELOG.md`](CHANGELOG.md) | history |
| `docs/V1_*.md`, `docs/RC3_*.md` | point-in-time release/validation reports (historical) |

## What would actually move the needle

Ordered by measured leverage, with the honest cost of each.

| # | lever | effect | cost/risk |
|---|---|---|---|
| 1 | **Gen4/Gen5 NVMe** | the disk-bound token is ~120 ms of 206. The 6 GB/s projection assumed the old probe's repeat-read ceiling; re-measure any candidate with fresh offsets (`tools/disk_probe.py`, `tools/io_bench.cu`) | money. Still the only single change that could reach the target |
| 2 | ~~**Close the exposure gap** (1.31 -> ~2.4 GB/s in the demand path)~~ **measured, no gap to close** | the 2.4 GB/s target was a repeat-read artifact of the old probe: with fresh offsets this drive reads ~1.05-1.45 GB/s and gains only ~15% from one reader to sixteen. The runtime reads at 1.38-1.46 GB/s | n/a. See `docs/DECODE_BUDGET.md` |
| 3 | **Fewer bytes per token** — batch verification / speculative decoding | measured union 982 -> 676 MB/token at B=4, i.e. ~31% less demand and proportional miss traffic | medium-high. Speculative *verification* keeps routing semantics exact, so it respects rule 2 |
| 4 | **The ~44 ms per token spent outside the runtime** | second only to `fetch`, and not moved by `-t 4..12` or `-fa`; measured, still unprofiled | low-medium |
| 5 | GTX 1080 as a second tier | +18% ceiling, ~54 ms/token of PCIe traffic | high risk, no P2P on consumer parts |
| 6 | Productisation: one-command launch, a harness that regenerates the whole results matrix as JSON, correctness as an automated gate | 0% speed, most of the project's remaining value | low |

## Status

**Proven:** end-to-end decode of an 83.6 GB MoE through patched llama.cpp, byte-exact
expert delivery, host RAM flat at 4.78 GiB with zero mapped fallbacks, and a
measured cost budget that attributes the token to the millisecond.

**Not done:** a faster drive (lever 1). Lever 2 is measured and closed - the
demand path is at the rate this drive serves it (see `docs/DECODE_BUDGET.md`), so
the remaining levers are fewer bytes per token (3) and the ~44 ms per token spent
outside the runtime (4). Prefetching is implemented and correct but disabled; so
is the merged layer fetch, which halves the wave count and costs ~13 ms/token for
it. The n-gram prefetch layer of the research harness has a placeholder
`index_fn`. The Python harness and the shipped runtime have not been unified.

## License

See [`LICENSE`](LICENSE).
