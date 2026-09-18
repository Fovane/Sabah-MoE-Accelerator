# Sabah MoE Accelerator

![status: experimental](https://img.shields.io/badge/status-experimental-orange)
![routing: exact](https://img.shields.io/badge/expert%20routing-exact-brightgreen)
![format: GGUF / qwen4exp](https://img.shields.io/badge/format-GGUF%20%2F%20qwen4exp-blue)
![backend: CUDA](https://img.shields.io/badge/backend-CUDA%20%2F%20NVIDIA%20(currently)-76b900)

**Sabah accelerates routed-MoE inference without changing routing semantics.**

If the router selects expert *E*, expert *E* executes — not an approximation of
it, not a substitute, not a predicted stand-in. Everything Sabah optimises lives
*below* the model's mathematical function: placement, caching, scheduling and
data movement.

```
Inspect → Measure → Place → Cache → Stream → Execute Exact Experts
```

The two quantities the whole design turns on:

```
D_miss(C) = W · [1 − H(C)]                     bytes that must cross PCIe per token
T_token  ≈ T_fixed + T_expert + T_unhidden_transfer + T_runtime
```

`W` is the routed expert weight per token (1.5043 GB for this artifact), `H(C)`
is the byte-weighted hit rate of a hot tier of capacity `C`, measured from real
routing traces. Sabah's job is to raise `H(C)` and hide what is left of
`T_unhidden_transfer`; it is never to reduce `W` by skipping an expert.

Currently supported architecture: `qwen4exp` (Qwen3.8-Flash-Next family),
verified against the real artifact in Sabah v3/v4.

---

## Status — `v0.1.0-alpha`

**This is not a "download it and your model gets faster" release.** It is an
alpha of the inspection, qualification and planning layer. The accelerator
runtime — the thing that would actually make tokens come out faster — is
Phase B, and it is not finished.

What you can do today: point Sabah at a GGUF and at your machine, and get an
honest answer about whether this hardware could run it well, including the
answer "no".

This is **Phase A complete**, on a single-GPU development machine.

| component | state |
|---|---|
| model inspector + expert-layout verification | **working, tested on the real 111 GB artifact** |
| hardware qualification (GPU/CPU/RAM/storage/H2D) | **working, measured on this machine** |
| multi-GPU H2D qualifier (`mgpu_qualify.cu`) | **built and running**; multi-device path untested (one GPU here) |
| expert pipeline micro-benchmark (`expert_pipe_bench.cu`) | **built, measured** |
| execution planner + estimator | **working**, validated by synthetic machine-class tests |
| CLI (`inspect` / `qualify` / `plan` / `doctor` / `status`) | **working** |
| GPU hot-tier runtime | **not built** |
| OpenAI-compatible server | **not built** |
| correctness harness vs reference | **not built** |

Nothing in this repository claims a measured speedup. The planner emits
**projections**, always labelled, and they are replaced by measurements only
when `sabah benchmark` exists and has run on the machine in question.

## Quick start

```bash
python -m sabah.tools.cli doctor
python -m sabah.tools.cli inspect  <model>-00001-of-00004.gguf
python -m sabah.tools.cli qualify  <model>-00001-of-00004.gguf
python -m sabah.tools.cli plan     <model>-00001-of-00004.gguf
```

Once the CUDA runtime is built, the two commands that produce evidence rather
than estimates:

```bash
python -m sabah.tools.cli selftest <model>-00001-of-00004.gguf
python -m sabah.tools.cli bench    <model>-00001-of-00004.gguf --bank-mode ram
```

`selftest` refuses to pass unless the GPU decodes every expert quant type
bit-exactly and reproduces a CPU reference block. Run it before trusting any
speed number.

Building the CUDA pieces (`<cc>` is the GPU's compute capability without the
dot, from `nvidia-smi --query-gpu=compute_cap --format=csv,noheader`):

```bash
cd sabah/hardware && nvcc -O3 -o mgpu_qualify.exe mgpu_qualify.cu
cd ../runtime/cuda && nvcc -O3 -arch=sm_<cc> -shared -o sabah_rt.dll sabah_rt.cu
```

On Windows, put MSVC's `cl.exe` on PATH first, e.g.
`C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<ver>\bin\Hostx64\x64`.

## What the tools actually report

`inspect` verifies the property Sabah depends on, rather than assuming it:

```
arch       : qwen4exp   supported: YES
geometry   : 48 blocks, d_model 2560, 512 experts, top-10, expert_ff 640
expert bank: 77.018 GB in 24576 objects; per-expert 3,072,000/3,584,000/3,993,600 bytes
per token  : 1.5043 GB of routed expert weight
fixed path : 4.8302 GB (read every token)
contiguous : YES
note       : expert slices verified contiguous and quant-block aligned across 144 expert tensors
```

If an artifact's expert slices are not contiguous and quantization-block
aligned, Sabah refuses to accelerate it rather than applying Flash-Next
assumptions to an arbitrary MoE.

`qualify` measures rather than assumes, and separates *nominal* from *usable*:

```
ram        : 29.8 GB total, 25.5 GB usable, ~14 GB/s read
gpus       : 1
  [0] NVIDIA GeForce RTX 4050 Laptop GPU   6.44 GB total /  4.89 usable  gen4 x8  H2D 12.76 GB/s
h2d        : sum(solo) 12.76 GB/s | SIMULTANEOUS 12.79 GB/s | contention 1.00
             the planner uses the SIMULTANEOUS figure: 12.79 GB/s
```

**`sum(solo)` is never treated as aggregate bandwidth.** On a multi-GPU board
the simultaneous figure can be far lower, and the planner is required to size
against it.

`plan` is allowed to say no:

```
execution  : STORAGE_BACKED
expert tier: 0.00 GB total = 0.0% of the bank
ram bank   : DOES NOT FIT (77.0 GB)
PROJECTED  : 1.6-2.4 tok/s at concurrency 1 (confidence: low)
note       : absolute throughput is very low whatever the speedup: this machine
             is undersized for a 111 GB artifact.
```

## Measured, on one RTX 4050 Laptop (6 GB)

One MoE block of the real model, driven by a real 18,960-token routing trace,
1,000 tokens measured, 512 experts of 3.07 MB each:

| hot tier | hit rate | ms/token | GPU stall | host staging | fetched |
|---|---|---|---|---|---|
| 16 slots (3%) | 0.077 | 4.646 | 0.223 | 3.759 | 28.4 MB |
| 64 slots (12%) | 0.309 | 3.377 | 0.203 | 2.569 | 21.2 MB |
| 256 slots (50%) | 0.841 | 1.520 | 0.061 | 0.551 | 4.9 MB |
| 512 slots (100%) | 1.000 | 1.274 | 0.000 | 0.000 | 0 |

The bottom row is the kernel cost with **zero** expert transfer; everything
above it is the streaming penalty the hot tier exists to shrink.

Two findings came out of this that no simulator would have produced:

**1. With an mmap-backed bank, the bottleneck is not PCIe — it is the host.**
At 16 slots, 3.759 ms of the 4.646 ms is the CPU copying expert bytes from the
page cache into pinned staging. The GPU waits only 0.223 ms. Putting the bank
in **pinned RAM** lets the DMA engine read it in place and deletes that copy
entirely:

| hot tier | mmap bank | pinned RAM bank | speedup |
|---|---|---|---|
| 16 slots | 4.646 ms | **2.879 ms** | 1.61x |
| 64 slots | 3.377 ms | **2.186 ms** | 1.55x |
| 256 slots | 1.520 ms | **1.330 ms** | 1.14x |

With the staging copy gone the accounting closes: 1.274 ms of kernel plus
1.407 ms of measured GPU stall accounts for the 2.879 ms observed. And because
28.4 MB at the measured 12.76 GB/s would take 2.22 ms, the 1.407 ms stall means
roughly **37% of the transfer is genuinely hidden** behind compute.

**2. LRU beats static placement by more than the planner assumed.** Sabah v4
calibrated its hit curve on a *static* hot set. The runtime does not do that —
it seeds by popularity and then runs LRU. Re-measured on the same trace, same
byte weighting, same capacity budget (`sabah calibrate`):

| capacity | static, oracle | static, deployable | LRU (the runtime) |
|---|---|---|---|
| 8 GB | 0.361 | 0.321 | **0.621** |
| 16 GB | 0.552 | 0.500 | **0.781** |
| 32 GB | 0.755 | 0.698 | **0.912** |
| 48 GB | 0.871 | 0.818 | **0.955** |

LRU beats even an *oracle* static placement — one that knows the future token
distribution — by 0.13–0.19 across the range. The planner has been recalibrated
onto the LRU curve and now reports the static value alongside it as a
conservative floor, using it for the low end of every projection.

## Design, in one paragraph

The expert bank lives in system RAM as a backing store. The hot experts live in
VRAM, chosen by **global popularity ordering** and replaced by **plain LRU** —
both deliberately simple, because Sabah v3/v4 measured that LRU reaches 96.4% of
an offline **Belady reference** (farthest-next-use; a heuristic here, not a
proven optimum, because expert objects are not uniform in size) and that
near-optimal "water-fill" hot-set allocation beats global popularity by 0.0002. A miss stalls and fetches the correct expert;
nothing is ever substituted. Block L's router consumes block L−1's output, so
there is no cross-layer prefetch lead time and only intra-block overlap is
modelled. One transfer per block, one event: micro-chunking was measured to be
strictly worse (11.79 → 13.67 ms from 1 to 8 stages).

## Layout

```
sabah/
  core/       model_inspector      GGUF inspection, expert layout, contiguity proof
  hardware/   profile              versioned HardwareProfile
              mgpu_qualify.cu      solo + SIMULTANEOUS multi-GPU H2D
              expert_pipe_bench.cu copy/compute overlap and pipeline behaviour
  planner/    planner              ExecutionPlan, estimator, "no speedup" verdict
  runtime/    rt                   ctypes binding to the CUDA runtime
              expert_bank          per-expert slices out of the real GGUF
              hot_tier             VRAM residency: 4 states, LRU, telemetry
              executor             one MoE block, plus an independent CPU reference
              cuda/sabah_rt.cu     Q4_K/Q5_K/Q5_1/Q8_0 decode-and-matvec kernels,
                                   copy stream, events, allocation
  tools/      cli                  inspect / qualify / plan / selftest / bench /
                                   calibrate / doctor / status
              bench_moe            measured per-block throughput, M1 A/B
              calib_check          static-placement vs LRU hit curves
  tests/      test_planner         synthetic machine classes and invariants
              test_quant_kernels   decode vs gguf-py, on real bytes
              test_moe_block       exactness, discrimination, residency independence
```

State (hardware profile, caches) is written to `%LOCALAPPDATA%\Sabah`, or
`$SABAH_HOME` if set. **The original GGUF is never modified.**

## Models, privacy and your files

Sabah does not ship model weights and never will. You bring your own GGUF.

- **The original GGUF is opened read-only and is never modified.** Anything
  Sabah derives is written to its own state directory.
- **Nothing leaves the machine.** No prompts, no model contents, no hardware
  report, no routing traces are sent anywhere. There is no telemetry endpoint.
- Any future Hugging Face presence will host **Sabah profiles and configs** —
  measured hit curves, expert popularity orderings, architecture descriptors —
  and not weights.

## Not yet validated

Needs hardware this project does not have:

- simultaneous multi-GPU H2D aggregate and the PCIe contention factor
- fixed-path sharding (M2), whose inter-GPU traffic is not yet priced
- any machine on which the 77 GB expert bank actually fits in RAM

Needs more building:

- the full-model pipeline, and therefore any end-to-end tokens/s claim
- a correctness harness against llama.cpp at the logits/greedy-token level
  (block-level exactness is proven; whole-model equivalence is not)
- the hit curve is measured on one 18,960-token trace of 13 families; it
  drifted ~0.01 per doubling of trace length in v4 and should be expected to
  come in 1-3 points lower on longer traffic

See `docs/IMPLEMENTATION_LOG.md` for the full list.
