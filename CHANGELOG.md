# Changelog

## Unreleased — the streaming path, measured per regime

Measurement, one new command, and a runtime bug the new command found. Nothing
here changes how experts are routed.

- **The decode number this project quoted was a warm-up artifact.** A generation
  has three regimes, and differencing `-n 32` and `-n 8` prices only the middle
  one. Measured on the reference rig: the fill (`-p "Hello"` is 53 templated
  tokens, but the 11.3 s is the expert tier's cold fill, 12.68 GB at 1.11 GB/s,
  not prompt processing), the warm-up (~128 decodes, 70.6 MB/token of disk,
  median 79 ms at step 128), and the steady state (**67 ms/token = ~15 tok/s**,
  3.0 MB/token, the disk effectively idle). A 256-token answer is 7.0 tok/s end
  to end. `docs/DECODE_BUDGET.md` now reports all three separately, because
  averaging them is how the project came to quote 4.84 tok/s.
- **One-command harness** (`sabah/tools/stream.py`, `sabah/tools/cli.py`):
  `sabah run` drives the shipped streaming path and splits its own per-step times
  into fill / warm-up / steady state; `sabah matrix` regenerates
  `results/matrix.json`; `sabah gate` runs the correctness switches as a
  pass/fail gate. The gate's checks are pure functions (`gate_checks`,
  `gate_notes`), so they are unit-tested offline (10 tests, no GPU needed).
- **Fixed: every process that used the direct I/O wave pool fast-failed at
  exit.** `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN/`__fastfail`) after the run
  had completed and its trace had been written. The pool's workers live in a
  `static std::vector<std::thread>` and nothing joined them, so the vector's
  destructor at process exit called `std::terminate` -> `abort`. The container is
  now heap-allocated and deliberately never destroyed — the workers are parked on
  a condition variable and the process reclaims them - and `sabah_rt_shutdown`
  joins them properly for a host that unloads the DLL. The pool is restartable
  now instead of `std::call_once`-once-only, so shutdown no longer costs a
  process its wave readers. Found by `sabah gate`, the first thing in this
  project that checked the exit code; the fix leaves the counters byte-identical
  (`reads=20362 bytes=13794278400` before and after). The CUDA runtime has to be
  rebuilt for it to take effect (`sabah/runtime/cuda/build_dll.bat sm_120`).
- **Fixed: the gate called `overflow` "an expert id escaped the model" and failed
  on it.** `overflow` counts over-budget expert allocations — the documented
  behaviour when every eviction candidate is still referenced by the call in
  flight, so the tier allocates past its budget rather than dangle a pointer. The
  routed result stays exact, which is what the byte comparison proves. Measured:
  448 over-budget allocations alongside 2000 ok / 0 fail verifications. The
  counter is now reported, not gated.
- **The verification oracle can be stale while the runtime is not, and now it
  says so.** The oracle byte-compares the served expert against the OS mapping of
  the GGUF, i.e. the page cache; the runtime reads the same file with
  `FILE_FLAG_NO_BUFFERING`, which the cache cannot answer. Those two disagreed by
  10 bytes of one 921600-byte expert (offset 1,367,138,304 of shard 1), the same
  10 bytes in three consecutive runs, at I/O depth 1, at one worker and at depth
  2 - so not a staging race. It reproduced outside the runtime entirely, with a
  buffered `read()` against an unbuffered `ReadFile` of that range, and it was
  gone on the next run of the same binary. A verification failure therefore now
  prints what it needs for that comparison: `SABAH_VERIFY_MISMATCH` carries the
  byte count, the number of differing bytes, the first differing positions and the
  resolved file and offset, and the host-visible error names the file, the offset
  and the caveat. The origin ring that makes that possible is only maintained
  while `SABAH_LLAMA_VERIFY_BYTES` is on, so the decode path pays nothing for it.
- **The harness no longer poisons its own timings.** `sabah matrix` used to run
  with `SABAH_LLAMA_STATUS_FILE` set to get the counters, which the runtime
  rewrites after every expert call: 144 synchronous writes per token, worth
  ~0.3 s per token on the reference rig (the steady state reported 1.86 tok/s
  instead of 14.6). The counters now come from the runtime's exit snapshot, which
  is exact and free, and a run directory is wiped of the previous run's step
  stream before each run so a killed run cannot be read back as the next one's.
- **One-click install on Windows** (`install.bat`). It finds Python and MSVC
  (`vswhere`, so any edition or Build Tools works), reads the compute capability
  of every installed GPU out of `nvidia-smi`, builds the runtime DLL for the
  highest one, creates `.venv` and installs the package, builds the optional H2D
  probe best-effort, and finishes with `sabah doctor` and the exact environment
  block to run a model. `install.bat llama` additionally fetches the pinned
  upstream commit from `third_party/llama.cpp.version` (shallow, by SHA), applies
  patches `0001` and `0002` only if they are not already applied, and builds
  `llama-sabah-diag` for every detected GPU architecture. It writes only inside
  the repository, never the registry, and never opens a model. `build_dll.bat`
  gained the same `vswhere` fallback, so it no longer assumes a specific Visual
  Studio edition.
- **Context size and the expert slab compete for the same VRAM, and at 32K the
  losing side is an abort, not a slow token.** `-c 32768` with the 9.8 GB slab
  loads and survives a one-token decode (peak 15,892 of 16,311 MiB), then dies
  once a longer run needs the tier's *documented* over-budget allocation: the
  slab is full, the extra `cudaMalloc` for that expert has no device memory left,
  the call returns null and the bridge aborts the process (`0xC0000409`,
  `Sabah over-budget expert: out of memory`). Reproduced 3 of 3 times, in the
  diag and in `llama-server`; the `overflow` counter shows the tier was already
  past its budget (485) when it failed. The context itself is cheap — ~0.5 GB of
  KV at 32K for this model, not the 3.22 GB the all-layer header arithmetic gives
  — it is the ~0.4 GB of headroom that is not. `-c 32768` works with an 8.0 GB
  slab, or with `-ub 8`. The measured peak table, the failure's evidence, and the
  `llama-server` HTTP path (which answers `/v1/chat/completions` once the ubatch
  is bounded, and matches the diag's own rate at the same length) are in
  `docs/CONTEXT_VRAM.md`. Writing it also re-measured a trap the harness was
  fixed for: `sabah serve` sets `SABAH_LLAMA_STATUS_FILE` unconditionally
  (`openai_proxy.py:273`), which costs ~0.3 s per token, so the same server
  request reports 2.04 tok/s with that file set and 5.16 tok/s without it.

## Unreleased — acceleration layers

Four runtime changes for the 24 GB VRAM / 16 GB RAM class, plus three bug fixes.
See `docs/ACCEL_LAYERS.md` for what each one is and how to test it. Not yet
benchmarked; the CUDA runtime must be rebuilt for any of it to take effect.

- **N-gram prefetch** (`sabah/runtime/ngram_prefetch.py`): the n-gram table is
  mmap'd read-only and its rows are hinted into the working set ahead of the
  gather, then advised away. It never occupies RAM the experts could use. The
  index function is injected; the built-in one is a labelled placeholder.
- **Warm RAM tier** (`ExpertBank.load_warm`, planner `warm_bytes` /
  `warm_blocks`): leftover usable RAM pins whole blocks, chosen smallest-first.
  A warm block is DMA'd straight from pinned RAM, removing both the SSD read and
  the host staging hop. `residency.apply_plan(plan, bank)` wires it in.
- **Fixed-path residency** (`sabah/runtime/residency.py`, planner
  `fixed_bytes` / `fixed_resident`): the fixed path is charged to VRAM before
  experts and is explicitly marked resident; expert capacity is now capped at
  the bank.
- **MTP expert prefetch** (`sabah/runtime/mtp_prefetch.py`, `HotTier.prefetch`):
  speculative cross-token prefetch, fenced as advice — it fills only reserved
  free slots, never evicts live work, and never changes which experts run. A
  prefetched expert the router asks for is used only after its copy is awaited
  on the compute stream.
- **Fixed: a reused slot could be overwritten while a submitted kernel still
  read it.** Added `sabah_copy_wait_event` plus a per-block `mark_submitted`
  event, so a copy waits for its reader precisely instead of draining the
  compute stream. The llama.cpp bridge keeps its conservative
  `cudaStreamSynchronize`, because it does not launch the kernel itself.
- **Fixed: the router-id download was made "async" while the host read it
  immediately.** Reverted to a synchronizing download; overlapping it is the
  prefetcher's job, not the download's.
- **Fixed the device slab allocator**, which could hand the same base pointer to
  every expert. It is now a best-fit free list with coalescing, and shuts down
  cleanly.
- **Fixed `rt.slab_free`**, which passed a `void` C function's result to
  `check()` and therefore raised on every successful free. `rt.slab_destroy` was
  added alongside it. (Found by the new GPU smoke test.)
- **Build fix**: `sabah/runtime/cuda/build_dll.bat` sources `vcvars64.bat` before
  `nvcc` and pins MSVC 14.44 when only VS 18 (MSVC 14.51) is installed, which
  CUDA 12.8 rejects.
- **New tests**: `tests/test_accel_layer.py` (24, no GPU) and
  `tests/test_gpu_smoke.py` (9, GPU but no model — slab, `copy_wait_event`,
  direct-DMA slot exactness, prefetch reserve, eviction exactness).

## 1.0.0

First release validated against a preregistered correctness contract
(`docs/V1_CORRECTNESS_CONTRACT.md`, frozen at `ca0a222` before any
validation prompt was run). Results: `docs/V1_VALIDATION_REPORT.md`,
`results/v1_validation/`.

- All hard gates pass: reproducible build; structural exactness (coverage,
  ids, bytes); float64 op-level ≤ 2e-6 (max 9.25e-7 over 27,972 samples);
  wrong-expert sentinels 12/12; residency invariance; 1/2/4-way concurrency;
  prompts across the ubatch boundary; full-graph non-inferiority (UB95
  R = 1.033 ≤ 1.25); greedy-flip non-inferiority (UB95 D = 0.0089 ≤ 0.03).
- **Slower than stock llama.cpp on the validation machine: 0.254× decode
  throughput (MEASURED).** v1.0 validates correctness, not speed.
- Runtime: in-runtime float64 self-check, exact status counters, byte
  verification, test-only sentinel builds, zero-row expert ops handled.
- Server: `--validate`, `--parallel`; no fixed backend timeout. Validation
  attempt 1 failed three API gates on a 600 s proxy timeout; fixed and rerun
  as the contract requires.
- Tooling: `tools/v1/` (driver, float64 oracle, API harness, evaluator,
  build check); `llama-sabah-diag` multi-prompt / multi-sequence modes.

## 0.9.0-rc4

Root cause of the RC3 drift found and fixed. Details and evidence:
`docs/RC4_NUMERICAL_EQUIVALENCE_REPORT.md`, `results/rc4_numerical/`.

- **Fixed a structural bug:** the native `MUL_MAT_ID` path fed every selected
  expert's down projection with expert slot 0's activation. Runtime ABI is now
  `sabah_rt_mul_mat_id_v2` (per-slot input rows); v1 is removed so a mismatched
  patch and library cannot load together.
- **Fixed:** decode crashed under CUDA-graph capture; CUDA graphs are now
  disabled for Sabah-routed `MUL_MAT_ID`.
- **Fixed:** ggml-cuda operator fusion could consume the host expert tensor
  directly (illegal memory access); fusion now never spans a Sabah-routed op.
- **Fixed:** a slot fetched earlier in a call could be evicted before that
  call's kernel ran when the tier was smaller than the call's working set.
- Found that RC3's "Sabah" run executed Sabah only for prompts of 32 or more
  tokens; decode ran on the CPU. `GGML_OP_OFFLOAD_MIN_BATCH=1` is now required
  and set by `sabah serve`.
- Added byte verification, coverage counters and a live status file; `/health`
  reports the runtime's own counters.
- Added `llama-sabah-diag` (patch 0002) and the `tools/rc4/` harness.
- Measured: Sabah's `MUL_MAT_ID` matches float64 math to 7.8e-7 (worst of 144
  ops); llama.cpp's CPU and CUDA backends are at 5e-3 to 3e-2 because they
  quantize activations to 8 bits. 16-token greedy equivalence with native CUDA
  passes; 64-token diverges at token 38, a contested step whose flip rate
  matches llama.cpp's own CPU-vs-CUDA variation.
- Measured speedup on the development machine: **0.30× vs stock llama.cpp**
  (Sabah is slower). v1.0.0 is not released.

## 0.9.0-rc3

- Added a reproducible llama.cpp integration patch pinned to upstream commit
  `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b`.
- Verified a clean CUDA llama.cpp checkout builds `llama-cli` and reaches the
  native Sabah `MUL_MAT_ID` bridge on the real Qwen3.8-Flash-Next GGUF.
- Added integrated router/weight/intermediate trace comparison artifacts.
- Kept v1.0 closed: CPU-reference versus CUDA expert arithmetic diverges on
  multi-token prompts and can change later router IDs; no speedup or API
  backend claim is made.

## 0.9.0-rc1

- Added explicit expert identities and exact range validation.
- Hardened the hot tier with serialized admission and a correctness fence
  before overwriting a slot that may still be read by the compute stream.
- Added byte-weighted hit telemetry and duplicate-request accounting.
- Added `sabah benchmark` with reference provenance and honest speedup gating.
- Added `sabah serve` as a localhost-only OpenAI-compatible reference-mode
  proxy; it refuses implicit fallback.
- Added packaging metadata, quickstart, architecture, benchmark, supported
  model, and limitation documentation.
