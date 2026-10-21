# Acceleration layers — what changed, and how to test it

Four changes were made to Sabah's own runtime. They are independent, and each
one has a test that fails if it is reverted.

| Piece | What it does | Files |
|-------|--------------|-------|
| **A — n-gram prefetch** | The n-gram table (~28.8 GB) is mmap'd read-only, the rows a token will read are hinted into the working set *before* the forward pass, and the previous token's rows are advised away. The table never competes with experts for RAM. | `sabah/runtime/ngram_prefetch.py` (new) |
| **B — warm RAM tier** | Leftover usable RAM is pinned as a per-block expert warm tier. A block's experts are then refilled from pinned RAM instead of the SSD. | `sabah/runtime/expert_bank.py` (`load_warm`), `sabah/planner/planner.py` (`warm_bytes`, `warm_blocks`) |
| **C — fixed path resident** | The fixed path is charged to VRAM *before* experts, is explicitly marked resident, and the remainder becomes the expert hot tier. A runtime helper pins the fixed tensors. | `sabah/planner/planner.py`, `sabah/runtime/residency.py` (new) |
| **D — MTP expert prefetch** | Speculative cross-token expert prefetch. Advice only: it can never change which experts run. | `sabah/runtime/mtp_prefetch.py` (new), `sabah/runtime/hot_tier.py` (`prefetch`, `mark_submitted`) |

## Two correctness fixes to earlier work

These are the reason the DLL must be rebuilt. Both were latent bugs, not
optimizations:

1. **A slot could be overwritten while a submitted kernel still read it.** After
   the earlier "remove the stream sync" change, an evicted slot was reused for
   the next expert's H2D before the kernel that read the old occupant had
   necessarily finished, which can silently corrupt a token. Fixed with a new
   `sabah_copy_wait_event` (copy stream waits on a compute-stream event) plus a
   per-block `mark_submitted` event. The llama.cpp bridge path keeps its
   conservative `cudaStreamSynchronize`, because that entry point does not
   launch the kernel itself and so has no point at which to record "the reader
   is done".
2. **The router-id download was made "async" but the host read it immediately.**
   Reverted to a synchronizing download: the residency decision needs the ids on
   the CPU, and no amount of stream juggling changes that. Overlapping it is the
   job of the MTP prefetcher (predict, don't wait).

The slab allocator was also rewritten as a real best-fit free-list with
coalescing; the earlier version handed the same base pointer to every expert.

---

## 1. Rebuild `sabah_rt.dll` (mandatory)

The `.cu` changes do not take effect until the DLL is rebuilt.

The build that works on this machine (CUDA 12.8 + Visual Studio) is:

```powershell
cd $WORKSPACE\sabah\runtime\cuda
.\build_dll.bat sm_120
```

`build_dll.bat` sources `vcvars64.bat` and then calls `nvcc`. Its only argument
is the target architecture; `sm_120` is the RTX 5060 Ti. Pass your own GPU's
compute capability (`nvidia-smi --query-gpu=compute_cap --format=csv`).

Verify it loaded:

```powershell
python -c "from sabah.runtime import rt; print(rt.available(), rt.device_count())"
```

Expected: `True 2`.

### Two environment failures that look like code problems

1. **`nvcc` cannot find `cl.exe`.** A plain PowerShell window is not a developer
   prompt. `build_dll.bat` sources `vcvars64.bat` for you; that is the fix. This
   is the normal first failure, not a broken install.
2. **`fatal error C1189: unsupported Microsoft Visual Studio version!`** CUDA
   12.8 refuses MSVC 14.5x. VS 18 Community ships 14.51 (rejected); VS 2022
   Professional ships 14.44 (accepted). `build_dll.bat` prefers VS 2022
   Professional and otherwise falls back to VS 18 with `-vcvars_ver=14.44`. If
   neither supported toolset is present, `nvcc` accepts
   `-allow-unsupported-compiler` at your own risk.

### If `nvcc` then fails with `cudafe++ died with status 0xC0000005`

This is an environment problem, not a code problem (it reproduces on a file
containing nothing but an empty `__global__`). In order of what usually fixes
it:

1. Confirm you are inside `vcvars64.bat` (see above) before blaming the toolkit.
2. Check for a stale/partial CUDA install and for antivirus intercepting
   `cudafe++.exe`. A fresh CUDA 12.8 or 13.3 toolkit resolves it in most cases.
3. Confirm the file at least parses: `nvcc -dryrun sabah_rt.cu` prints the
   command line without running the front end.

Nothing else in this document can be validated until the DLL builds, because
every runtime test calls `sabah_rt_init`.

---

## 2. Fast tests first (no GPU, no model)

These require only Python + numpy. They cover the pure-logic pieces and are the
quickest way to see whether the planner and the prefetch scaffolding behave.

```powershell
cd $WORKSPACE
python -m pytest tests\test_accel_layer.py -q
python -m pytest tests\test_public_contracts.py sabah\tests\test_planner.py -q
```

Expected: all pass. `tests\test_accel_layer.py` fails if the n-gram addressing
is off by the tensor's file offset, if warm-block selection stops preferring the
smallest blocks, if the planner stops capping expert capacity at the bank, if
the warm tier is subtracted from RAM using VRAM bytes, or if the prefetcher ever
touches the tier when disabled.

If pytest is not installed:

```powershell
python -m pip install pytest
```

---

## 3. GPU smoke tests (GPU, no model)

These need the GPU and the rebuilt DLL but **no GGUF model**, so they run even
when the model artifact Sabah needs is missing. They cover the transfer
machinery the acceleration work touched.

```powershell
python -m pytest tests\test_gpu_smoke.py -q
```

Expected: 9 passed. They fail if the slab allocator stops doing best-fit or
coalescing, if `sabah_copy_wait_event` stops being a legal copy-stream wait, if
a direct-DMA slot is not byte-exact, if prefetch spends more than the reserve,
if prefetch evicts live work, or if eviction corrupts a requested expert.

---

## 4. Correctness tests (GPU + model required)

> **The model matters.** Sabah's CUDA kernels implement exactly these expert
> quantization types: `Q4_K`, `Q5_K`, `Q5_1`, `Q8_0` plus the six the
> `GSQ-RCO-IQ3_S` artifact needs: `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_XS`,
> `IQ4_NL`, `Q2_0`.
>
> The previously validated artifact is
> `Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf` (gate/up `Q4_K` + one
> `Q5_K` block, down `Q5_1` + five `Q8_0` blocks) — see
> `results/rc4_numerical/environment.json`.
>
> The target artifact for this work is the GSQ-RCO IQ3_S pair,
> `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf` (83.6 GB total):
> gate/up `IQ2_S` / `IQ3_XXS` / `IQ3_S` / `IQ4_XS`, down `IQ4_NL` / `Q2_0`.
> `sabah inspect` now reports it as `supported: YES` — all six types are
> decoded by the CUDA kernels (they were added together with their kernels, so
> the inspector cannot claim support Sabah cannot deliver).
>
> Any other quant mix still needs its kernels added before it can run.

Set the paths once (two different variables are used by different tests):

```powershell
$env:SABAH_MODEL      = "$DRIVE\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
$env:SABAH_TEST_MODEL = $env:SABAH_MODEL
$env:SABAH_GGUF_PY    = "$DRIVE\Projects\qwen4exp-llama\gguf-py"
$env:SABAH_SENTINEL_DIR = "$WORKSPACE\build\sentinels"
```

`SABAH_GGUF_PY` must point at a `gguf-py` that knows the newer quant type codes
(`Q2_0 = 42`, `Q1_0 = 41`, `NVFP4 = 40`). A stock pip `gguf` raises
`ValueError: np.uint32(42) is not a valid GGMLQuantizationType` when it reads
those artifacts. The `qwen4exp-llama` fork's copy works.

### 4a. The quant kernels still read real weights correctly

```powershell
python sabah\tests\test_quant_kernels.py
```

This is unchanged by this work, but run it first: if dequantization is wrong,
every other number is meaningless.

### 4b. The MUL_MAT_ID contract (the llama.cpp-facing entry point)

```powershell
python -m pytest tests\test_mul_mat_id_contract.py -q
```

### 4c. **The prefetch equivalence test — the important one**

```powershell
python tests\test_prefetch_equivalence.py
```

This is the test that decides whether piece D is shippable. It replays a
routing sequence through a hot tier so small that it evicts on nearly every
step, twice: once with prefetch off, once with prefetch on and an **oracle**
predictor that always names the next step's exact experts. It asserts:

* the block outputs are **bit-identical** between the two runs;
* every slot resident at the end is **byte-identical** to its GGUF range (this
  is what catches the "overwrite a slot a kernel still reads" bug);
* prefetch actually fired (`prefetch_issued > 0`) and actually became hits
  (`prefetch_hits > 0`), so a pass is not a pass because the feature did
  nothing.

A non-zero exit means do not ship piece D. Send me the output.

Useful knobs:

```powershell
$env:SABAH_TEST_BLOCK = "0"     # which block to hammer
$env:SABAH_TEST_STEPS = "80"    # more steps = more evictions
```

---

## 5. Measuring whether it is faster

### 5a. Per-block streaming A/B, with and without prefetch

```powershell
python sabah\tools\bench_moe.py --model $env:SABAH_MODEL `
  --trace $WORKSPACE\tests\data\bench_trace `
  --block 0 --slots 16,32,64,128 --prefetch-slots 10
```

The trace file was vendored into the repo at `tests\data\bench_trace` because
the upstream path (`$DRIVE\sabah_scaling\v3\traces\main2`) does not exist on this
machine. It holds 3000 routing steps for block 0 (top-10 over 512 experts).
```

New columns:

* `pf iss` / `pf hit` — prefetch experts issued / used as hits, per token.

Compare the same command **without** `--prefetch-slots`. The prefetch rows
should show a higher `hit` rate and a lower `gpu stall`. If `pf hit` stays near
zero, the reserve is too small for the predictor to land anything — that is a
real finding, not a test failure.

To measure the warm-tier ceiling (piece B), compare `--bank-mode mmap` against
`--bank-mode ram`: the latter pins the block's experts and, because pinned RAM
needs no staging hop, the `host stage` column should collapse toward zero.

### 5b. Wiring piece B into a run

Nothing calls `load_warm` automatically; the planner only *chooses* the blocks.
For a full-model runner the wiring is two lines:

```python
from sabah.runtime.residency import apply_plan
apply_plan(plan(model, hw), bank)      # pins plan.warm_blocks in pinned RAM
```

or, at construction time:

```python
bank = ExpertBank(prof, mode="mmap", warm_blocks=plan(model, hw).warm_blocks)
```

`tests/test_accel_layer.py::test_apply_plan_pins_exactly_the_planners_warm_blocks`
and `::test_pinned_block_is_true_only_for_a_fully_pinned_block` cover this
without a GPU.

### 5c. The planner's view (no GPU needed)

```powershell
sabah plan $env:SABAH_MODEL
```

In the output look for:

* `fixed path : X GB resident in VRAM` — piece C;
* `warm tier  : Y GB pinned for N block(s)` — piece B.

On 24 GB VRAM + 16 GB RAM you should see a non-zero warm tier (roughly
`usable RAM − 2 GB`, capped by the bank) and a fixed path marked resident.
If the warm tier reads `budget (below one block; unused)`, the budget is
smaller than a single block and nothing will be pinned.

---

## 6. Wiring piece A (n-gram) into a real run

`sabah/runtime/ngram_prefetch.py` is complete and tested, but the **index
function is a placeholder**: `rolling_hash_indices` is a rolling polynomial
hash, not the model's n-gram scheme. A wrong schema costs bandwidth, never
correctness (prefetch is advice), so you can measure it as-is, but real hit
quality needs the model's actual n→row mapping.

Plumbing check without knowing the scheme:

```powershell
python -c "from sabah.core.model_inspector import inspect_model; print(inspect_model(r'%SABAH_MODEL%').describe())" 
```

The output should list `n-gram tab : <tensor>: <rows> rows x <bytes> B`.

To wire it in, give `NgramPrefetcher` an index function read from the model's
own code/config and call `prefetcher.step(token_ids)` at the point where the
layer-2 n-gram gather is about to happen.

---

## 7. What is deliberately *not* done

* **Eviction-based prefetch.** `HotTier.prefetch` fills only slots the
  `_ensure_locked` reserve keeps free. It never evicts, because evicting a slot
  a submitted kernel may read needs the same "copy waits for its reader"
  ordering that is now in place for misses; extending it to prefetch is a
  contained follow-up, not a silent risk.
* **A real MTP head.** `mtp_prefetch.py` defines the seam
  (`ExternalPredictor`) and ships a `FrequencyPredictor` baseline. The MTP head
  lives in the model/llama.cpp; feeding its candidates into
  `ExternalPredictor` is the integration point.
* **The full expert bank in VRAM.** For the `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`
  artifact the expert bank is 50.29 GB (24576 objects, 512 experts × 48 blocks),
  and the fixed path is a further 4.18 GB read every token, so a 24 GB card
  cannot hold either one whole. The path to the target speed is hiding transfer
  latency (prefetch, warm tier, hot tier), not making it all resident.
  Measured per-block kernel cost at full residency is ~0.65 ms/token for block
  0 (~1550 tok/s for that block alone); end-to-end tokens/s is bounded by how
  much of the bank stays resident versus how fast the disk feeds misses
  (measured 1.56-2.01 GB/s sequential on this machine's NVMe).
* **The projection model for the warm tier.** The planner reports the warm tier
  and uses it for the recommendation gate, but does not add a RAM→VRAM
  bandwidth term to the projected tokens/s. That would be inventing a number the
  benchmark should measure instead.
