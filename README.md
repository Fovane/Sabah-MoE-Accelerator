# Sabah-MoE-Accelerator

Sabah's own MoE inference runtime, optimized for **Qwen3.8-Flash-Next (qwen4exp)** on **24 GB VRAM + 16 GB RAM**, targeting **≥10–15 tok/s**.

## Target Artifact

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` — **IQ3_S** pair:
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf` (54.8 GB)
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf` (28.8 GB)

Expert bank: **50.29 GB** (24576 objects, 512 experts × 48 blocks), fixed path: **4.18 GB/token**. The model cannot fit in 24 GB VRAM under any quantization, so the path to target speed is **hiding transfer latency** (prefetch, warm tier, hot tier), not full residency.

## Hardware

| Component | Value |
|-----------|-------|
| GPU 0 | NVIDIA RTX 5060 Ti, 16 GB VRAM, sm_120 (Sabah runs here) |
| GPU 1 | NVIDIA GTX 1080, 8 GB VRAM, sm_61 (Sabah does NOT run here) |
| RAM | 16 GB DDR5 (~12 GB usable) |
| Disk | NVMe SSD, 1.56–2.2 GB/s sequential read |

## Acceleration Pieces (A–D)

| Piece | What it does | Status |
|-------|--------------|--------|
| **A — n-gram prefetch** | ~28.8 GB n-gram table mmap'd read-only; hinted rows fetched before forward pass. `index_fn` is still a placeholder (`rolling_hash_indices`). | Implemented, `index_fn` pending |
| **B — warm RAM tier** | Pinned RAM expert warm tier. VRAM refills read pinned RAM instead of SSD. `cudaHostAlloc` conflict fixed with per-expert pinned allocation. | ✅ Done |
| **C — fixed path resident** | Attention, router, embedding, output head (4.18 GB/token) pinned into VRAM. Never re-read from disk per token. | ✅ Done |
| **D — MTP expert prefetch** | FrequencyPredictor + advisory expert prefetch. Never changes which experts run; output-identical. | ✅ Done |

## New IQ Quant Kernels

Six new GGML quant types added with CUDA kernels, bit-exact vs gguf-py reference:

| Code | Type | Elems/block | Bytes/block |
|------|------|-------------|-------------|
| 18 | IQ3_XXS | 256 | 98 |
| 20 | IQ4_NL | 32 | 18 |
| 21 | IQ3_S | 256 | 110 |
| 22 | IQ2_S | 256 | 82 |
| 23 | IQ4_XS | 256 | 136 |
| 42 | Q2_0 | 64 | 18 |

Together with existing `Q4_K`, `Q5_K`, `Q5_1`, `Q8_0`, Sabah decodes **10 quant types**. Lookup tables are extracted from `ggml-common.h` by `tools/gen_iq_tables.py`.

## Setup

### 1. Clone

```powershell
git clone https://github.com/Fovane/Sabah-MoE-Accelerator.git
cd Sabah-MoE-Accelerator
```

### 2. Python environment

```powershell
python -m pip install pytest numpy
$env:SABAH_GGUF_PY = "C:\Projects\qwen4exp-llama\gguf-py"  # gguf-py fork (knows Q2_0=42)
```

### 3. Build CUDA DLL

Requires CUDA 12.8 + MSVC 14.44 (VS 2022 Professional):

```powershell
cd sabah\runtime\cuda
.\build_dll.bat sm_120
```

Verify:

```powershell
python -c "from sabah.runtime import rt; print(rt.available(), rt.device_count())"
# Output: True 2
```

### 4. Model

```powershell
$env:SABAH_MODEL = "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
$env:SABAH_TEST_MODEL = $env:SABAH_MODEL
$env:SABAH_SENTINEL_DIR = "C:\Projects\quantization\Sabah-MoE-Accelerator\build\sentinels"
```

## Usage

### Inspect model

```powershell
python sabah\tools\cli.py inspect %SABAH_MODEL%
```

### Execution plan

```powershell
python sabah\tools\cli.py plan %SABAH_MODEL% --context 4096 --concurrency 1
```

### Block benchmark (measured throughput)

```powershell
python sabah\tools\bench_moe.py --model %SABAH_MODEL% `
  --trace tests\data\bench_trace --block 0 --tokens 500 `
  --slots 16,32,64,128,256 --bank-mode ram --prefetch-slots 0 --pin-fixed-path
```

- `--bank-mode ram`: Pinned RAM warm tier (piece B), `host stage = 0 ms`.
- `--pin-fixed-path`: Pin fixed path into VRAM (piece C).

### Run all tests

```powershell
python -m pytest tests sabah\tests -q
```

### Test modules

- `sabah/tests/test_quant_kernels.py` — bit-exact validation of 6 new quant types vs gguf-py
- `tests/test_prefetch_equivalence.py` — prefetch off/on bit-identical (piece D gate)
- `tests/test_mul_mat_id_contract.py` — MUL_MAT_ID slot/row contract
- `tests/test_gpu_smoke.py` — GPU model-free slab/prefetch tests (9/9)

## Measured Results (Block 0, IQ3_S)

Fixed path pinned in VRAM + warm tier in RAM:

| Slots | VRAM | Hit Rate | ms/token | blk tok/s | Fetch MB/tok |
|-------|------|----------|----------|-----------|--------------|
| 16 | 0.035 GB | 3.2% | 2.848 | 351.1 | 20.86 |
| 32 | 0.070 GB | 6.6% | 2.870 | 348.5 | 20.13 |
| 64 | 0.139 GB | 13.1% | 2.678 | 373.5 | 18.73 |
| 128 | 0.279 GB | 25.3% | 2.342 | 427.1 | 16.11 |
| 256 | 0.557 GB | 50.0% | 1.753 | 570.5 | 10.79 |

Full-model projection (approximate): **8–12 tok/s**. Reaching 10–15 tok/s requires larger warm tier / VRAM capacity and higher hit rate.

## Architecture Notes

- **Full-model `llama-cli` does not work on this hardware**: the 84 GB model cannot be loaded by `llama.cpp` on 16 GB physical RAM (OOM). Even with a 128 GB pagefile, random-access expert reads from SSD are too slow.
- Sabah's yield is measured at **block-level benchmarks**; end-to-end `llama.cpp` integration exists under `integration/llama.cpp/` (patched bridge) but cannot be measured on this hardware due to RAM limits.
- `sabah_rt.dll` is built for **sm_120 only** (RTX 5060 Ti). GTX 1080 (sm_61) cannot run it.
- `SABAH_LLAMA_HOT_BYTES` env var controls VRAM hot tier size (default 2 GiB).

## Repository Structure

```
sabah/
  runtime/
    cuda/
      sabah_rt.cu        # CUDA kernels + IQ quant + slab allocator
      sabah_iq_tables.h  # Generated IQ lookup tables
      build_dll.bat      # VS 2022 Professional + nvcc build
      build_sentinels.bat # Test-only sentinel builds
    rt.py                # QTYPE/QBLOCK registries, host_register/unregister
    residency.py         # FixedPathResident, apply_plan (pieces C + B)
    expert_bank.py       # Expert bank, warm tier, file-handle I/O
    hot_tier.py          # VRAM hot tier + prefetch (pieces B + D)
    mtp_prefetch.py      # FrequencyPredictor, ExternalPredictor (piece D)
    ngram_prefetch.py    # Piece A (index_fn placeholder)
    executor.py          # MoEExecutor
  core/
    model_inspector.py   # Model inspection, QUANT_BLOCK, RUNTIME_QUANTS
  planner/
    planner.py           # Execution planner, hit-rate curves
  server/
    openai_proxy.py      # OpenAI-compatible proxy (sabah/reference backend)
  tests/
    test_moe_block.py
    test_planner.py
    test_quant_kernels.py
tools/
  gen_iq_tables.py       # IQ table generator from ggml-common.h
  bench_moe.py           # Per-block benchmark
  cli.py                 # Sabah CLI (inspect, plan, bench, serve, ...)
integration/
  llama.cpp/             # Patched llama.cpp bridge (sabah-bridge.cu)
patches/
  llama.cpp/             # 0001-sabah-native-mul-mat-id.patch, 0002-diag-capture
docs/
  ACCEL_LAYERS.md        # Acceleration layers documentation
results/
  bench_moe_blk0_iq3s.json
```
