# Sabah Accelerator quickstart

Sabah currently supports the verified `qwen4exp` GGUF layout used by
Qwen3.8-Flash-Next UD-Q4_K_XL. The model remains user-supplied and is opened
read-only.

## Install

One click, from the repository root:

```bat
install.bat
```

It finds Python and MSVC, detects your GPU's compute capability, creates
`.venv`, installs the package, builds the runtime DLL for that architecture and
runs `sabah doctor`. Add `llama` to also fetch the pinned llama.cpp base, apply
both patches and build `llama-sabah-diag` (needs git + cmake + ninja, 10-40
minutes). By hand, the same thing:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

For CUDA validation, install the NVIDIA CUDA toolkit and ensure `nvcc` and
the MSVC developer environment are available.

## Inspect and qualify

```powershell
sabah doctor
sabah inspect $DRIVE\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
sabah qualify $DRIVE\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
sabah plan $DRIVE\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

## Build and verify the expert runtime

```powershell
cd sabah\runtime\cuda
nvcc -O3 -arch=sm_89 -shared -Xcompiler /MD -o sabah_rt.dll sabah_rt.cu
cd ..\..\hardware
nvcc -O3 -arch=sm_89 -o mgpu_qualify.exe mgpu_qualify.cu
cd ..\..
sabah selftest $DRIVE\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf
```

Replace `sm_89` with the installed GPU's compute capability. The self-test
uses real expert bytes and checks all qtypes present in the target artifact,
expert/block numerical agreement, wrong-expert discrimination, and
eviction/refetch independence.

## API

The full-model Sabah backend is not yet integrated. To use the local API in a
clearly labelled reference mode:

```powershell
sabah serve $DRIVE\models\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf --allow-reference
```

The default bind is `127.0.0.1`. The endpoints are `/health`, `/v1/models`,
`/v1/chat/completions`, and `/v1/completions`.

