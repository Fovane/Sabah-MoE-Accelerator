param(
  [string]$Out = "results\phase0\real-trace\capture",
  [int]$Ctx = 131072,
  [string]$Hot = "6.0e9",
  [switch]$Capture
)
$ErrorActionPreference = "Stop"
$root = "C:\Projects\quantization\Sabah-MoE-Accelerator"
Set-Location $root
$outDir = Join-Path $root $Out
New-Item -ItemType Directory -Force -Path $outDir | Out-Null

$env:SABAH_LLAMA = "1"
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"
$env:SABAH_LLAMA_DIRECT_IO = "1"
$env:SABAH_LLAMA_IO_DEPTH = "8"
$env:SABAH_LLAMA_TRACE = "1"
$env:SABAH_LLAMA_HOT_BYTES = $Hot
$env:SABAH_LLAMA_PREFETCH = "0"
$env:SABAH_RT_LIB = Join-Path $root "sabah\runtime\cuda\sabah_rt.dll"
$env:SABAH_DIAG_OUT = $outDir
$env:SABAH_DIAG_MANIFEST = Join-Path $root "results\phase0\real-trace\manifest.jsonl"
if ($Capture) { $env:SABAH_DIAG_NAMES = "ffn_moe_topk" } else { Remove-Item Env:SABAH_DIAG_NAMES -ErrorAction SilentlyContinue }
Remove-Item Env:SABAH_LLAMA_STATUS_FILE -ErrorAction SilentlyContinue

$exe = Join-Path $root "third_party\llama.cpp\build\bin\llama-sabah-diag.exe"
$model = "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

$ErrorActionPreference = "Continue"
& $exe -m $model -dev CUDA0 -ngl 99 --cpu-moe -ot "\.per_layer_token_embd\.weight=CPU" -t 8 -c $Ctx -n 64 -p Hello 2>&1 |
  Tee-Object -FilePath (Join-Path $outDir "run.log")
"exit=$LASTEXITCODE" | Add-Content (Join-Path $outDir "run.log")
Write-Output "exit=$LASTEXITCODE out=$outDir"
