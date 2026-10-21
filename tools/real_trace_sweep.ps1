param(
  [int]$Ctx = 131072,
  [string]$Slabs = "6.0e9,7.0e9,8.0e9",
  [string]$Tag = "real"
)
$ErrorActionPreference = "Continue"
$root = (Get-Item $PSScriptRoot).Parent.FullName
Set-Location $root
if (-not $env:SABAH_MODEL_DIR) { $env:SABAH_MODEL_DIR = Join-Path "C:" "Strata-data\models\IQ3_S" }
$model = Join-Path $env:SABAH_MODEL_DIR "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
$exe = Join-Path $root "third_party\llama.cpp\build\bin\llama-sabah-diag.exe"
$manifest = Join-Path $root "results\phase0\real-trace\manifest.jsonl"

foreach ($slab in $Slabs.Split(",")) {
  $name = "$Tag-$slab"
  $out = Join-Path $root "results\phase0\sweep\$name"
  New-Item -ItemType Directory -Force -Path $out | Out-Null
  $env:SABAH_LLAMA = "1"
  $env:GGML_OP_OFFLOAD_MIN_BATCH = "1"
  $env:SABAH_LLAMA_DIRECT_IO = "1"
  $env:SABAH_LLAMA_IO_DEPTH = "8"
  $env:SABAH_LLAMA_TRACE = "1"
  $env:SABAH_LLAMA_HOT_BYTES = $slab
  $env:SABAH_LLAMA_PREFETCH = "0"
  $env:SABAH_RT_LIB = Join-Path $root "sabah\runtime\cuda\sabah_rt.dll"
  $env:SABAH_DIAG_OUT = $out
  $env:SABAH_DIAG_MANIFEST = $manifest
  Remove-Item Env:SABAH_DIAG_NAMES -ErrorAction SilentlyContinue
  Remove-Item Env:SABAH_LLAMA_STATUS_FILE -ErrorAction SilentlyContinue
  $log = Join-Path $out "run.log"
  & $exe -m $model -dev CUDA0 -ngl 99 --cpu-moe -ot "\.per_layer_token_embd\.weight=CPU" -t 8 -c $Ctx -n 64 -p Hello 2>&1 |
    Tee-Object -FilePath $log | Out-Null
  "exit=$LASTEXITCODE" | Add-Content $log
  Write-Output "$name exit=$LASTEXITCODE"
}
