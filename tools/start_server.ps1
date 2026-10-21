# Start a local chat server (browser GUI + OpenAI-compatible API).
#
#   powershell tools\start_server.ps1                      # f16 KV, 128K, the recommended setting
#   powershell tools\start_server.ps1 -Kv q8_0             # q8 KV (opt-in, see REPORT 28)
#   powershell tools\start_server.ps1 -Context 32768 -Slab 6.0e9
#
# Talks to:  http://127.0.0.1:8080            (browser chat)
#            http://127.0.0.1:8080/v1/chat/completions   (OpenAI-compatible API)
param(
    [string]$Model = "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    [int]$Context = 131072,
    [string]$Slab = "2.4e9",        # 128K needs the small slab; see docs/HOW_TO_RUN.md
    [string]$Kv = "f16",            # f16 = shipped; q8_0 = opt-in, 6.3x slower decode
    [int]$Port = 8080,
    [int]$Parallel = 2,
    [int]$Threads = 8,
    [switch]$NoGui
)

$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\quantization\Sabah-MoE-Accelerator"

$exe = (Resolve-Path "third_party\llama.cpp\build\bin\llama-server.exe").Path
$rt = (Resolve-Path "sabah\runtime\cuda\sabah_rt.dll").Path

# the same environment backend_launch() builds for the proxy, so a direct
# launch is the same backend -- plus -b and -ctk/-ctv, which the proxy cannot pass
$env:SABAH_LLAMA = "1"
$env:SABAH_RT_LIB = $rt
$env:SABAH_LLAMA_HOT_BYTES = $Slab
$env:SABAH_LLAMA_STATUS_FILE = Join-Path $env:TEMP "sabah_server_status.json"
if (Test-Path $env:SABAH_LLAMA_STATUS_FILE) { Remove-Item $env:SABAH_LLAMA_STATUS_FILE }
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"

$kvArgs = @()
if ($Kv -and $Kv -ne "f16") { $kvArgs = @("-ctk", $Kv, "-ctv", $Kv) }

$gui = @()
if ($NoGui) { $gui = @("--no-webui") }

$cmd = @(
    "-m", $Model,
    "--host", "127.0.0.1",
    "--port", $Port,
    "-c", $Context,
    "-ngl", "99",
    "--cpu-moe",
    "-t", $Threads,
    "-b", "2048",
    "--parallel", $Parallel
) + $kvArgs + $gui

Write-Host ""
Write-Host "=== Sabah local chat server ==="
Write-Host ("kv      : " + $Kv + ($kvArgs.Count -eq 0 -and $Kv -eq "f16" ? " (shipped default)" : " (opt-in)"))
Write-Host ("context : " + $Context)
Write-Host ("slab    : " + $Slab)
Write-Host ("browser : http://127.0.0.1:" + $Port + ($gui.Count -eq 0 ? " (chat page)" : " (API only)"))
Write-Host ("api     : http://127.0.0.1:" + $Port + "/v1/chat/completions")
Write-Host ""

if ($Kv -eq "q8_0") {
    Write-Host "note: q8 KV costs 6.3x decode here (REPORT 28); f16 is the fast path." -ForegroundColor Yellow
    Write-Host ""
}

& $exe @cmd
