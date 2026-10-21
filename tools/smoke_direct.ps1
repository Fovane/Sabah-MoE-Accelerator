# Smoke test the direct llama-server route (no proxy, no pinned warm tier).
$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\quantization\Sabah-MoE-Accelerator"

$log = "results\phase0\direct-smoke.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

$env:SABAH_LLAMA = "1"
$env:SABAH_RT_LIB = (Resolve-Path "sabah\runtime\cuda\sabah_rt.dll").Path
$env:SABAH_LLAMA_HOT_BYTES = "2.4e9"
$env:SABAH_LLAMA_STATUS_FILE = Join-Path $env:TEMP "sabah_direct_status.json"
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"

$exe = (Resolve-Path "third_party\llama.cpp\build\bin\llama-server.exe").Path

$p = Start-Process -FilePath $exe -ArgumentList @(
    "-m", "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
    "--host", "127.0.0.1", "--port", "8080",
    "-c", "131072", "-ngl", "99", "--cpu-moe",
    "-t", "8", "-b", "2048", "--parallel", "2"
) -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru

Write-Host "llama-server pid $($p.Id) - waiting for the model to load ..."
$ok = $false
foreach ($i in 1..120) {
    Start-Sleep -Seconds 3
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health" -TimeoutSec 5
        if ($h.status -eq "ok") {
            Write-Host ""
            Write-Host "healthy after ~$($i*3)s"
            $ok = $true
            break
        }
    } catch {
        if ($i % 10 -eq 0) { Write-Host "  ... loading ($($i*3)s)" }
    }
}

if (-not $ok) {
    Write-Host ""
    Write-Host "=== NO HEALTH ==="
    Get-Content "$log.err" -Tail 15
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$body = @{
    model = "sabah"
    messages = @(@{ role = "user"; content = "Merhaba. Turkce ve kisa cevap ver: 2+2 kacar?" })
    max_tokens = 48
    temperature = 0.7
} | ConvertTo-Json -Compress

Write-Host ""
Write-Host "=== POST /v1/chat/completions ==="
try {
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" `
        -Method POST -ContentType "application/json" -Body $body -TimeoutSec 900
    $sw.Stop()
    Write-Host ("model     : " + $r.model)
    Write-Host ("answer    : " + $r.choices[0].message.content)
    Write-Host ("usage     : prompt " + $r.usage.prompt_tokens + ", completion " + $r.usage.completion_tokens)
    Write-Host ("wall      : {0:N1}s  -> {1:N2} tok/s generation" -f `
        $sw.Elapsed.TotalSeconds, ($r.usage.completion_tokens / $sw.Elapsed.TotalSeconds))
} catch {
    Write-Host ("FAILED: " + $_.Exception.Message)
    Get-Content "$log.err" -Tail 12
}

Write-Host ""
Write-Host "=== /v1/models ==="
try { (Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/models" -TimeoutSec 10) | ConvertTo-Json -Compress } catch {}

Write-Host ""
Write-Host "server still running as pid $($p.Id) - close this shell or:"
Write-Host "  Stop-Process -Id $($p.Id)"
$p.Id | Out-File "results\phase0\direct-smoke.pid.txt"
