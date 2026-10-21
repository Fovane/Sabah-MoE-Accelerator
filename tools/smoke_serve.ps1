# Smoke test: start the OpenAI-compatible server and ask it one question.
$ErrorActionPreference = "Stop"
Set-Location "C:\Projects\quantization\Sabah-MoE-Accelerator"

$model = "C:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
$server = "third_party\llama.cpp\build\bin\llama-server.exe"
$rt = "sabah\runtime\cuda\sabah_rt.dll"

$env:SABAH_DIAG_EXE = (Resolve-Path "third_party\llama.cpp\build\bin\llama-sabah-diag.exe").Path
$env:SABAH_RT_LIB = (Resolve-Path $rt).Path

$log = "results\phase0\serve-smoke.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

$p = Start-Process -FilePath ".venv\Scripts\sabah.exe" `
    -ArgumentList @(
        "serve", $model,
        "--backend", "sabah",
        "--llama-server", (Resolve-Path $server).Path,
        "--context", "131072",
        "--hot-bytes", "2400000000",
        "--parallel", "2",
        "--port", "8080"
    ) `
    -RedirectStandardOutput $log -RedirectStandardError "$log.err" -PassThru -NoNewWindow

Write-Host "server pid $($p.Id) - waiting for /health ..."
$ok = $false
foreach ($i in 1..90) {
    Start-Sleep -Seconds 2
    try {
        $h = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/health" -TimeoutSec 5
        if ($h.status -eq "ok") {
            Write-Host ""
            Write-Host "health after ~$($i*2)s : $($h | ConvertTo-Json -Compress)"
            $ok = $true
            break
        }
    } catch {
        if ($i % 10 -eq 0) { Write-Host "  ... still loading ($($i*2)s)" }
    }
}

if (-not $ok) {
    Write-Host ""
    Write-Host "=== SERVER DID NOT COME UP ==="
    Get-Content "$log.err" -Tail 20
    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
    exit 1
}

$body = @{
    model = "sabah"
    messages = @(
        @{ role = "user"; content = "Merhaba. Turkce ve kisa cevap ver: 2+2 kacar?" }
    )
    max_tokens = 40
    temperature = 0.7
} | ConvertTo-Json -Compress

Write-Host ""
Write-Host "=== POST /v1/chat/completions ==="
try {
    $r = Invoke-RestMethod -Uri "http://127.0.0.1:8080/v1/chat/completions" `
        -Method POST -ContentType "application/json" -Body $body -TimeoutSec 600
    Write-Host ("model      : " + $r.model)
    Write-Host ("finish     : " + $r.choices[0].finish_reason)
    Write-Host ("answer     : " + $r.choices[0].message.content)
    Write-Host ("prompt tok : " + $r.usage.prompt_tokens)
    Write-Host ("compl tok  : " + $r.usage.completion_tokens)
} catch {
    Write-Host ("REQUEST FAILED: " + $_.Exception.Message)
    Get-Content "$log.err" -Tail 15
}

Write-Host ""
Write-Host "=== server still alive: $($null -eq $p.HasExited -or -not $p.HasExited) ==="
Write-Host "(left running so you can talk to it; Ctrl-C here does not kill it)"
$p.Id | Out-File "results\phase0\serve-smoke.pid.txt"
