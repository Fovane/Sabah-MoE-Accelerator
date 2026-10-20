# A.5 detached competing-process / clock-pin benchmark.
# Runs three conditions and writes every result to files in results/phase0/a5-*,
# so it does not depend on the agent or the UI staying open.
#   1) as-is          (whatever is running right now)
#   2) after countdown (the user closes the browser / GPU users during it)
#   3) clocks pinned high (nvidia-smi -lgc 2800,3000 / -lmc 13000,14001), then restored
param([int]$Countdown = 60, [int]$Tokens = 256, [string]$Hot = "9.8e9")

$root = "C:\Projects\quantization\Sabah-MoE-Accelerator"
Set-Location $root
$out = Join-Path $root "results\phase0\a5"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$jsonl = Join-Path $out "results.jsonl"
$status = Join-Path $out "status.txt"
"started $(Get-Date -Format o)" | Set-Content $status
Remove-Item $jsonl -ErrorAction SilentlyContinue

function Run-Condition([string]$cond, [int]$n) {
    for ($i = 0; $i -lt $n; $i++) {
        $label = "a5-$cond-$i"
        & .\.venv\Scripts\python.exe tools\phase0_measure.py --label $label --tokens $Tokens --context 512 --hot-bytes $Hot > $null 2>&1
        $m = Get-Content (Join-Path $root "results\phase0\$label\measurement.json") | ConvertFrom-Json
        $rec = [pscustomobject]@{
            condition = $cond
            run       = $i
            steady_ms = [math]::Round($m.steady_ms_median, 3)
            tok_s     = [math]::Round($m.steady_tok_per_s, 3)
            warm_ms   = [math]::Round($m.regimes.warmup.median_ms, 3)
            fill_ms   = [math]::Round($m.regimes.fill_ms, 1)
            ids_us    = if ($m.calls -gt 0) { [math]::Round($m.phase_ids_ms / $m.calls * 1000, 0) } else { 0 }
            when      = (Get-Date -Format o)
        }
        ($rec | ConvertTo-Json -Compress) | Add-Content $jsonl
        "$cond run $i : $($rec.steady_ms) ms  $($rec.tok_s) tok/s" | Add-Content $status
    }
}

Run-Condition "asis" 3

"countdown_start $(Get-Date -Format o) close GPU apps now" | Add-Content $status
1..$Countdown | ForEach-Object {
    "countdown $($Countdown - $_ + 1)" | Set-Content (Join-Path $out "countdown.txt")
    Start-Sleep -Seconds 1
}
"countdown_end $(Get-Date -Format o)" | Add-Content $status

Run-Condition "closed" 3

nvidia-smi -lgc 2800,3000 2>&1 | Out-Null
nvidia-smi -lmc 13000,14001 2>&1 | Out-Null
"locked $(Get-Date -Format o)" | Add-Content $status
Run-Condition "locked" 3
nvidia-smi -rgc 2>&1 | Out-Null
nvidia-smi -rmc 2>&1 | Out-Null
"restored $(Get-Date -Format o)" | Add-Content $status
"done $(Get-Date -Format o)" | Add-Content $status
