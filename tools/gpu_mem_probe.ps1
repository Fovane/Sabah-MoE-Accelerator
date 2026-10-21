param(
  [string]$Label = "vram-9.8",
  [int]$Ctx = 131072,
  [int]$Tokens = 128,
  [string]$Hot = "9.8e9"
)
$ErrorActionPreference = "Stop"
# the repo root is this script's parent directory, so no absolute path is baked in
$root = (Get-Item $PSScriptRoot).Parent.FullName
$out  = Join-Path $root "results\phase0\$Label"
New-Item -ItemType Directory -Force -Path $out | Out-Null

$env:SABAH_LLAMA = "1"
$env:GGML_OP_OFFLOAD_MIN_BATCH = "1"
$env:SABAH_LLAMA_DIRECT_IO = "1"
$env:SABAH_LLAMA_IO_DEPTH = "8"
$env:SABAH_LLAMA_TRACE = "1"
$env:SABAH_LLAMA_HOT_BYTES = $Hot
$env:SABAH_LLAMA_PREFETCH = "0"
$env:SABAH_RT_LIB = Join-Path $root "sabah\runtime\cuda\sabah_rt.dll"
$env:SABAH_DIAG_OUT = $out
Remove-Item Env:SABAH_LLAMA_STATUS_FILE -ErrorAction SilentlyContinue

$exe = Join-Path $root "third_party\llama.cpp\build\bin\llama-sabah-diag.exe"
# the model directory comes from SABAH_MODEL_DIR (tools/model_dir.py is the
# single source of truth); llama.cpp resolves shard 2 from the shard-1 suffix
if (-not $env:SABAH_MODEL_DIR) { $env:SABAH_MODEL_DIR = Join-Path "C:" "Strata-data\models\IQ3_S" }
$model = Join-Path $env:SABAH_MODEL_DIR "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"

function Read-Gpu {
  $a = Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
  $p = Get-Counter '\GPU Process Memory(*)\Dedicated Usage','\GPU Process Memory(*)\Shared Usage' -ErrorAction SilentlyContinue
  $ad = 0.0; $as = 0.0
  foreach ($s in $a.CounterSamples) {
    if ($s.Path -like '*dedicated usage*') { $ad += $s.CookedValue }
    elseif ($s.Path -like '*shared usage*') { $as += $s.CookedValue }
  }
  $od = 0.0; $os = 0.0; $mine = 0.0; $mines = 0.0; $n = 0
  foreach ($s in $p.CounterSamples) {
    $inst = $s.InstanceName
    $isMine = $inst -like ("pid_" + $script:pid_ + "_*")
    if ($s.Path -like '*dedicated usage*') {
      if ($isMine) { $mine += $s.CookedValue } else { $od += $s.CookedValue }
    } elseif ($s.Path -like '*shared usage*') {
      if ($isMine) { $mines += $s.CookedValue } else { $os += $s.CookedValue }
    }
    if (-not $isMine -and $s.CookedValue -gt 0) { $n++ }
  }
  [pscustomobject]@{ ad_mib = $ad/1MB; as_mib = $as/1MB; mine_ded = $mine/1MB;
                     mine_shared = $mines/1MB; other_ded = $od/1MB;
                     other_shared = $os/1MB; n_other = $n }
}

# baseline with nothing of ours running
$base = Read-Gpu

$log = Join-Path $out "driver.log"
$p = Start-Process -FilePath $exe -ArgumentList @(
  "-m", $model, "-dev", "CUDA0", "-ngl", "99", "--cpu-moe",
  "-ot", "\.per_layer_token_embd\.weight=CPU", "-t", "8",
  "-c", "$Ctx", "-n", "$Tokens", "-p", "Hello"
) -RedirectStandardOutput $log -RedirectStandardError "$out\driver.err" -PassThru -NoNewWindow
$script:pid_ = $p.Id

$csv = Join-Path $out "gpu_mem.csv"
"t_s,adapter_ded_mib,adapter_shared_mib,ours_ded_mib,ours_shared_mib,other_ded_mib,other_shared_mib,n_other" | Set-Content $csv
$sw = [Diagnostics.Stopwatch]::StartNew()
while (-not $p.HasExited) {
  $r = Read-Gpu
  Add-Content $csv ("{0:F2},{1:F1},{2:F1},{3:F1},{4:F1},{5:F1},{6:F1},{7}" -f `
    $sw.Elapsed.TotalSeconds, $r.ad_mib, $r.as_mib, $r.mine_ded, $r.mine_shared, $r.other_ded, $r.other_shared, $r.n_other)
  Start-Sleep -Milliseconds 400
}
$p.WaitForExit()
$p.Refresh()
"exit=$($p.ExitCode)" | Add-Content $csv
"baseline_before: adapter_ded={0:F1} adapter_shared={1:F1} other_ded={2:F1} n_other={3}" -f `
  $base.ad_mib, $base.as_mib, $base.other_ded, $base.n_other | Add-Content $csv
Write-Output ("{0} exit={1}" -f $Label, $p.ExitCode)
