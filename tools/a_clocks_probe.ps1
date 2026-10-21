param([string]$Label = "a-clocks2", [int]$Tokens = 256, [string]$Hot = "9.8e9")
$root = "C:\Projects\quantization\Sabah-MoE-Accelerator"
Set-Location $root
$out = Join-Path $root "results\phase0\$Label"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$csv = Join-Path $out "clocks.csv"

$sampler = @'
$csv = $args[0]
$sw = [diagnostics.stopwatch]::StartNew()
"t_s,sm_mhz,mem_mhz,pstate,power_w,temp_c,util_pct,throttle" | Set-Content $csv
while ($sw.Elapsed.TotalSeconds -lt 120) {
  $q = & nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>$null
  if ($q) {
    $t = "{0:F2}" -f $sw.Elapsed.TotalSeconds
    "$t,$q" | Add-Content $csv
  }
  Start-Sleep -Milliseconds 400
}
'@
$sf = Join-Path $out "_sampler.ps1"
Set-Content -Path $sf -Value $sampler
$proc = Start-Process powershell.exe -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$sf,$csv) -PassThru
Start-Sleep -Seconds 1
& .\.venv\Scripts\python.exe tools\phase0_measure.py --label $Label --tokens $Tokens --context 512 --hot-bytes $Hot 2>&1 | Select-Object -Last 3
Start-Sleep -Milliseconds 500
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
