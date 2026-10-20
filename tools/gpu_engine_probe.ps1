param([string]$Label = "a-engines", [int]$Tokens = 256, [string]$Hot = "9.8e9")
$root = "C:\Projects\quantization\Sabah-MoE-Accelerator"
Set-Location $root
$out = Join-Path $root "results\phase0\$Label"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$raw = Join-Path $out "engines.csv"
$map = Join-Path $out "pidmap.csv"

$sampler = @'
$raw = $args[0]
$map = $args[1]
"t_s,pid,engtype,util" | Set-Content $raw
$sw = [diagnostics.stopwatch]::StartNew()
$seen = @{}
while ($sw.Elapsed.TotalSeconds -lt 90) {
  $t = "{0:F2}" -f $sw.Elapsed.TotalSeconds
  try {
    $c = Get-Counter '\GPU Engine(*)\Utilization Percentage' -ErrorAction Stop
    foreach ($s in $c.CounterSamples) {
      $inst = $s.InstanceName
      if ($inst -match 'pid_(\d+)_.*engtype_(\w+)') {
        $pidv = $Matches[1]; $eng = $Matches[2]
        $u = [math]::Round($s.CookedValue, 2)
        if ($u -gt 0.05) { "$t,$pidv,$eng,$u" | Add-Content $raw }
        if (-not $seen.ContainsKey($pidv)) {
          $seen[$pidv] = $true
          try { $pname = (Get-Process -Id $pidv -ErrorAction Stop).ProcessName } catch { $pname = "?" }
          "$pidv,$pname" | Add-Content $map
        }
      }
    }
  } catch {}
  Start-Sleep -Milliseconds 500
}
'@
$sf = Join-Path $out "_engsampler.ps1"
Set-Content -Path $sf -Value $sampler
$proc = Start-Process powershell.exe -ArgumentList @("-NoProfile","-ExecutionPolicy","Bypass","-File",$sf,$raw,$map) -PassThru
Start-Sleep -Seconds 1
& .\.venv\Scripts\python.exe tools\phase0_measure.py --label $Label --tokens $Tokens --context 512 --hot-bytes $Hot 2>&1 | Select-Object -Last 2
Start-Sleep -Milliseconds 500
Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
