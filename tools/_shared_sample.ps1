param(
  [int]$Seconds = 600,
  [string]$Out = "shared_samples.csv"
)
$ErrorActionPreference = "SilentlyContinue"
"t_s,cooked_bytes" | Set-Content -Path $Out -Encoding ascii
$sw = [Diagnostics.Stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
  try {
    $c = Get-Counter '\GPU Adapter Memory(*)\Shared Usage'
    foreach ($s in $c.CounterSamples) {
      if ($s.InstanceName -like 'luid_0x00000000_0x0000d43d_*') {
        "$([math]::Round($sw.Elapsed.TotalSeconds,2)),$([int64]$s.CookedValue)" |
          Add-Content -Path $Out -Encoding ascii
      }
    }
  } catch {}
  Start-Sleep -Milliseconds 500
}
