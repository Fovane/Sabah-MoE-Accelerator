param(
  [Parameter(Mandatory = $true)][int]$ProcessId,
  [int]$Seconds = 3600,
  [Parameter(Mandatory = $true)][string]$Out
)
# Sample this process's GPU memory from the Windows GPU performance counters:
#   \GPU Process Memory(<pid>_...)\Dedicated Usage   (VRAM)
#   \GPU Process Memory(<pid>_...)\Shared Usage      (system RAM used by the GPU)
# Appends CSV rows as they are taken (t_s,dedicated_bytes,shared_bytes), so the
# probe can be killed when the workload exits without losing the samples.

$ErrorActionPreference = "Stop"
$paths = @("\GPU Process Memory(*)\Dedicated Usage",
           "\GPU Process Memory(*)\Shared Usage")
if (-not (Test-Path $Out)) {
  [IO.File]::AppendAllText($Out, "t_s,dedicated_bytes,shared_bytes`r`n")
}
$sw = [Diagnostics.Stopwatch]::StartNew()
$pat = "_{0}_" -f $ProcessId

while ($sw.Elapsed.TotalSeconds -lt $Seconds) {
  $ded = 0L; $shr = 0L; $seen = $false
  foreach ($set in (Get-Counter -Counter $paths -ErrorAction SilentlyContinue)) {
    foreach ($s in $set.CounterSamples) {
      if ($s.InstanceName -like "*$pat*") {
        if ($s.Path -like "*dedicated usage*") { $ded += [int64]$s.CookedValue; $seen = $true }
        elseif ($s.Path -like "*shared usage*") { $shr += [int64]$s.CookedValue; $seen = $true }
      }
    }
  }
  [IO.File]::AppendAllText($Out, ("{0:F1},{1},{2}`r`n" -f $sw.Elapsed.TotalSeconds, $ded, $shr))
  if (-not $seen) { Start-Sleep -Milliseconds 400 } else { Start-Sleep -Milliseconds 600 }
}
