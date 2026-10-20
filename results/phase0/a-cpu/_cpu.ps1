$log = $args[0]
$sw = [diagnostics.stopwatch]::StartNew()
while ($sw.Elapsed.TotalSeconds -lt 90) {
  try { $p = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue } catch { $p = -1 }
  "{0:F2},{1:F1}" -f $sw.Elapsed.TotalSeconds, $p | Add-Content $log
  Start-Sleep -Milliseconds 300
}
