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
