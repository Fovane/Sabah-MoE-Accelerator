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
