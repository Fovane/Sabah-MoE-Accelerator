# Clock hygiene

A decode measurement on this machine is only comparable with another one if the
card was at the same clocks. It is not pinned by default, and the driver's
adaptive policy does not simply "stay at boost while a model runs": on this
workstation the card steps **down** to P3 mid-decode when the host leaves gaps
between kernels, then steps back up. That is a closed loop with the workload,
so the same binary can measure 44 ms/token or 89 ms/token.

## What the 1 s trace shows

`tools/clock_trace.py` samples SM clock, memory clock, P-state, power, utilisation,
display state and the throttle bitmask once a second across idle -> decode -> idle.

Decode, **unpinned** (256 tokens, `-c 512`, 6.0e9 slab, TILE3 on):

| t (s) | SM MHz | mem MHz | P | W | state |
|---|---|---|---|---|---|
| 0-15 | 412-472 | 405 | P8 | 8.3 | idle |
| 16-21 | 2617-2857 | 13801 | P1 | 17-32 | ramp |
| 22-29 | 1470 -> 825 | **7001** | **P3** | 20-21 | **downclocked mid-decode** |
| 30-31 | 2392-2422 | 7001 | P3 | 32 | recovering |
| 32-40 | 2400-2527 | 13801 | P1 | 34-38 | partial |
| 41-51 | 2820-2842 | 13801 | P1 | 76-86 | full boost |

The 7 s P3 / 7001 MHz window is the 88.9 ms episode in miniature: kernels
stretch, the host waits, utilisation stays low, and the driver keeps the low
state a little longer. It recovers only when the expert tier warms up and the
GPU duty cycle rises.

The same decode with **`-lgc 2800,3090` only** (minimum pinned, maximum free)
holds SM at 2805-2842 MHz for the whole run (0 of 59 samples low), **but the
memory clock still drops to 7001 MHz for 7 s**. Pinning the SM floor alone is
therefore *not* enough on this card.

With **`-lgc 2800,3090 -lmc 13000,14001`**, SM stays 2805-2842 MHz and memory
stays 13801 MHz for the whole run, idle included.

## Pinning (needs an elevated shell)

```powershell
# pin: SM floor 2800 MHz, memory floor 13000 MHz, ceilings left at stock
nvidia-smi -lgc 2800,3090
nvidia-smi -lmc 13000,14001

# verify
nvidia-smi --query-gpu=clocks.sm,clocks.mem,pstate,power.draw --format=csv,noheader

# restore default clock management (do this after every batch)
nvidia-smi -rgc
nvidia-smi -rmc
```

Notes:

* `-lgc` takes `min,max`. Only the minimum is needed; the maximum is left at the
  stock 3090 MHz so the card can still boost above 2800.
* Pins are not persistent: a reboot or driver restart clears them. They are also
  not applied at idle the same way — with the pin active the card idles at
  P1/2805 MHz/~26 W instead of P8/430 MHz/~8.4 W. Unpin when finished.
* Nothing here changes the driver, the power limit (180 W) or the throttle
  reasons. `clocks_throttle_reasons.active` reads `0x400` (reliability) both
  pinned and unpinned; it is not the cause of the downclock.

## How a run is declared valid

`tools/phase0_measure.py` and `sabah bench` start a `sabah.tools.clocks.ClockSampler`
(0.5 s resolution) for every run, and write the whole-run and steady-window
statistics into the measurement JSON:

* `clocks.steady.sm_mhz` — min/median/max over the decode window
  (the tail of the run the decodes occupy);
* `clocks.steady.mem_mhz`, `pstate`, `power_w`, `throttle`, `display_active`;
* `clocks.valid` — false when the steady SM median is below **2400 MHz**
  (or the memory median is below **10000 MHz**), with `clocks.reason`;
* `machine` — power plan, HAGS key, NVIDIA power-management registry values,
  driver version and power limits, display state.

An invalid run exits **4** (driver failures still exit 3). `--allow-low-clock`
keeps the exit code at 0 while still marking the record invalid — use it only
when the low clock is the thing being measured.

Machine state at the time of these traces: Windows power plan **Balanced**
(`381b4222-f694-41f0-9685-ff5bb260df2e`), HAGS key **absent (off)**, display
**attached and active**, driver **617.42**, power limit **180 W** (default 180,
max 200), no `PowerMizer*` overrides in the driver registry.

## Consequences for every earlier table

Any table produced before this module existed was taken without a clock record.
The Phase 0 sweep (88.43 ms) matches the low-clock episode and must be treated
as invalid until re-run pinned; that re-run is item 3 of the current round.
