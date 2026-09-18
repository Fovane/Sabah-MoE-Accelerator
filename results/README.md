# Measured results

The raw output behind every number the top-level README labels **MEASURED**.
All of it comes from one machine: RTX 4050 Laptop (6 GB, sm_89, PCIe gen4 x8),
29.8 GB RAM, Windows 11, CUDA 13.3.

| file | produced by | what it shows |
|---|---|---|
| `bench_moe_blk0.json` | `sabah bench --block 0` | first capacity sweep, before the staging-event fix |
| `bench_moe_blk0_staged.json` | `sabah bench --block 0` | same sweep with per-buffer staging events, host staging timed |
| `bench_moe_blk0_ram.json` | `sabah bench --block 0 --bank-mode ram` | pinned RAM bank, staging copy removed |
| `calib_check.json` | `sabah calibrate` | static placement vs LRU, 4-48 GB |
| `calib_check_hi.json` | `sabah calibrate --caps 2,6,20,28,36,56,64,72` | the rest of the curve |

Reproduce with the commands in the table. The numbers will differ on other
hardware; that is the point of `sabah qualify`.

`bench_moe_blk0.json` is kept deliberately even though a later fix superseded
it: the 9.127 -> 4.646 ms/token improvement at 16 slots is the measurement of
that fix.
