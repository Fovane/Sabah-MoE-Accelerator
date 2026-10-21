# Measured results, and what they do not prove

Every number below has a command, a commit, an N and a min/median/max column in
[`REPORT.md`](REPORT.md) or in the run directory under `results/`. A number is
labelled by regime, never quoted as one number for the model.

## What the headline numbers are

| regime | number | source |
|---|---|---|
| 128K chat, 2048 decode steps, slab 6.0e9, tail window (last 256 steps) | 42.74 -> **30.36 ms/token** (23.40 -> **32.93 tok/s**), N=1 | `item6b-chat-2048`, `accept-chat2048-seg48` in REPORT 16.6 |
| 128K chat baseline (SEG=0) on the same prompt | 42.74 ms/token, 23.40 tok/s, N=1 | `item6-chat` |
| `-c 2048`, 640 decodes across n_kv 256/512 | 43.66 -> **29.91 ms/token** (SEG=48) | `ref-kv640`, `seg48-c512-1` |
| steady decode at 53-token KV depth | ~67 ms/token = ~15 tok/s, disk-free (3.0 MB/token) | `a-baseline-now` |
| host RAM | flat at 4.78 GiB across a 2048-step run, `mapped_fallbacks=0` | `item6b-chat-2048` |

Clocks for every run above: driver 32.0.16.1742, SM 2805-2857 MHz, memory
13801 MHz, P1, 35-41 W, i.e. the pin `-lgc 2800,3090 -lmc 13000,14001` was in
force. **That pin is no longer available on this machine** (`nvidia-smi -lgc`
now needs admin), so any timed number recorded from October 21 onward must be
read with the harness's own steady-clock gate: the machine can double ms/token
(88.9 ms vs 58 ms for the same decode, REPORT 9).

## What is claimed, and what is not

* **Claimed**: byte-identical outputs for what the runtime serves. The gate
  compares every served expert block against the GGUF range
  (`verify_ok=2000 verify_fail=0`), and the id-reuse path is checked 768 times
  byte-for-byte per run.
* **Not claimed**: model quality. Byte-identical means the runtime serves the
  same bytes the reference would; it does not measure perplexity, an accuracy
  benchmark or a benchmark-vs-baseline comparison, and none is in this document.
* **Not claimed**: 128K decode at 30-35 tok/s. The measured 30.36 ms (32.93
  tok/s) is a 2048-step chat tail at slab 6.0e9 with a ~3.7K-deep KV at the
  point the tail window is read. It is not a filled-128K-context number.
* **Not claimed**: the layer-group prefill speed. The feature is off by default
  and its divergence is not fixed (REPORT 18).

## The limitations to state plainly

* **Prefill is the wall.** A 2048-token chunk reads 113-134 GB, the whole expert
  pool per chunk; the 128K prompt in item 6 needed ~130 min. REPORT 15.2.
* **Warm-up is the second wall.** The tier is 9.8 GB and the first prompt's
  working set is 12.68 GB, so the first ~128 decodes pay 70.6 MB/token.
* **The coding trace does not warm up** in 2048 steps at slab 6.0e9 (REPORT
  15.4: still ~45 ms/token at the last window), which is why the README's
  headline decode number is the chat trace only.
* **The second 128K prompt aborts.** An over-budget expert
  (`ggml-cuda.cu:2068-2072`) ends a `-c 131072` run at ~285 s on both the
  segmented and the eager path. One 128K correctness gate exists, not two.
* **710 Ti class hardware, M371 PCIe 3.0 SSD, Windows/WDDM.** All numbers are
  this machine. Comparison with Glyd/Strata/GyroStrata is not measured by us and
  is not attempted.
