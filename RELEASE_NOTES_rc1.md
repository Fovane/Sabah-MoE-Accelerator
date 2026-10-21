# Release notes: v1.1.0

**What it is.** A streaming MoE inference runtime for machines that have the VRAM
but not the RAM. An 83.6 GB model (`Qwen3.8-Flash-Next` IQ3_S, 48 blocks x 512
experts, top-10) runs on a 16 GB-VRAM / 15 GB-RAM box by streaming expert weights
disk -> pinned staging -> VRAM, with no RAM tier, no page cache and no page file.

**Hardware this is measured on.** RTX 5060 Ti 16 GB, Ryzen 5 7600, 15.4 GiB usable
RAM, MSI SPATIUM M371 PCIe gen3 x4 (driver 617.42 / CUDA 12.8). Every number below
is this rig; a different drive or GPU changes all of them.

**The regime decides the number.** Never quote one figure for this project.

| regime | tok/s | notes |
|---|---|---|
| steady decode, short context, SEG=48 | **33.2** | 30.12 ms/token, clocks valid (SM 2745 MHz), byte-identical to the SEG=0 path |
| steady decode, 128K window, short prompt | **32.9** | 30.36 ms/token, ~3.7K-deep KV (REPORT 16.6, N=1) |
| steady decode, filled ~30K KV | **3.6** | 280.45 ms/token, 339 MB/token, 501 misses/token, still not steady (REPORT 21-22) |
| prefill, 32K-token prompt | - | 1174 s, 1013 GB (REPORT 21) |

**Which branch.** `release-candidate` = `bd9fbc2`; this tag points at the same tree.
`phase-g-longctx` (`c20564e`) carries the long-context measurements on top.
`phase-f-prefill-group` is an experiment and stays off by default.

## The correctness contract

This release's claim is not a speed number, it is that the runtime is *checkable*:
`sabah gate` re-checks on every run that the bytes it serves are the GGUF's bytes.

```
checks       verify 2000/0, mapped_fallbacks=0, overflow=1006
  [ok] the bridge executed expert ops       4608
  [ok] every served byte matched the GGUF   2000 ok / 0 fail
  [ok] the id reuse is byte for byte        3072 ok / 0 fail
  [ok] no mapped fallback                   0
  [ok] decode finished                      0
gate: PASS
```

and the decode number, from `sabah run` / `phase0_measure.py` with the clocks the
harness accepted as valid:

```
release-evidence: exit=0 wall=30.4s
  fill 10.86s  warm 40.588499999999996ms  steady 30.120199999999997ms = 33.20031075490867 tok/s
  VRAM peak 15678 MiB  WS peak 5224 MiB  avail min 6534 MiB
  clocks: VALID steady SM 832/2745/2842 MHz, mem 13801 MHz, P1, 44.5 W
```

## What is in it

| feature | default | proof |
|---|---|---|
| exact expert execution through the patched bridge (`SABAH_LLAMA`) | on | `sabah gate`: 2000/2000 bytes checked |
| direct unbuffered I/O with pinned staging, no RAM tier | on | host process working set flat; `mapped_fallbacks=0` |
| TILE3 quant kernel (bit-exact, 60.56 -> 44.21 ms/token) | on | byte-identical outputs vs the previous kernel |
| segmented CUDA graphs (`SABAH_CUDA_GRAPH_SEG=48`) | on | byte-identical; 42.74 -> 30.36 ms at 128K |
| prefetch, merged layer fetch | **off** | correct but measured net-negative on this rig |
| layer-group prefill (`SABAH_PREFILL_GROUP`) | **off** | the fused-kernel set changes at the split boundary; not byte-identical |

## Known limitations, stated rather than hidden

- **Deep context is disk-bound and no policy reaches a tier here.** ~30K KV decodes
  at 280 ms/token; replaying the routing trace offline puts the Belady optimum at
  68% hit where 10 tok/s needs 88% (REPORT 25). The gap is in the trace, not in LRU.
- **At `-c 131072` the "flat RAM" claim has a boundary.** The process working set
  stays flat, but the driver moves slabs into system RAM when the 6 GB slab no longer
  fits beside the KV cache: Shared Usage +3517 MiB (REPORT 23.1). Size the slab for
  the context.
- **A low P-state can double the measured ms/token** (88.9 vs 58 ms for the same
  binary). Pin `-lgc 2800,3090 -lmc 13000,14001`; the harness marks a run invalid
  below 2400 MHz steady.
- **No perplexity or quality benchmark is run.** Byte-exact means the runtime serves
  the GGUF's weights, not that the model is good; the IQ3_S quality figures are the
  quantiser's (ISTA-DASLab).
- A second 128K prompt aborts with "over-budget expert: out of memory" on both the
  segmented and eager paths.

## Reproducing it

```powershell
install.bat            # package + runtime DLL, architecture auto-detected
install.bat llama      # plus the pinned llama.cpp, the bridge and the diag patch

$env:SABAH_MODEL_DIR = "C:\Strata-data\models\IQ3_S"
.venv\Scripts\sabah.exe gate   <model.gguf>
.venv\Scripts\sabah.exe run    <model.gguf> --tokens 256 --context 131072
python -m sabah.tools.bench_protocol      # refuses / flags an invalid machine state
```

`SABAH_MODEL_DIR` is the single source of truth for the shard location; no script in
the repository carries a drive letter.
