# Context size and the VRAM slab

The expert slab and the KV cache are paid from the same 16.31 GB. Every gigabyte
the context takes is a gigabyte the slab cannot hold, and the slab is what makes
the decode cheap: 9.8 GB holds the working set (steady state, 3.0 MB/token of
disk), 1 GB does not (559 ms/token). So "can this run at N context" is not a
question about the model. It is a question about which side of that trade the
run lands on — and one of the answers is a hard abort, not a slow token.

Everything below is measured on the reference rig (RTX 5060 Ti 16 GB `sm_120`,
15.38 GiB RAM, MSI M371 NVMe; the model and flags are the ones in the README's
Run section). Hardware and method caveats are in
[`docs/BENCHMARK_METHOD.md`](BENCHMARK_METHOD.md).

## Summary

| context (`-c`) | slab (`SABAH_LLAMA_HOT_BYTES`) | ubatch | result |
|---|---|---|---|
| 512 | 9.8e9 | default | works — the README's headline configuration |
| 8192 | 9.8e9 | default | works |
| 32768 | 9.8e9 | default | **abort** (VRAM exhausted), 3 of 3 runs |
| 32768 | 8.0e9 | default | works — the slab pays for the context |
| 32768 | 9.8e9 | `-ub 8` | works — the prefill pays for the context |
| 131072 | 6.0e9 | default | works - full speed (17.2 tok/s) |
| 131072 | 9.8e9 | default | works, but 34% slower (11.3 tok/s) at the VRAM ceiling |

The `-ub 8` row is `llama-server` only — the diag driver takes no ubatch argument,
and that driver is the one that aborts at `-c 32768` with the 9.8 GB slab. The
512 row is the README's headline run; the rest were run for this note, each with
`sabah run --tokens 32` and, where a ubatch applies, with `llama-server`, at the
same prompt.

The abort is not a hard context limit: the Phase 0 measurement ran `-c 131072`
with the same 9.8 GB slab through the diag driver for 255 decodes and exited 0
(88.4 ms/token, peak 16,000 MiB). It sits at the same ceiling the 32K run
aborted at, so whether a run survives is decided by the timing of the
over-budget allocation against whatever free block is left - which is also why
that configuration is 34% slower there (see "128K context" below).

`-c 32768` with a 9.8 GB slab *loads* and even survives a one-token decode
(peak 15,892 of 16,311 MiB). It dies once the run is long enough for the
over-budget path to need a device allocation that no longer fits. Both the
diagnostic driver and `llama-server` fail this way; where the boundary sits
between 8K and 32K was not located.

## The failure is VRAM exhaustion, not a routing error

`ggml-cuda.cu` hands every host-backed `MUL_MAT_ID` to the bridge, logs the
bridge's error if it refuses (`ggml/src/ggml-cuda/ggml-cuda.cu:269`) and aborts
the process (`:2058-2064`):

```
C:\...\ ggml\src\ggml-cuda\ggml-cuda.cu:2061: SABAH_LLAMA could not execute the host-backed exact expert operation
0.16.573.058 E SABAH_LLAMA MUL_MAT_ID failed: Sabah over-budget expert: out of memory
```

That second line is the real cause. The tier's documented over-budget behaviour
(`sabah/runtime/cuda/sabah_rt.cu:628`, `sb_mmid_reserve_locked`) is: take space
from the slab, and when the slab is full and nothing can be evicted, count an
`overflow` (`:643`) and allocate the expert outside the budget rather than fail
the call. That follow-up `cudaMalloc` **is** checked (`:644`) — and on a 16 GB
card at 32K context there is no room left for it, so the call returns `nullptr`
and the process aborts.

The counters at the moment of the abort, read from `SABAH_LLAMA_STATUS_FILE`,
show a run that routed normally and simply ran out of VRAM — 68.7% of lookups
served from the tier, `mapped_fallbacks` never in play (the fields that matter,
of a longer object):

```json
{"calls":506,"tokens":7416,"lookups":73428,"hits":50464,"misses":22964,
 "evictions":8735,"bytes_fetched":15464409600,"resident_bytes":9799513600,
 "overflow":485,"verify_ok":0,"verify_fail":0}
```

`overflow=485` is the tell: the tier was already allocating past its budget
before the allocation that failed. With more headroom the same counter climbs and
the run still exits 0 (982 at `-c 8192`, 1239 at `-c 32768` with an 8.0 GB slab,
both 32 tokens) — so `overflow` is not the failure, the *absence of VRAM behind
it* is.

Two notes on reading the exit code. The abort surfaces as `3221226505`
(`0xC0000409`), which is also what the **fixed** exit-time fast-fail used to
produce. They are not the same event: that one happened after the trace had been
written, this one happens mid-decode with the `GGML_ABORT` line above on stderr.
And because the runtime prints its counters from an `atexit` handler
(`sb_report_diag`, `sabah/runtime/cuda/sabah_rt.cu:2839`, registered at `:2916`
and gated on `SABAH_LLAMA_TRACE`), an abort means the `SABAH_DIAG` /
`SABAH_DISK` / `SABAH_SNAPSHOT` lines are **never printed at all** — an aborting
run reports no counters unless `SABAH_LLAMA_STATUS_FILE` is set.

## Peak VRAM, measured

`nvidia-smi`, sampled every 0.5 s; total is 16,311 MiB.

| configuration | peak | headroom |
|---|---|---|
| `-c 512`, slab 9.8e9 (README headline) | 15,360 | 951 |
| `-c 32768`, slab 9.8e9, one decode token | 15,892 | **419** |
| `-c 32768`, slab 9.8e9, default ubatch | 15,605 at load, then abort | — |
| `-c 32768`, slab 9.8e9, `-ub 8` | 15,223 | 1,088 |
| `-c 8192`, slab 9.8e9 | 14,804 | 1,507 |
| `-c 32768`, slab 8.0e9 | 13,823 | 2,488 |

### 128K context: the context is cheap, the last 300 MiB are not

Measured with `-c 131072 -n 256`, same flags. The KV cache at 131,072 cells is
**384 MiB at f16, 204 MiB at q8_0, 108 MiB at q4_0** (llama.cpp's own
`KV buffer size` line): only 12 of the 48 blocks keep a cache
(`full_attention_interval=4`), it is **K-only** (V is empty), 2 KV heads x 128
elements. The compute buffer is 964-976 MiB. The context itself costs ~1 ms of
decode.

What costs 30 ms is running the slab into the VRAM ceiling. Same prompt, same
131.6 misses/token, same kernel set:

| `-c` | slab | steady median | tok/s | VRAM plateau | free |
|---|---|---|---|---|---|
| 512 | 9.8e9 | 58.34 ms | 17.14 | 15,367 | 916 |
| 8192 | 9.8e9 | 58.39 ms | 17.13 | 15,475 | 808 |
| 32768 | 9.8e9 | 60.04 ms | 16.65 | 15,982 | 301 |
| 131072 | 6.0e9 | **58.23 ms** | **17.17** | 15,377 | 906 |
| 131072 | 7.0e9 | 63.29 ms | 15.80 | 15,991 | 292 |
| 131072 | 8.0e9 | 71.54 ms | 13.98 | 15,994 | 289 |
| 131072 | 9.8e9 | 88.43 ms | 11.31 | 16,000 | 283 |
| 131072 | 9.8e9, `-ub 64` | 79.29 ms | 12.61 | 15,988 | 295 |

An Nsight capture at 128K / 9.8e9 shows GPU busy 92.3 ms/token with
`k_mul_mat_id` at 465 us/call against ~188 us at 512 context: the same kernel on
the same bytes, 2.5x slower within 300 MiB of the ceiling. At a 6.0 GB slab the
run reads 149.5 objects/token instead of 131.6 (14% more bytes) and is still full
speed - the extra reads overlap, the pressure does not.

So 128K is affordable in KV and is paid for in slab: **6.0e9 is the value that
keeps this workload at full speed at `-c 131072`**; 9.8e9 costs 34%. The
mechanism (WDDM eviction, driver bookkeeping, or the runtime's over-budget
allocations) is not separated, only the effect is measured.

The 32K KV cache costs less than the header arithmetic suggests. From the GGUF:
48 blocks, 2 KV heads, key/value length 256, f16 — 2 KB per token per block, i.e.
96 KB per token summed over all 48 blocks. That is the upper bound (3.22 GB at
32K) *if every block kept a full cache*. The measured peak difference between
`-c 512` and `-c 32768` at the same slab is **~0.5 GB**, because this model does
not grow a cache on every block (`full_attention_interval: 4`,
`attention.compress_ratios` of `0,0,0,4` per group). So the context is not
expensive here; the *margin* is. ~0.4 GB of headroom is not enough for a run that
allocates past its budget.

## What the context changes for the user

- **Time to first token does not move.** Fill was 11.2 s at 32K against 11.2 s at
  512 for the same 53-token prompt: the fill is the expert tier, not the context.
- **The decode does not move either**, as long as the slab survives.
- **The prompt does.** Prefill runs at the fill rate, ~4.4-4.7 tok/s (measured
  through `llama-server`). A 500-token prompt is ~110 s to first token, 4,096 is
  ~15 min, and a 32K prompt is ~2 h. This is the practical limit on context for
  anything interactive, well before the VRAM problem.

## The HTTP path: `llama-server` with the bridge

The project's own server *is* this binary. `sabah serve` inspects the model and
then spawns the pinned, patched `llama-server` with the experts on the host and
`GGML_OP_OFFLOAD_MIN_BATCH=1` (`sabah/server/openai_proxy.py:252-280`). So a
client needs no special path — it needs that backend plus the placement flags.
The bridge lives in `ggml-cuda`, so the server loads it the same way the diag
does:

```powershell
$env:SABAH_LLAMA = "1"; $env:SABAH_RT_LIB = "<repo>\sabah\runtime\cuda\sabah_rt.dll"
$env:SABAH_LLAMA_HOT_BYTES = "9.8e9"; $env:SABAH_LLAMA_DIRECT_IO = "1"
$env:SABAH_LLAMA_IO_DEPTH = "8"; $env:GGML_OP_OFFLOAD_MIN_BATCH = "1"

llama-server.exe -m <model.gguf> -dev CUDA0 -ngl 99 --cpu-moe `
  -ot "\.per_layer_token_embd\.weight=CPU" -t 8 `
  -c 32768 -ub 8 --host 127.0.0.1 --port 8086 --log-file server.log
```

Measured through `POST /v1/chat/completions`, with **no diagnostic file set**:

| configuration | prompt eval | decode | request wall |
|---|---|---|---|
| `-c 32768`, `-ub 8` | 58 tok in 12.36 s = 4.69 tok/s | 31 tok in 5.82 s = **5.16 tok/s** (194 ms/token) | 18.2 s |
| `-c 8192`, default ubatch | 58 tok in 11.94 s = 4.86 tok/s | 30 tok in 5.73 s = **5.06 tok/s** (197 ms/token) | 17.7 s |

That is the diag driver's own rate at this length (`sabah run --tokens 32`,
`-c 8192`, slab 9.8e9: 212.2 ms/token median), so the HTTP path costs nothing
measurable at 32K once the ubatch is bounded. The prompt eval rate is the fill
rate in both cases (~4.1-4.9 tok/s, disk-bound).

These are **warm-up** numbers, not the steady state, and they must not be read
as the README's 15 tok/s: a request that generates 30-64 tokens ends inside the
regime where the tier is still discovering its working set, and the harness says
exactly that in its own output at the same length. The steady state needs ~130
decodes ([`docs/DECODE_BUDGET.md`](DECODE_BUDGET.md)).

### The served path always sets a diagnostic, and it is not free

`openai_proxy.py:273` sets `SABAH_LLAMA_STATUS_FILE` unconditionally for the
`sabah` backend, and the runtime rewrites that file after every expert call — 144
of them per token. `docs/BENCHMARK_METHOD.md` prices that at ~0.3 s per token.
The same two configurations above, run *with* the status file set, report
**2.04-2.53 tok/s** instead of 5.06-5.16:

| configuration | decode with the status file | decode without it |
|---|---|---|
| `-c 32768`, `-ub 8` | 31 tok in 14.68 s = 2.04 tok/s | 31 tok in 5.82 s = 5.16 tok/s |
| `-c 32768`, `-ub 8`, 8-token request | 8 tok in 2.77 s = 2.53 tok/s | — |

This is the same trap the harness was fixed for (see `CHANGELOG.md`, "the harness
no longer poisons its own timings"); the proxy still has it. Any latency taken
through `sabah serve` today is roughly half the real rate at short lengths. It
does not affect correctness, and the counters it buys are exact.

Two more things about the served path, measured while writing this:

- **`sabah serve` did not come up on this artifact at all.** It needs a `gguf-py`
  that knows the artifact's quant type (`qwen4exp` uses type 42; the PyPI `gguf`
  raises `np.uint32(42) is not a valid GGMLQuantizationType` — point
  `SABAH_GGUF_PY` at the pinned tree's `third_party/llama.cpp/gguf-py`), and with
  that fixed it then failed on the Python side before any backend started:
  `pinned warm alloc 704000 B failed: cudaHostAlloc: out of memory` from the
  warm-RAM tier planner, on a rig whose 15.38 GiB of host RAM is largely in use.
  Neither failure is about this note's subject; both mean the numbers above come
  from `llama-server` directly, which is the same backend the proxy spawns.
- **`sabah serve` has no ubatch knob.** `--context 32768` with the default slab
  on that path is the crashing configuration of this document, because the proxy
  passes no `-ub`. At 32K either lower the slab (`--hot-bytes 8000000000`) or run
  `llama-server` directly with `-ub 8`.

One integration note: this is a reasoning model, and the server's default
template puts the chain of thought in `choices[0].message.reasoning_content`.
With a small `max_tokens` the `content` field comes back empty because the answer
never left the reasoning block. A client must read both fields and budget tokens
accordingly.

## Reproduce

```powershell
# the abort: 32K context, slab unchanged
python -m sabah.tools.cli run <model.gguf> --tokens 32 --context 32768 --hot-bytes 9.8e9

# the same context, paid for by the slab
python -m sabah.tools.cli run <model.gguf> --tokens 32 --context 32768 --hot-bytes 8.0e9

# the same slab, paid for by the ubatch (server)
llama-server.exe ... -c 32768 -ub 8 --log-file server.log
```

To see the cause rather than the abort, set `SABAH_LLAMA_STATUS_FILE` and pass
the server `--log-file`: without a log file the server's logger loses the
`SABAH_LLAMA MUL_MAT_ID failed:` line (its file ends mid-line), and without the
status file the process dies before the counters are printed. The status file
costs ~0.3 s per token (`docs/BENCHMARK_METHOD.md`), so it is a diagnostic and
never a timing configuration.

## Not measured

- Where the boundary lies between 8K and 32K at a 9.8 GB slab, and whether the
  32K failure is deterministic across slab sizes and drivers. It reproduced 3 of
  3 times here, on this rig.
- The steady state through the HTTP path. Every request measured here generated
  30-64 tokens, so all of it is warm-up regime; a 256-token request through the
  bridge was not run.
- Context sizes above 32K were measured in Phase 0 (128K: 384 MiB of KV at f16,
  see above); 192K and 256K were not run. At 131,072 cells the KV is 384 MiB, so
  a 256K context would be ~768 MiB. The earlier "~4 GB" estimate assumed a value
  cache on every block; this model keeps one on 12 of 48 blocks and has no V.
- The 128K configuration was run to 255 decodes, not to a full 128K prefill, and
  not through `llama-server`.
- A graceful failure mode. Today the runtime aborts the process when the
  over-budget allocation fails; failing the call (and letting the caller decide)
  is not implemented.
