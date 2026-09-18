# RC4 — numerical drift: root cause and correctness closure

## The question

> Is Sabah's multi-token drift a bug, or is it normal backend numerical
> variation?

**Both, in sequence, and the order matters.** The drift RC3 observed was
caused by a structural Sabah bug. After that bug was fixed, what remains is a
numerical difference between Sabah and llama.cpp that is fully accounted for by
llama.cpp quantizing activations to 8 bits inside its own expert kernels.
Sabah does not; its expert operation matches exact float64 arithmetic to
fp32 precision.

**Conclusion (pre-fix): SABAH STRUCTURAL BUG FOUND.**
**Conclusion (post-fix): BACKEND NUMERICAL DIFFERENCE CHARACTERIZED.**

Both are supported below by tensor-level measurements. Raw data:
`results/rc4_numerical/`.

---

## 1. Reproduction of RC3

| item | value |
|---|---|
| Sabah commit at start | `b988fd2` |
| llama.cpp base | `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b` + `patches/llama.cpp/0001` |
| tests at start | 8 passed |
| first token, reference | `1596` ("We") |
| first token, Sabah | `1596` ("We") |
| Sabah cache counters | `hits=42477 misses=16743 evictions=15771` — identical to the RC2 record |

The RC3 path reproduced exactly, including cache counters, so it was
deterministic. It was then not changed until the diagnosis below was complete.

## 2. Two confounds in the RC3 comparison

**RC3's "Sabah" run executed Sabah for only one prompt.** With `--cpu-moe`
the llama.cpp scheduler sends an expert `MUL_MAT_ID` to CUDA only when the
batch is at least `GGML_OP_OFFLOAD_MIN_BATCH` tokens (default 32). Below that
the op runs on the CPU and never reaches the Sabah bridge. RC3's own counters
prove it: 33,780 + 13,740 = 47,520 lookups = 144 ops × 10 experts × **33
tokens**, which is exactly the Turkish prefill. The two 21-token prompts and
every decode step in the RC3 "Sabah" run executed on the CPU.

**RC3's "CPU reference" was not purely CPU either.** The same rule sent its
33-token Turkish prefill through llama.cpp's native CUDA kernel.

RC4 therefore pins placement explicitly for every run:

| path | variables | what executes each expert `MUL_MAT_ID` |
|---|---|---|
| A — CPU | `GGML_OP_OFFLOAD_MIN_BATCH=1e9` | llama.cpp CPU kernel, every token |
| B — native CUDA | `GGML_OP_OFFLOAD_MIN_BATCH=1` | llama.cpp CUDA kernel; the scheduler copies only the routed experts |
| C — Sabah | as B, plus `SABAH_LLAMA=1` | `sabah_rt_mul_mat_id_v2` |

B and C have identical graphs and op placement; they differ only in which
implementation executes `MUL_MAT_ID`. Placement is recorded rather than
assumed: every captured tensor carries its backend buffer (`CUDA_Host` for A,
`CUDA0` for B and C), and Sabah's call counter must equal 144 ops × steps.

Everything else is identical: model, prompt (59 tokens via the model's chat
template), context 512, batch/ubatch 512, 8 threads, `-ngl 99`, argmax
decoding over the full logit vector, no warm-up. See `environment.json`.

## 3. Instrumentation

`patches/llama.cpp/0002-sabah-diag-capture-tool.patch` adds
`llama-sabah-diag`, a greedy driver that observes the graph through the
scheduler's eval callback without changing it. Per step it writes the full
logit vector, the argmax and the fed token. On request it also writes selected
graph tensors, and for every expert projection the exact `MUL_MAT_ID`
input (`src[1]`) and ids (`src[2]`): the tensors at the Sabah boundary. It
supports teacher forcing, so two paths can be compared on the same context at
every step.

## 4. First divergence — pre-fix

Prefill, native CUDA (B) vs Sabah (C) as shipped in RC3, block 0:

| tensor | rel L2 |
|---|---|
| `MUL_MAT_ID` input (gate/up) | **0** — bit-identical |
| gate | 1.2e-2 |
| up | 1.4e-2 |
| SwiGLU activation (`down` input) | 1.1e-2 |
| **down** | **1.60** |
| MoE aggregate | 0.86 |
| block output | 0.31 |

From block 1 onward, 0 of 59 tokens had matching router ids, and the first
greedy token was `12` instead of `1596`. The single-token decode step then
crashed: `operation not permitted when stream is capturing`.

The error shape is structural, not numerical. For the down projection, expert
slot 0 matched; slots 1–9 had **cosine ≈ 0** with the reference (orthogonal),
and every slot carried the norm of slot 0's activation (≈16.5 against reference
norms of 8–13).

### Root cause (proved from RC3's own captures, before any code changed)

In ggml's `MUL_MAT_ID` the input row for selected slot *i* of token *t* is
`src1 + (i mod ne11)·nb11 + t·nb12`. For gate and up, `ne11 = 1`: one hidden
state is broadcast to every expert. For the **down projection**,
`ne11 = n_used`: each selected expert has its own SwiGLU activation. The
bridge passed only `nb12` and the kernel read `x + t·nb12`, so every expert's
down projection consumed **slot 0's** activation.

Replaying the RC3 block-0 capture against gguf-py-dequantized weights:

| Sabah down output, slots 1–9, compared with | median rel L2 |
|---|---|
| `W_down[e_i] · h_i` (correct semantics) | **1.72** |
| `W_down[e_i] · h_0` (the bug hypothesis) | **2.55e-7** |

The right expert weights were applied to the wrong input. The first token
matched in RC2 only because slot 0 is always correct and the resulting error
did not flip that particular argmax.

The isolated tests could not catch this. They drive `sabah_moe_block`,
which builds its own per-expert activation, and never called the
llama.cpp-facing ABI with more than one input row.

### Three further defects found in the same audit

1. **CUDA-graph capture.** ggml-cuda captures single-token decode into a CUDA
   graph. The bridge downloads ids and may allocate, which is illegal during
   capture. RC3 never hit this, because its decode never reached Sabah.
2. **In-call eviction.** A slot fetched earlier in the same `MUL_MAT_ID` call
   could be evicted later in that call, before the kernel ran, when one call's
   working set exceeded the hot-tier budget, leaving a dangling pointer. It was
   latent at the default 2 GiB and reachable at small capacities.
3. **Operator fusion.** Uninstrumented runs crashed in prefill with an illegal
   memory access, while instrumented runs passed. Isolated by experiment: with
   `GGML_CUDA_DISABLE_FUSION=1` the crash disappears and the token is correct;
   with `CUDA_LAUNCH_BLOCKING=1` the faulting kernel is a ggml kernel, not
   Sabah's. A ggml-cuda fusion consumed the Sabah-routed `MUL_MAT_ID` and read the
   host expert tensor as device memory, bypassing the bridge. Instrumented runs
   hid it, because observing a tensor splits the graph and prevents fusion
   across it.

## 5. Fixes

| defect | fix |
|---|---|
| down projection read slot 0's input | ABI **v2** `sabah_rt_mul_mat_id_v2` takes `x_id_stride` (`nb11`) and `x_rows` (`ne11`); the kernel reads row `id_i mod x_rows`; `x_rows` must be 1 or `n_used`. The v1 symbol is removed, so a stale DLL paired with a new patch (or the reverse) fails to load instead of silently computing v1 semantics. |
| CUDA-graph capture | the patch disables CUDA graphs when a host-backed `MUL_MAT_ID` runs through Sabah, following llama.cpp's own `TAG_MUL_MAT_ID_CUDA_GRAPHS` precedent for synchronizing `MUL_MAT_ID`s |
| in-call eviction | each slot records the call that last referenced it; such slots are never victims. If nothing else can be evicted the tier temporarily exceeds its budget (counted as `overflow`) rather than dangle a pointer |
| fusion bypass | the patch refuses any fusion window containing a Sabah-routed `MUL_MAT_ID`, so the op always reaches the bridge |
| (verification) | `SABAH_LLAMA_VERIFY_BYTES=1` byte-compares every hit and every fetch against the original GGUF range; `SABAH_DIAG` reports call, token, lookup, overflow and verification counts |

A regression test for the contract that failed,
`tests/test_mul_mat_id_contract.py`, drives the native ABI with a distinct input
row per slot on real block-0 weights. It asserts exactness, asserts that the v1
row-0 behaviour would be caught, runs the wrong-expert sentinel, and checks a
hot tier smaller than one call's working set.

## 6. Structural exactness — post-fix

| check | result |
|---|---|
| Sabah coverage | 288 calls = 144 ops × 2 steps; 8,466 token-rows = 144 × 60 − 3 × 58 (block 47 computes output rows only); 84,660 lookups |
| ids received at `MUL_MAT_ID` vs router `ffn_moe_topk` | **84,660 checked, 0 mismatches** |
| byte integrity, every hit and every miss | **84,660 verified, 0 failures** |
| expert substitution / pruning | none; no code path exists in the shipped library |
| over-budget allocations | 0 at 1 GiB |

## 7. Operation-level exactness — the decisive measurement

Every captured expert projection was replayed in float64 on the path's **own**
captured input and ids against gguf-py's dequantized weights. This measures
what each path's `MUL_MAT_ID` computed, independently of any drift upstream of
it.

Decode step, all 48 blocks × gate/up/down (144 ops per path):

| path | op | rel L2 vs exact, median | worst |
|---|---|---|---|
| A — CPU | gate / up / down | 6.2e-3 / 6.9e-3 / 8.4e-3 | 1.0e-2 |
| B — native CUDA | gate / up / down | 4.6e-3 / 5.2e-3 / 2.1e-2 | 3.0e-2 |
| **C — Sabah** | gate / up / down | **2.4e-7 / 2.7e-7 / 2.6e-7** | **7.8e-7** |

Prefill blocks 0 and 2 (Q4_K, Q5_K, Q5_1, Q8_0), 12 tokens × 10 slots:
Sabah 1.2e-7 – 3.6e-7; CPU 7.8e-3 – 1.0e-2; native CUDA 8.5e-3 – 1.9e-2.

**Why llama.cpp is 10⁴ further from exact math than Sabah.** ggml's
quantized matmuls do not multiply by the float activation: they first quantize
it to 8 bits. The CPU uses Q8_K per 256 values for K-quant weights; CUDA uses
Q8_1 per 32 values. Emulating that step explains the deviation:

| path | compared with its own emulated activation quantization |
|---|---|
| CPU gate/up (Q8_K) | **2.1e-7 – 2.5e-7** — fully explained |
| CPU down (Q8_1) | 6.1e-4 — mostly explained |
| native CUDA Q8_0 down, prefill | **8.3e-8** — fully explained |
| native CUDA gate/up, decode (Q8_1, MMVQ) | 1.7e-4 – 2.0e-4 — mostly explained (the Q8_1 scale is stored in fp16) |
| native CUDA down, decode | 2.0e-2 — **not explained** by this emulation |

The last row is an honest residual: the mechanism behind native CUDA's larger
down-projection deviation is not identified here. It is llama.cpp-internal
and does not involve Sabah, but it is part of the reference variation Sabah is
compared against.

## 8. Path-level comparison (prefill, 59 tokens, all 48 blocks)

| pair | router slot mismatches | final logits rel L2 | top-1 |
|---|---|---|---|
| A CPU vs B native CUDA (**control**) | 5,412 / 27,740 (19.5%) | 0.30 | same |
| B native CUDA vs C Sabah | 5,398 / 27,740 (19.5%) | 0.28 | same |
| A CPU vs C Sabah | **4,987** / 27,740 (18.0%) | **0.17** | same |
| B vs C, **pre-fix** | 23,171 / 27,740 (83.5%) | — | differs |

The control row is the central fact of this report: **llama.cpp's own two
backends disagree on one routed expert slot in five** for this prompt. A
correctness gate of "router ids exactly equal to the CPU reference" was never
achievable by llama.cpp itself. Post-fix, Sabah is closest to the CPU
reference of any pair.

### All-block curve (B vs C, block output `l_last` rel L2, post-fix)

block 0: 5.1e-3 · 8: 1.1e-2 · 16: 1.4e-2 · 24: 2.8e-2 · 32: 4.5e-2 ·
40: 9.7e-2 · 46: 1.2e-1. Gradual accumulation through depth, with no jump.
The control pair A vs B follows the same curve (block 0: 4.9e-3, 16: 1.4e-2).
Pre-fix, the same quantity was 0.31 at block 0: a single-operation jump.

### Expert-by-expert, block 0 down projection (id-agreeing tokens)

| pair | per-(token, expert) rel L2 p50 / p95 / max | cosine per slot |
|---|---|---|
| B vs C | 2.1e-2 / 4.1e-2 / 6.8e-2 | ≥ 0.99964 |
| A vs B (control) | 2.3e-2 / 4.2e-2 / 6.9e-2 | ≥ 0.99960 |
| B vs C pre-fix | 1.56 / 4.23 / 10.4 | slot 0: 0.9998, slots 1–9: ≈ 0 |

B vs C and the control share the same three worst (token, slot, expert)
pairs, (44, 0, 261), (57, 3, 64) and (38, 2, 349), with the same magnitudes.
The common term is native CUDA's own quantization error. The error is dense
and small in every slot, not sparse and catastrophic.

## 9. Cache-capacity invariance and byte integrity

Sabah, 16 greedy tokens, byte verification on, three hot-tier budgets:

| budget | over-budget allocations | bytes verified | logits vs 1 GiB |
|---|---|---|---|
| 128 MiB | 4,030 (tier smaller than one call's working set) | 104,820 / 104,820 | **bit-identical, 16/16 steps** |
| 512 MiB | 0 | 104,820 / 104,820 | **bit-identical, 16/16 steps** |
| 1 GiB | 0 | 104,820 / 104,820 | reference |

Byte verification on and off also gives bit-identical logits. Whether an expert
was already resident, freshly fetched, evicted and refetched, or held past
budget because the call still needed it, the result does not change by one bit.

## 10. Sequential decoding

Uninstrumented graph, argmax over full logits, 59-token prompt.

### Determinism (run-to-run, same path)

| path | 64 tokens | logits, all 64 steps |
|---|---|---|
| native CUDA | identical | **bit-identical** |
| Sabah | identical | **bit-identical** |

### Greedy-token equivalence (free-running)

| pair | 16 tokens | 64 tokens |
|---|---|---|
| native CUDA vs Sabah | **identical** | first divergence at **token 38** |
| CPU vs native CUDA (control) | identical | identical |
| CPU vs Sabah | identical | first divergence at token 38 |

### Error growth by token (teacher-forced on native CUDA's tokens, so every step has an identical context)

| path vs native CUDA | top-1 agreement | logit rel L2 median | p95 | max | first half → second half (median) |
|---|---|---|---|---|---|
| CPU (control) | 64/64 | 0.151 | 0.281 | 0.329 | 0.135 → 0.170 |
| Sabah | 62/64 | 0.148 | 0.268 | 0.356 | 0.129 → 0.164 |

The error stays bounded and grows slowly, and it grows identically for the
control and for Sabah. Sabah's two top-1 disagreements are steps 38 and 52,
where native CUDA's own top-1/top-2 margin is 1.315 and 0.163 logits.

### Is the divergence at token 38 a Sabah defect?

At step 38, on the same context:

| logit of | native CUDA | CPU | Sabah |
|---|---|---|---|
| token 846 (native CUDA's top-1) | 20.219 | 17.978 | 16.987 |
| token 248045 (native CUDA's top-2) | 18.904 | 17.829 | **18.196** |
| margin (846 − 248045) | +1.315 | **+0.149** | −1.209 |

The CPU reference keeps native CUDA's token by 0.149 logits; Sabah crosses by
1.2. To decide whether that is abnormal, the shift that each path applies to
native CUDA's top-1/top-2 gap was measured at all 64 aligned steps:

| path | mean shift | std | p50 \|shift\| | p95 \|shift\| | max \|shift\| |
|---|---|---|---|---|---|
| CPU (control) | −0.077 | 0.727 | 0.38 | 1.77 | 2.37 |
| Sabah | −0.150 | 0.656 | 0.33 | 1.40 | 2.52 |

The distributions match; Sabah's is slightly tighter. From each path's own
fitted distribution and native CUDA's 64 margins, the expected number of top-1
flips in 64 steps is **1.42 for CPU and 1.46 for Sabah**. Observed: CPU 0
(Poisson probability 0.24), Sabah 2 (probability 0.25). Nine of the 64 steps
have a margin below the control's own p95 shift. On this prompt, llama.cpp's
CPU backend would be expected to leave native CUDA's greedy path about once
per 64 tokens; this time it did not.

**The divergence at token 38 is the expected behaviour of two correct
backends at a contested step. It is not evidence of a Sabah defect.** It is
still reported as a greedy-equivalence failure, because that is what it is.

## 11. Correctness criterion — derived from the measurements

The criterion has three layers, each with its own pass condition. They are
reported separately and never collapsed into one PASS.

**Layer 1 — structural exactness (no tolerance).** Every expert
`MUL_MAT_ID` is executed by Sabah (call and lookup counters equal the graph's
op count × tokens × top-k); the ids Sabah receives equal the router's
`ffn_moe_topk` exactly; every expert slice used equals its GGUF byte range;
and no substitution path exists in the shipped library.

**Layer 2 — operation-level numerical equivalence (primary).** On its own
captured input, Sabah's `MUL_MAT_ID` must equal float64 math on the
dequantized GGUF weights to **relative L2 ≤ 2e-6** per op.

- *Why this bound:* fp32 accumulation over at most 2,560 terms has a
  worst-case relative error of about √K·2⁻²⁴ ≈ 3e-6.
- *Measured:* median 2.4e-7 – 2.7e-7, worst **7.8e-7** over 144 decode ops (all
  48 blocks), plus prefill ops on all four quant types.
- *Discrimination:* a wrong expert measures 1.42, which is **7 × 10⁵ × the
  tolerance**. The v1 row-0 defect measures 1.33–2.55. llama.cpp's own backends
  measure 4.6e-3 – 3e-2 on this metric, because they quantize activations.
- *Why this is the primary layer:* it is deterministic and local, and upstream
  drift cannot move it. Unlike any end-to-end metric, it cannot mistake
  legitimate backend variation for a bug, or a bug for variation.

**Layer 3 — path-level consistency (secondary, distributional).** Against
native CUDA, Sabah's deviation must lie within **1.25 × the CPU-vs-native-CUDA
control** on the same contexts, for the median and p95 of logit rel L2, the
router slot-mismatch rate, and the std of the top-2 gap shift.

- *Why comparative:* llama.cpp's own backends disagree on 19.5% of routed
  slots and differ by 0.15 median logit rel L2. No absolute tolerance on these
  quantities is meaningful.
- *Measured ratios:* 0.98 (logit median), 0.95 (logit p95), 1.00 (router,
  5,398 vs 5,412), 0.90 (gap-shift std).
- *Sentinel:* 4.6 × control at the median, and above the control's maximum at
  all 16 steps. Pre-fix Sabah: router mismatch 4.3 × control.
- *Limits:* one prompt, 59 prefill tokens, 64 decode steps. The 1.25 factor is
  a judgement, not a statistical bound, and needs more prompts to become one.
  It separates the sentinel with a thinner margin than layer 2 does (sentinel
  minimum 0.393 vs control maximum 0.329), which is why it is secondary.

**Greedy-token equivalence** is measured and reported as its own metric. It is
not a numerical tolerance, and a token match is not used as evidence of
numerical equivalence (§12 shows why the reverse would be wrong too).

## 12. Wrong-expert sentinel, after the criterion was fixed

| level | right expert | wrong expert (slot 0 of every call → neighbour) | ratio |
|---|---|---|---|
| operation (real block-0 weights, native ABI) | rel L2 2.8e-7 | 1.42 | **5.2 × 10⁶** |
| end to end, logits vs native CUDA (16 teacher-forced steps) | median 0.13 | median 0.70, min 0.39 | 5.3× median; 16/16 steps above the control's maximum |
| end to end, greedy | 16/16 top-1 agree | 13/16 top-1 agree | — |

The sentinel is a separate test-only build (`-DSABAH_SENTINEL_WRONG_EXPERT`),
never a runtime switch of the shipped library. It fails layer 2 by six
orders of magnitude and fails layer 3.

The end-to-end rows carry a lesson: a wrong expert in one of ten slots of
every call still produced the right greedy token at 13 of 16 steps. **Token
agreement is weak evidence of correctness**, which is why it is not the gate.
## 13. Sabah API backend

`sabah serve --backend reference --allow-reference` vs `sabah serve --backend
sabah`: the same patched `llama-server` and placement, 3 prompts, temperature 0,
seed 1, 32 tokens each, per-token logprobs.

| prompt | result |
|---|---|
| "Reply with exactly: SABAH_OK" | **32/32 tokens identical**; max \|Δ logprob\| 0.074 |
| "What is 17 times 23? Answer with the number only." | **32/32 tokens identical**; max \|Δ logprob\| 0.070 |
| "Türkiye'nin başkenti neresidir? Tek kelimeyle cevap ver." | diverges at token 0. The reference's own top-2 are "The" −1.042 / "We" −1.126 (margin **0.084**); Sabah chooses "We". A gap shift of about 1.06, inside the control's p95 of 1.77 (§10) |

`/health` for the Sabah server reported `execution: SABAH_NATIVE_MUL_MAT_ID`,
and the runtime's own counters grew with every request: calls 5,178 → 9,925 →
14,673, lookups exactly 10 × tokens at every snapshot. The status file is
written at most every 250 ms, so a snapshot can lag the last few calls. Exact
coverage accounting comes from the diag runs (§6, §10: 9,216 calls = 144 ops
× 64 steps).

**API backend: functional PASS.** Sabah serves the API and is proven to
execute the expert ops. **Deterministic A/B:** 2/3 prompts identical; 1/3
diverges at a near-tie, consistent with §10.

## 14. Benchmark (measured, after correctness)

Sustained greedy generation, 64 tokens, same prompt and settings, uninstrumented,
nothing else running. Decode throughput excludes prefill.

| path | decode tok/s | median ms/token | p95 ms/token | prefill (59 tok) |
|---|---|---|---|---|
| **stock llama.cpp** (defaults: prompt ≥ 32 tokens on GPU, decode experts on CPU) | **1.583 / 1.644** (2 runs) | 594 / 565 | 1181 / 1109 | 93 / 91 s |
| llama.cpp, CPU experts forced | 1.448 | 594 | 1231 | 51 s |
| llama.cpp, native CUDA experts forced (Sabah's placement) | 0.721 | 1225 | 2444 | 95 s |
| **Sabah**, 1 GiB hot tier | **0.468 / 0.493** (2 runs) | 2097 / 1879 | 3447 / 3285 | 104 / 109 s |

**MEASURED SPEEDUP = Sabah / stock llama.cpp = 0.493 / 1.644 = 0.30.**
On this machine Sabah is **3.3× slower** than stock llama.cpp. Against
llama.cpp's own CUDA expert path with the same placement it is 0.68×.

Why, measured rather than guessed: the 6 GB GPU holds the 4.8 GB fixed path, so
only 1 GiB is left for the expert tier (1.3% of the 77 GB bank). Over the
64-token run the tier hit rate was **35%** (61,287 of 173,940 lookups), and
**117.6 GB** crossed PCIe. Every miss is a synchronous `cudaMalloc` plus a
copy from pageable mmap memory, and every eviction synchronizes the device.
Stock llama.cpp avoids the transfers altogether on this machine by running
decode experts on the CPU next to the page cache. This is the machine class
the Phase A planner already rated `STORAGE_BACKED`, "not recommended"; the
benchmark confirms that verdict. None of the RC4 changes were performance
work.

## 15. Release decision

| gate (as specified) | result |
|---|---|
| Structural exactness | **PASS** — coverage, ids, bytes, no substitution (§6) |
| Numerical equivalence with documented criterion | **PASS** — layer 2 worst 7.8e-7 ≤ 2e-6; layer 3 ratios 0.90–1.00 ≤ 1.25 (§11) |
| Wrong-expert sentinel | **PASS** — 5.2 × 10⁶ at op level; fails layer 3 (§12) |
| 16-token greedy equivalence | **PASS** — native CUDA vs Sabah identical |
| 64-token greedy equivalence | **FAIL** — first divergence at token 38 (§10) |
| Sabah API backend | see §13 |
| Reproducible integration | **PASS** — patches apply to pristine `96ffdc41c` and reproduce the tested tree byte for byte |
| Measured benchmark | **AVAILABLE** — 0.30× vs stock llama.cpp |
| Claim hygiene | **PASS** — no speedup claimed; every number labelled |

**v1.0.0 is not released.** The 64-token greedy gate fails as written. The
evidence in §10 says this gate cannot be met by any pair of correct backends
that round differently: llama.cpp CPU vs native CUDA would be expected to
diverge about 1.4 times per 64 tokens on this prompt as well. Replacing a
failed gate with a better one is a decision for the project owner, not for
the release process, so it is recorded here rather than applied. Released as
**v0.9.0-rc4**.

Options for closing the greedy gate, for that decision:

1. Replace strict greedy equivalence with the layered criterion of §11, and
   report greedy agreement as a statistic against the CPU-vs-native-CUDA control
   (Sabah 1.46 expected flips per 64 tokens vs control 1.42).
2. Add an opt-in reference-arithmetic mode in which Sabah reproduces ggml
   CUDA's Q8_1 activation quantization. That would target near-bitwise
   agreement with native CUDA by making Sabah *less* exact (§7). It is a
   compatibility mode, not a correctness fix.
3. Keep the gate and require agreement with the CPU reference instead. The
   data does not support this either: Sabah and the CPU diverge at token 38 too.

## 16. What RC4 changed

| file | change |
|---|---|
| `sabah/runtime/cuda/sabah_rt.cu` | ABI v2 with per-slot input rows; in-call eviction protection; byte verification; coverage counters; live status file; test-only sentinel under a compile-time macro |
| `patches/llama.cpp/0001-…` | ABI v2 call; CUDA graphs disabled for Sabah-routed `MUL_MAT_ID`; fusion guard |
| `patches/llama.cpp/0002-…` | new: `llama-sabah-diag` capture tool |
| `sabah/server/openai_proxy.py` | `--backend sabah|reference` with identical placement; `/health` reports live runtime counters; pinned llama.cpp build as default |
| `sabah/tools/cli.py` | `serve` requires an explicit backend choice |
| `tests/test_mul_mat_id_contract.py` | new: the ggml `MUL_MAT_ID` contract on real weights |
| `tests/test_server_contract.py` | backends differ only in the executor |
| `tools/rc4/` | the harness that produced every number in this report |

