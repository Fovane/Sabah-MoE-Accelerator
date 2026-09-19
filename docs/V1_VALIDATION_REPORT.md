# Sabah v1.0 — validation report

**Did the preregistered v1.0 contract pass? Yes, on the second API attempt,
as the contract prescribes.** Attempt 1 failed three API gates on a real proxy
defect. The defect was fixed, the affected runs were rerun in full, and every
gate passed. Both attempts are recorded here and in
`results/v1_validation/`.

| identity | value |
|---|---|
| `V1_CONTRACT_COMMIT` | `ca0a222777d1bff1461d079826b0043a67848b5d` (`docs/V1_CORRECTNESS_CONTRACT.md`) |
| implementation commit, diag runs R1–R13, R17 | `ca0a222` |
| implementation commit, API runs R15–R16 | `3395267` (proxy timeout fix; see §3) |
| llama.cpp | `96ffdc41ceb055e1c2d3d96667ae6d9f0ccb710b` + patches 0001/0002 (hashes in `environment.json`) |
| model | Qwen3.8-Flash-Next UD-Q4_K_XL, shard 1 SHA-256 `44481862…3ec8082` |
| hardware | RTX 4050 Laptop 6 GB (sm_89), ~30 GB RAM, Windows 11, CUDA 13.3 |
| runtime / llama.cpp binaries | byte-identical across both attempts (`environment*.json`) |
| tests | 23 (unit, server contract, native `MUL_MAT_ID` contract incl. 3 sentinels) |
| corpus | 14 chat prompts (7 categories × 2) + 4 boundary prompts (511, 512, 513, 1100 tokens); never executed before the contract commit |
| decode steps | 32 teacher-forced steps per prompt per path (448 per path) |
| concurrency | API groups of 1, 2 and 4 simultaneous requests; 4 sequences sharing every decode ubatch (diag) |

## 1. Gates

| gate | rule (frozen) | result | evidence |
|---|---|---|---|
| A build / reproducibility | patches reproduce the tested sources, clean build, same smoke argmax | **PASS** | 5/5 sources identical; build OK; argmax equal (`R18_build.json`) |
| B structural exactness | coverage exact per prompt; lookups = 10 × tokens; ids = router top-k; weighted = down × weight bitwise; bytes verified; sentinel = 0 | **PASS** | 13 runs; 1,054,077 token-rows; 2,058,750 verified fetches, 0 failures; 4,662 captured rows with exact id/weight association |
| C float64 op-level ≤ 2e-6 | every sampled op, all blocks, all 10 slots, every prompt | **PASS** | 27,972 samples, max **9.25e-7**; self-check 0 failures in 152,622 product samples |
| D wrong-expert discrimination | 12/12 sentinel trials caught by both detectors; multi-sequence sentinel caught | **PASS** | 12/12; multi-sequence S3 flagged 39,840 of 39,960 samples |
| E residency invariance, determinism | logits bitwise equal at 128 MiB / 512 MiB / 1 GiB; repeat bitwise equal | **PASS** | 2 prompts × 8 steps × 3 capacities bitwise; repeat bitwise |
| F multi-sequence / concurrency | F1: diag 4-sequence shared ubatches exact; F2: API groups 1/2/4 exact | **PASS** (attempt 2) | 39,960 all-slot samples, max 8.5e-7; each decode step one ubatch of 4 rows; API 4-request group 62,418 token-rows exact |
| G prefill > ubatch | 511/512/513/1100: coverage exact, op-level clean, every ubatch captured | **PASS** | 1, 1, 2, 3 ubatches; max 8.1e-7 |
| H full-graph non-inferiority | UB95(R) ≤ 1.25 | **PASS** | R = **0.981**, UB95 = **1.033** |
| I greedy flip non-inferiority | UB95(D) ≤ 0.03 | **PASS** | D = **0.0022**, UB95 = **0.0089** (Sabah 4 flips, CPU 3, of 448) |
| J real Sabah API backend | identity, F2, stream = non-stream twin, exact counters | **PASS** (attempt 2) | `llama.cpp+sabah` / `SABAH_NATIVE_MUL_MAT_ID`; 11-chunk stream identical to its twin |
| K evidence-grade counters | every counter check exact | **PASS** (attempt 2) | all diag and API windows exact |
| L measured benchmark | available | **AVAILABLE** | **0.254× stock llama.cpp** (decode) |
| M claim hygiene | checklist | **PASS** | 6/6 (`claims_check.json`) |

## 2. Numbers

**Operation level.** Each sampled `MUL_MAT_ID` output was recomputed in
float64 from its own captured input and ids against gguf-py's dequantization
of the original GGUF bytes. 27,972 samples, worst relative L2 9.25e-7. The
in-runtime self-check, an independent host dequantizer, found no failure
across all product runs.

**Full graph (H).** Per-prompt ratio of mean logit error vs native CUDA
(Sabah / CPU), from 0.80 (`turkish_prose_1`) to 1.25 (`python_code_1`). The
aggregate is R = 0.981 (paired prompt-cluster bootstrap, 20,000 resamples, seed
20260919, one-sided 95% UB 1.033).

**Greedy (I).** Teacher-forced flips vs native CUDA: CPU 3, Sabah 4, of 448
steps each. Reference margins at Sabah's flips: 0.018, 0.065, 0.129 and
**1.348** (`json_2`, step 2 — not a near tie). CPU's flips: 0.065, 0.129,
0.129. Free-running agreement, derived exactly from the same runs: identical
for 16 tokens on 12/14 prompts (CPU 13/14), and for 32 tokens on 10/14 (CPU
11/14).

**API A/B.** Reference vs Sabah through `sabah serve`, 4 prompts × 16 tokens:
**4/4 identical**.

**Performance (MEASURED).** Prompt `math_1`, 64 tokens, 2 runs per path,
checks off:

| path | decode tok/s | median ms/token | prefill tok/s |
|---|---|---|---|
| stock llama.cpp | 1.624 / 1.615 | 517 / 536 | 0.82 |
| Sabah | 0.410 / 0.412 | 2,248 / 2,330 | 0.68 |

**MEASURED SPEEDUP = 0.254**: Sabah is about 3.9× slower on this machine.
Sabah's hot-tier hit rate was 46%, and 122.5 GB moved over PCIe. The machine
cannot hold the 77 GB expert bank in RAM; see `KNOWN_LIMITATIONS.md`.

## 3. Attempt 1: what failed and what was done

In attempt 1, the API runs failed gates F2, J and K. Under `--validate`, the
4-request group took longer than the proxy's hard-coded **600 s** backend
timeout. The proxy returned HTTP 502 for all four requests while llama-server
kept computing, and that work then landed in the next request's counter
window. The exact counters caught it: expected 7,920 token-rows, observed
8,276.

- **Root cause:** a fixed timeout in `sabah/server/openai_proxy.py`.
- **Fix (`3395267`):** no backend read limit by default, with an opt-in
  `--backend-timeout`, plus regression tests using a deliberately slow
  backend.
- **Scope:** only R15 and R16 load the proxy. Their attempt-1 outputs are
  archived (`attempt1_api_failure.json`), and both were rerun in full on
  `3395267`. The runtime and llama.cpp binaries are byte-identical across the
  attempts, so all other evidence remains valid under contract §0.
- **Rules:** no threshold, sample or rule was changed.

## 4. Development evidence vs confirmatory evidence

RC4 (`results/rc4_numerical/`) and the v1 development runs (`dev_*` prompts)
designed the contract. Only the frozen corpus above is confirmatory.

## 5. Known limitations

See `KNOWN_LIMITATIONS.md`, in particular:

- the tested configuration is slower than stock llama.cpp;
- one machine, one model, 14 prompt clusters;
- gate I has ~0.79 power at its margin;
- full-graph comparisons were made on identically instrumented graphs;
- llama.cpp's native CUDA Q5_1 down-projection residual is unexplained.

Multi-GPU remains PROJECTED / UNVALIDATED.
