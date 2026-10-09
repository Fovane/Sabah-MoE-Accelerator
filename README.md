# Sabah Accelerator

![status: experimental](https://img.shields.io/badge/status-experimental-orange)
![routing: exact](https://img.shields.io/badge/expert%20routing-exact-brightgreen)
![format: GGUF / qwen4exp](https://img.shields.io/badge/format-GGUF%20%2F%20qwen4exp-blue)
![backend: CUDA](https://img.shields.io/badge/backend-CUDA%20%2F%20NVIDIA-76b900)

**Sabah Accelerator** is an experimental runtime for routed Mixture-of-Experts (MoE) inference. Its core design goal is to preserve the model's routing decisions and execute the expert selected by the router, without substituting a different expert or skipping expert computation as an acceleration shortcut.

**Sabah Scaling** is the hardware-adaptive execution strategy behind the runtime: keep the complete expert bank in the memory tier that can accommodate it, retain frequently used experts in VRAM, and transfer the exact required expert data when needed.

The runtime is designed to improve execution efficiency on suitable hardware configurations. Acceleration is not guaranteed and depends on memory capacity, expert placement, host-memory bandwidth, PCIe bandwidth, GPU contention, workload characteristics, and runtime maturity.

If the router selects expert *E*, Sabah's intended execution semantics are straightforward: execute expert *E*. Optimization takes place below the model's mathematical function through placement, caching, scheduling, and data movement.

```text
Inspect -> Measure -> Place -> Cache -> Stream -> Execute Exact Experts
```

The principal quantities are:

```text
D_miss(C) = W * [1 - H(C)]

T_token ~= T_fixed + T_expert
           + T_unhidden_transfer + T_runtime
```

Where:

* `W` is the routed expert weight volume per token. For the inspected artifact, the reported value is 1.5043 GB per token.
* `H(C)` is the byte-weighted cache hit rate at hot-tier capacity `C`, estimated from measured routing traces.
* `D_miss(C)` estimates the expert bytes that must be fetched when they are not resident in the hot tier.
* `T_fixed` represents the fixed model execution path.
* `T_expert` represents expert computation.
* `T_unhidden_transfer` represents transfer latency that cannot be hidden behind useful computation.
* `T_runtime` represents scheduling, allocation, synchronization, and other runtime overhead.

These equations describe the intended performance model. They are not a guarantee that the individual latency terms are independent or that every workload follows the same model.

Sabah aims to increase `H(C)` and reduce unhidden transfer and runtime overhead. It does not reduce the routed expert workload by deliberately executing the wrong expert or skipping a required expert.

**Currently supported architecture:** `qwen4exp`, targeting the Qwen3.8-Flash-Next family and compatible GGUF layouts that pass Sabah's structural checks.

---

## 1. Measured Performance — Read This First

**The current validated full-model configuration is slower than stock llama.cpp.**

On the validation machine, an NVIDIA RTX 4050 Laptop GPU with 6 GB VRAM and approximately 30 GB system RAM, the v1.0 validation report records the following results:

| Runtime                   |    Decode throughput |
| ------------------------- | -------------------: |
| Stock llama.cpp           |          1.619 tok/s |
| Sabah                     |          0.411 tok/s |
| Sabah relative throughput | Approximately 0.254x |
| Relative slowdown         |  Approximately 3.94x |

Source: `results/v1_validation/benchmark.json`. The README's original measurement notes report two runs per runtime.

These results apply to the tested configuration. They must not be interpreted as a universal performance estimate for all hardware or models.

The reported bottlenecks include:

* The 77.018 GB expert bank does not fit in the validation machine's available system RAM, resulting in storage-backed expert access.
* The fixed execution path consumes most of the GPU memory budget, limiting the available expert hot tier.
* The reported full-model configuration uses a 1 GiB expert tier and records a 46% hit rate.
* Approximately 122.5 GB of data crosses PCIe during the reported benchmark.
* Expert misses use synchronous allocation/copy operations, and eviction introduces device synchronization overhead.

The precise memory configuration used for the 1 GiB tier must be distinguished from other experiments in this repository. The per-block microbenchmarks, planner examples, and full-model benchmark do not necessarily use identical memory allocations or execution paths.

This machine is currently useful for correctness validation and bottleneck analysis, not as evidence of acceleration.

**No validated multi-GPU speedup is established by the results presented in this README.**

---

## 2. What v1.0 Means

v1.0 is the first reproducible implementation of Sabah's routed-expert residency runtime with a user-accessible local API and a documented numerical-validation methodology.

The declared validation contract is:

* `docs/V1_CORRECTNESS_CONTRACT.md`
* `docs/V1_VALIDATION_REPORT.md`

The repository reports that the preregistered hard correctness gates pass. The claims below describe the intended scope of that validation; they do not establish full-model equivalence or universal performance.

### Established within the documented test scope

* Reproducible integration against pinned llama.cpp commit `96ffdc41c`, using patches 0001 and 0002.
* Structural checks that verify expert-operation routing, selected expert IDs, and expert bytes against the source GGUF.
* Numerical validation of sampled expert operations against float64 reference calculations, with a reported maximum error of `2e-6` or less under the documented comparison.
* A residency runtime with a VRAM hot tier backed by RAM or storage.
* Hardware qualification and an execution planner.
* An OpenAI-compatible local API through `sabah serve --backend sabah`.
* A correctness methodology that includes test-only wrong-expert sentinels intended to detect incorrect expert selection.
* A reproducible build and validation workflow described by the repository's documentation.

### Not established by v1.0

* Universal acceleration or acceleration on the validation machine.
* A 3x speedup.
* A validated multi-GPU inference path.
* Production readiness across all supported hardware configurations.
* Bit-exact equality of complete model outputs with llama.cpp.
* Full-model logits or greedy-token equivalence.
* Performance predictions that generalize to arbitrary routing distributions.

Sabah's expert arithmetic is reported to be closer to a float64 reference than llama.cpp's corresponding arithmetic in the tested cases. Small numerical differences can affect token selection when candidate logits are close. Consequently, matching expert-operation results within a specified tolerance does not imply that the entire model will produce identical tokens.

---

## 3. Status — `v1.0.0`

The project reports that the documented v1.0 correctness gates pass. Performance remains limited on the tested full-model configuration.

| Component                                      | Status                                                                       |
| ---------------------------------------------- | ---------------------------------------------------------------------------- |
| Model inspector and expert-layout verification | Implemented; validates supported layouts                                     |
| Hardware qualification and execution planner   | Implemented; multi-GPU execution remains unvalidated                         |
| Expert residency runtime                       | Implemented; structural and numerical claims are limited to documented tests |
| Native expert `MUL_MAT_ID` execution           | Implemented for the supported runtime path                                   |
| llama.cpp integration                          | Pinned commit `96ffdc41c`, patches 0001/0002                                 |
| Reproducible integration build                 | Documented and reported as validated                                         |
| OpenAI-compatible local API                    | Implemented; explicit backend selection required                             |
| `--validate` option                            | Intended for self-checking runs; verify availability against the current CLI |
| Correctness methodology                        | Contract, sentinels, and evaluator under `tools/v1/`                         |
| Full-model performance                         | Measured on the validation machine; slower than stock llama.cpp              |
| Multi-GPU acceleration                         | Not validated                                                                |
| Full-model logits/greedy-token equivalence     | Not yet established                                                          |

---

## 4. Quick Start

Run the following commands from the repository root:

```bash
python -m sabah.tools.cli doctor

python -m sabah.tools.cli inspect <model>-00001-of-00004.gguf

python -m sabah.tools.cli qualify <model>-00001-of-00004.gguf

python -m sabah.tools.cli plan <model>-00001-of-00004.gguf

python -m sabah.tools.cli benchmark <model>-00001-of-00004.gguf --dry-run
```

Replace the example shard name with the actual first shard of your GGUF model. For a single-file GGUF, use that file directly if supported by the CLI.

Once the CUDA runtime is built, the following commands are intended to produce execution evidence rather than planner estimates:

```bash
python -m sabah.tools.cli selftest <model>-00001-of-00004.gguf

python -m sabah.tools.cli bench <model>-00001-of-00004.gguf --bank-mode ram
```

Check `python -m sabah.tools.cli --help` if command names or options differ in your checkout.

### Self-test scope

The documented `selftest` is intended to check GPU dequantization against the supported reference representation and reproduce the designated CPU reference block.

The repository reports bit-exact dequantization for tested expert quantization types and numerical agreement for the relevant reference operations.

These are separate properties:

1. Bit-exact dequantization means the decoded representation matches the relevant reference representation under the test's comparison.
2. Numerical validation checks the resulting calculations against a numerical reference and tolerance.
3. Full-model equivalence requires additional graph-level, logits-level, or token-level comparisons.

Passing the first two does not automatically prove the third.

Run the self-test before trusting performance measurements, and inspect its actual output to confirm which checks were executed.

### Local API

Select the backend explicitly:

```bash
python -m sabah.tools.cli serve <model>-00001-of-00004.gguf --backend sabah
```

For the reference executor, where supported:

```bash
python -m sabah.tools.cli serve <model>-00001-of-00004.gguf --backend reference --allow-reference
```

Both backends are intended to use the same patched llama.cpp server, graph, and placement configuration, with the expert `MUL_MAT_ID` executor being the primary difference.

The `/health` endpoint is intended to report the active backend and, for Sabah, runtime call and lookup counters.

Backend behavior, endpoint availability, and option names must be checked against the actual build. The server should refuse to start without an explicit backend selection.

Do not expose the local server to untrusted networks without appropriate access controls.

---

## 5. Building the CUDA Components

The following commands describe the repository's basic CUDA build workflow.

`<cc>` is the GPU compute capability with the decimal point removed. Obtain it using:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Then build the qualification executable and runtime library:

```bash
cd sabah/hardware
nvcc -O3 -o mgpu_qualify.exe mgpu_qualify.cu

cd ../runtime/cuda
nvcc -O3 -arch=sm_<cc> -shared -o sabah_rt.dll sabah_rt.cu
```

These are example commands, not a universal multi-GPU build recipe. They target the CUDA toolchain and architecture specified by the build command. A binary built for one architecture is not proof that all GPU architectures in a heterogeneous system are supported.

On Windows, make the appropriate Visual Studio MSVC toolchain available before building components that require it. A typical path resembles:

```text
C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\<ver>\bin\Hostx64\x64
```

Adjust the Visual Studio edition, installation path, and version to match the actual machine.

The pinned llama.cpp integration has its own build requirements. Follow the repository's patching and rebuild instructions rather than assuming that compiling the two CUDA files alone produces a complete working server.

---

## 6. What the Tools Report

### `inspect`: Model layout

The following is the reported inspection output for the artifact used during development:

```text
arch       : qwen4exp   supported: YES
geometry   : 48 blocks, d_model 2560, 512 experts, top-10, expert_ff 640
expert bank: 77.018 GB in 24576 objects; per-expert 3,072,000/3,584,000/3,993,600 bytes
per token  : 1.5043 GB of routed expert weight
fixed path : 4.8302 GB (read every token)
contiguous : YES
note       : expert slices verified contiguous and quant-block aligned across 144 expert tensors
```

These values describe the inspected artifact and must not be generalized to other GGUF files.

The per-expert sizes are not uniform. The three reported sizes should therefore not be reduced to a claim that every expert is exactly 3.07 MB.

If an artifact's expert slices are not contiguous and quantization-block aligned under Sabah's supported layout assumptions, the runtime should refuse the optimized path rather than applying unsupported layout assumptions.

### `qualify`: Hardware measurements

The following is a reported example from the RTX 4050 Laptop validation machine:

```text
ram        : 29.8 GB total, 25.5 GB usable, ~14 GB/s read
gpus       : 1
  [0] NVIDIA GeForce RTX 4050 Laptop GPU
      6.44 GB total / 4.89 usable
      gen4 x8
      H2D 12.76 GB/s
h2d        : sum(solo) 12.76 GB/s | SIMULTANEOUS 12.79 GB/s | contention 1.00
             the planner uses the SIMULTANEOUS figure: 12.79 GB/s
```

This is an example of the qualification output, not a hardware specification.

`sum(solo)` is not a reliable estimate of aggregate multi-GPU bandwidth. Concurrent transfers may contend for shared host-memory, PCIe, or other system resources. The planner is intended to use measured simultaneous-transfer performance when available.

The simultaneous bandwidth value is meaningful only for the GPU count, transfer sizes, topology, and test conditions under which it was measured.

### `plan`: Capacity-aware planning

The repository reports the following example:

```text
execution  : STORAGE_BACKED
expert tier: 0.00 GB total = 0.0% of the bank
ram bank   : DOES NOT FIT (77.0 GB)
PROJECTED  : 1.6-2.4 tok/s at concurrency 1 (confidence: low)
note       : absolute throughput is very low whatever the speedup: this machine
             is undersized for a 111 GB artifact.
```

The throughput range is a planner projection, not a measured result. It must not be presented as an achieved benchmark.

The 0.00 GB expert tier belongs to this reported planner scenario. It should not be conflated with the 1 GiB tier reported for the full-model validation benchmark or with the hot-tier capacities used in the isolated MoE microbenchmarks.

Those experiments describe different configurations and must be compared only after their memory budgets and execution paths have been reconciled.

---

## 7. Measured MoE Microbenchmarks

The following table reports an isolated MoE-block experiment on one RTX 4050 Laptop GPU. The original measurement notes describe a real 18,960-token routing trace and 1,000 measured tokens, with 512 experts and a nominal expert size of approximately 3.07 MB.

These are per-block measurements, not full-model tokens-per-second results.

| Hot tier         | Hit rate | ms/token | GPU stall (ms) | Host staging (ms) | Fetched |
| ---------------- | -------: | -------: | -------------: | ----------------: | ------: |
| 16 slots (3%)    |    0.077 |    4.646 |          0.223 |             3.759 | 28.4 MB |
| 64 slots (12%)   |    0.309 |    3.377 |          0.203 |             2.569 | 21.2 MB |
| 256 slots (50%)  |    0.841 |    1.520 |          0.061 |             0.551 |  4.9 MB |
| 512 slots (100%) |    1.000 |    1.274 |          0.000 |             0.000 |       0 |

The reported zero-transfer row represents the measured kernel cost when all relevant experts are resident for this experiment. It does not establish the latency of the complete model or guarantee zero transfer in other workloads.

The measurements suggest that host-side staging and expert residency can dominate performance when the hot tier is small.

### 7.1. mmap-backed storage versus pinned RAM

The original measurements compare an mmap-backed expert bank with a pinned-RAM bank:

| Hot tier  | mmap bank | Pinned RAM bank | Reported speedup |
| --------- | --------: | --------------: | ---------------: |
| 16 slots  |  4.646 ms |        2.879 ms |            1.61x |
| 64 slots  |  3.377 ms |        2.186 ms |            1.55x |
| 256 slots |  1.520 ms |        1.330 ms |            1.14x |

These values are reported microbenchmark observations. Their interpretation depends on the exact transfer, allocation, synchronization, and timing boundaries used by the benchmark.

The original analysis attributes much of the small-tier latency to copying expert data from the page cache into pinned staging memory. A pinned-RAM bank can remove that particular staging copy if the runtime and transfer path support direct DMA from the resident source buffer.

However, the following timing values reported for the 16-slot pinned-RAM case do not fully add up:

```text
Kernel time:       1.274 ms
GPU stall:         1.407 ms
Sum:               2.681 ms
Observed total:    2.879 ms
Difference:        0.198 ms
```

The remaining 0.198 ms must be attributed to another measured cost or explained by different timing boundaries before claiming that the accounting closes exactly.

Likewise, transferring 28.4 MB at a measured bandwidth of 12.76 GB/s gives an idealized transfer time of approximately 2.23 ms using decimal units. Comparing this with a reported stall can provide an indication of potential overlap, but it does not independently establish the exact percentage of transfer hidden behind computation. The byte count, effective bandwidth, overlap window, and event timing must refer to the same transfer path.

**Conclusion:** The experiment supports further investigation of pinned-RAM placement and host staging. The exact overlap percentage should remain provisional until the timing accounting is reconciled.

### 7.2. Static placement versus LRU

The repository reports the following byte-weighted hit-rate comparison from `sabah calibrate`:

| Capacity | Static, oracle | Static, deployable | LRU runtime |
| -------- | -------------: | -----------------: | ----------: |
| 8 GB     |          0.361 |              0.321 |       0.621 |
| 16 GB    |          0.552 |              0.500 |       0.781 |
| 32 GB    |          0.755 |              0.698 |       0.912 |
| 48 GB    |          0.871 |              0.818 |       0.955 |

The reported experiment indicates that dynamic LRU replacement outperformed the tested static hot-set policies on the same routing trace and nominal capacity budgets.

The term *static, oracle* refers to the reported static placement that has access to future distribution information. It should not be confused with an optimal dynamic replacement policy.

The results are specific to the tested trace, capacity assumptions, expert sizes, and hit-rate calculation. They do not prove that LRU is universally optimal or that the same hit rates will occur for arbitrary traffic.

The planner reportedly uses the LRU curve for its main calibration and retains the static result as a conservative comparison. Any resulting throughput estimate remains a projection until confirmed by a full-model benchmark.

The original development notes also report that the measured LRU hit rate reached 96.4% of an offline Belady reference and that a more complex water-fill allocation improved on global popularity ordering by approximately 0.0002. These are historical experiment claims and should be interpreted according to the metric and workload used in those experiments, not as proofs of optimality.

---

## 8. Design Overview

The intended architecture has three principal components:

1. **Backing store:** The expert bank resides in system RAM when it fits, or uses storage-backed access when it does not.
2. **Hot tier:** Frequently accessed experts are retained in VRAM, subject to the configured capacity budget.
3. **Execution runtime:** On a cache miss, the runtime retrieves the correct expert data and executes the expert selected by the router.

The current design uses global popularity ordering for initial placement and LRU for replacement. The design favors simple, measurable policies rather than assuming that a more complex policy is necessarily faster.

A cache miss can introduce allocation, copying, synchronization, and transfer latency. Consequently, preserving expert selection alone does not guarantee efficient execution.

The model is autoregressive: block L consumes the output of block L-1. This constrains cross-layer prefetching because the next block's input and routing decisions depend on the preceding block's computation. Intra-block overlap may still be possible, but its feasibility depends on the graph and scheduling strategy.

The original development notes report that micro-chunking performed worse in a particular experiment, with latency increasing from 11.79 ms to 13.67 ms when the number of stages increased from one to eight. That observation supports the tested configuration's choice, but does not establish that micro-chunking is inferior for every workload or GPU.

---

## 9. Repository Layout

```text
sabah/
  core/
    model_inspector       GGUF inspection, expert layout, contiguity proof

  hardware/
    profile               Versioned HardwareProfile
    mgpu_qualify.cu       Solo and simultaneous multi-GPU H2D tests
    expert_pipe_bench.cu  Copy/compute overlap and pipeline measurements

  planner/
    planner               ExecutionPlan, estimator, no-speedup verdict

  runtime/
    rt                    ctypes binding to CUDA runtime
    expert_bank           Per-expert slices from the original GGUF
    hot_tier              VRAM residency, LRU and telemetry
    executor              MoE block execution and CPU reference
    cuda/
      sabah_rt.cu         Supported quantization kernels, copy stream,
                          events and allocation

  tools/
    cli                   inspect, qualify, plan, selftest, bench,
                          calibrate, doctor and status
    bench_moe             Per-block throughput and A/B measurements
    calib_check           Static-placement versus LRU hit curves

  tests/
    test_planner           Synthetic hardware classes and invariants
    test_quant_kernels     Quantization checks against reference data
    test_moe_block         Exactness, discrimination and residency tests
```

The CUDA implementation is reported to include Q4_K, Q5_K, Q5_1 and Q8_0 decode-and-matvec kernels. Actual support depends on the code in the checked-out revision and the quantization types encountered in the target model.

Runtime state, including hardware profiles and cache-related state, is stored under:

```text
%LOCALAPPDATA%\Sabah
```

or under `$SABAH_HOME` when configured.

**The original GGUF is opened read-only and is not modified by Sabah.**

---

## 10. Models, Privacy and Files

Sabah does not distribute model weights. Users provide their own GGUF files.

* The original GGUF is opened read-only and is not modified.
* Derived state is written to Sabah's own state directory.
* The project is designed to run locally without transmitting prompts, model contents, hardware reports, or routing traces to a remote telemetry service.
* No remote telemetry endpoint is intended by the documented design.
* Any future Hugging Face presence is intended to host Sabah profiles, configurations, architecture descriptors, measured hit curves, and expert popularity metadata rather than model weights.

These are project design and implementation claims. Verify the actual network behavior of a particular release before relying on privacy guarantees in a security-sensitive environment.

The presence of local APIs or third-party dependencies does not by itself establish that the entire execution environment is isolated from the network.

---

## 11. Not Yet Validated

The following limitations remain relevant to interpreting the current results.

### Hardware and placement

* Fixed-path sharding (M2), including its inter-GPU communication costs, remains unvalidated.
* No reported benchmark establishes multi-GPU Sabah inference performance.
* A complete expert bank fitting in system RAM has not been validated on the reported test hardware.
* Heterogeneous GPU execution across different CUDA architectures requires separate qualification and end-to-end testing.

### Runtime and performance

* The full-model Sabah pipeline requires further work before reliable end-to-end acceleration claims can be made.
* The measured validation configuration is slower than stock llama.cpp.
* Planner throughput estimates have not been established as universal predictors of observed performance.
* Allocation and synchronization overheads remain important bottlenecks.
* Longer routing traces and different workloads may produce different cache hit rates.

### Correctness

* A dedicated full-model logits/greedy-token comparison harness against llama.cpp remains outstanding.
* Expert-operation correctness does not imply bit-exact full-model equivalence.
* Numerical differences may change selected tokens when candidate logits are close.
* Validation applies to the model layouts, quantization types, execution paths and test cases covered by the documented contract.

### Routing-trace generalization

The reported hit-rate calibration uses a single 18,960-token routing trace spanning 13 families, according to the development notes.

The notes also report a hit-rate drift of approximately 0.01 per doubling of trace length in v4. The magnitude and direction of that drift should be treated as an observation from the tested trace, not as a universal correction factor.

Longer traces, different prompts, different decoding behavior and different traffic distributions may change expert popularity and cache performance.

See `docs/IMPLEMENTATION_LOG.md` for the project's implementation history and remaining work.

---

## 12. Historical / Exploratory Integration Tests

This section records a separate integration experiment. **It is not the primary v1.0 validation benchmark and must not be used as evidence of validated multi-GPU acceleration.**

The original development notes report the following test environment:

### Test configuration

* **Model:** Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S
* **Reported parameter count:** 125B
* **Reported quantization:** IQ3_S, approximately 3.44 bpw
* **Reported architecture:** 48 layers, hybrid MoE
* **Host memory:** 16 GB physical DDR RAM
* **Storage-backed memory:** 48 GB NVMe pagefile
* **Primary GPU:** NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
* **Secondary GPU:** NVIDIA GeForce GTX 1080, 8 GB VRAM
* **Reported offload:** `-ngl 16`
* **Reported tensor split:** `10,5`

The exact model artifact, operating-system memory configuration, llama.cpp commit, patch state, CUDA build targets, and test commands should accompany any reproducible publication of these results.

The reported 16 GB host-memory configuration and 48 GB pagefile should not be interpreted as 64 GB of equivalent physical RAM. Storage-backed paging has different latency and bandwidth characteristics.

### Reported exploratory measurements

| Workload                         | Runtime          | Prompt speed | Generation speed | Reported observation              |
| -------------------------------- | ---------------- | -----------: | ---------------: | --------------------------------- |
| Short math/logic                 | Native llama.cpp |    1.9 tok/s |        2.3 tok/s | Expected prime-number output      |
| Short math/logic                 | Sabah MoE Bridge |    2.0 tok/s |        2.1 tok/s | Same reported prime-number output |
| Long-form generation, 256 tokens | Native llama.cpp |    1.9 tok/s |        2.7 tok/s | Coherent narrative                |
| Long-form generation, 256 tokens | Sabah MoE Bridge |    1.9 tok/s |        2.6 tok/s | Coherent narrative                |

The original notes specify deterministic settings including `--temp 0.0` and `--seed 42`.

These measurements are retained as reported exploratory observations. Without the original logs and complete test configuration, they cannot be independently verified here or directly compared with the RTX 4050 Laptop benchmark in Section 1.

The two benchmark groups differ in reported hardware and memory configurations. Differences in throughput therefore cannot be attributed to the Sabah runtime alone.

Matching outputs on two selected prompts does not establish general mathematical equivalence, full-model correctness or equality of the underlying logits.

### Engineering observations

The original development notes identify storage-backed memory pressure as a potential bottleneck and suggest that larger physical RAM capacity may improve expert residency.

The 64 GB or larger RAM target mentioned in the development history should be understood as a proposed hardware configuration for further experiments, not as a validated minimum or a guarantee that the full expert bank will fit. The inspected expert bank alone is reported as 77.018 GB, before accounting for other model data, runtime allocations, operating-system requirements and available-memory constraints.

The original notes also report successful CUDA operation across the RTX 5060 Ti (`sm_120`) and GTX 1080 (`sm_61`) architectures during the exploratory work. This observation does not establish sustained multi-GPU inference stability, correctness of all expert operations, aggregate transfer bandwidth or performance scaling.

Before presenting these results as a validated multi-GPU benchmark, reproduce the run and document:

1. The exact llama.cpp commit and applied patches.
2. The CUDA architectures included in the build.
3. The actual GPU layer placement and tensor-split behavior.
4. Whether both GPUs performed inference work in the measured run.
5. The expert bank's memory placement and paging behavior.
6. The number of repetitions, warm-up procedure and timing boundaries.
7. The correctness checks performed beyond matching selected generated outputs.
8. The raw benchmark output and any runtime warnings or CUDA errors.

Until those details are available, the results should remain classified as exploratory.

---

## 13. Reproducibility and Reporting Rules

To keep future results comparable, every published benchmark should identify:

* Sabah version and Git commit.
* llama.cpp commit and applied patches.
* Model artifact and cryptographic hash where practical.
* Quantization format and model file size.
* GPU model, VRAM capacity and CUDA compute capability.
* Physical system RAM and available memory.
* Expert-bank placement: RAM, pinned RAM, mmap or storage-backed.
* Hot-tier capacity and cache replacement policy.
* Layer offload and tensor-split settings.
* Prompt workload, routing trace, context length and generation length.
* Warm-up procedure, run count and timing methodology.
* Prompt-processing throughput and decode throughput, reported separately.
* Correctness checks actually performed.
* Whether the result is measured, projected or unvalidated.

Use the following distinctions consistently:

* **MEASURED:** Directly observed in a specified reproducible experiment.
* **PROJECTED:** Estimated by the planner or another model.
* **UNVALIDATED:** Not established by the available tests.
* **EXPLORATORY:** Observed in a test that does not meet the project's complete validation requirements.

Do not combine measurements from different machines or configurations into a single performance claim without explaining the differences.

---

## 14. Project Goals

Sabah Accelerator is an experimental effort to investigate exact routed-expert execution under constrained GPU memory.

Its engineering priorities are:

1. Preserve the router's selected expert IDs.
2. Verify expert data against the source GGUF.
3. Qualify the target hardware instead of assuming bandwidth or capacity.
4. Make placement and cache behavior observable.
5. Measure data movement, synchronization and computation separately.
6. Prefer reproducible evidence over projected speedup claims.
7. Establish full-model correctness and performance on additional hardware before claiming broader acceleration.

The current implementation establishes a foundation for investigating these goals. It does not yet demonstrate universal acceleration, validated multi-GPU scaling or full-model equivalence with llama.cpp.

**The next milestone is not a larger speedup claim. It is a reproducible, end-to-end benchmark that demonstrates correctness and measures performance under a fully documented memory and GPU configuration.**
