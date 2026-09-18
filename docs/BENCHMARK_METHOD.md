# Benchmark method

There are two deliberately separate measurements:

1. `sabah bench` measures one real MoE block, real expert bytes, a routing
   trace, and hot-tier residency. Its resident-vs-streamed A/B isolates the
   streaming penalty. It is not full-model tokens/s.
2. `sabah benchmark` records a comparable full-model reference invocation and
   its provenance. It prints `Measured speedup unavailable` until a full-model
   Sabah invocation under the same artifact, prompt, context, decoding,
   concurrency, and generation length exists.

Every result should retain the Sabah commit, model hash, runtime command,
hardware, execution plan, and raw timings. `MEASURED`, `PROJECTED`,
`SIMULATED`, and `ORACLE` are not interchangeable labels.

