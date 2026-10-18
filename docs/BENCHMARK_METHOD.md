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

## Any disk number must draw fresh offsets

This drive serves a repeated read about 1.5x faster than a fresh one, so a
probe that draws one offset list and re-reads it for every case reports a
throughput the model can never see. On the reference rig (676 KB reads, the
expert stride, whole file): fresh offsets 1187-1293 MB/s at qd 1-8, the same
list re-read 2318-2320 MB/s at qd 4-8. Every disk number in this repository
before `disk_probe.py` learned to redraw per case - including the 2.4 GB/s that
levens 1 and 2 were projected from - is the second column.

`tools/disk_probe.py` draws fresh offsets per case (`--reuse-offsets` brings the
old behaviour back for comparison); `tools/io_bench.cu` takes a seed and reports
the sum of the individual read calls next to the wall time, so overlap and
throughput can be read separately. `SABAH_IO` prints the same overlap ratio for
the real runtime.

