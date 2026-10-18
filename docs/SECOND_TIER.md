# Is a second tier worth it? (lever 5, measured)

The plan's fifth lever was to put the GTX 1080's 8 GB of VRAM in front of the
disk: on a tier miss, look in the 1080 instead of reading 635 KB from the NVMe;
on an eviction, park the victim there instead of dropping it. The projection in
the README claimed a +18% ceiling and ~54 ms/token of PCIe traffic.

That projection was about *capacity* and assumed the path was cheap. The path is
not cheap. This document is the measurement, taken with
[`tools/second_tier_probe.cu`](../tools/second_tier_probe.cu) on this rig
(RTX 5060 Ti as the serving GPU, GTX 1080 as the candidate tier, 15.38 GiB host
RAM), and the conclusion is that **no configuration of the 1080 beats the disk**,
while the one path that does beat it is host RAM - a decision this project has
already made the other way.

## What the probe measures

One expert-sized chunk (650240 B, the runtime's grain) transferred repeatedly,
per direction, alone and eight in flight, on dated hardware with the current
driver. Both GPUs are allocated, and every pair is measured, including each
device to itself (the host-RAM-tier path).

```
peer access:          0 -> 1: no        1 -> 0: no
--- via pinned host, one chunk at a time ---
  0 -> host -> 0: 0.047 ms/chunk (0.023 out + 0.024 in)  13.88 GB/s  [host-RAM tier]
  1 -> host -> 0: 0.224 ms/chunk (0.200 out + 0.024 in)   2.90 GB/s
  0 -> host -> 1: 0.230 ms/chunk (0.023 out + 0.206 in)   2.83 GB/s
  1 -> host -> 1: 0.407 ms/chunk (0.200 out + 0.207 in)   1.60 GB/s  [host-RAM tier]
--- via pinned host, 8 chunks in flight ---
  0 -> host -> 0: 0.048 ms/chunk (0.025 out + 0.024 in)  13.48 GB/s  [host-RAM tier]
  1 -> host -> 0: 0.224 ms/chunk (0.200 out + 0.024 in)   2.90 GB/s
  0 -> host -> 1: 0.231 ms/chunk (0.025 out + 0.207 in)   2.81 GB/s
  1 -> host -> 1: 0.407 ms/chunk (0.200 out + 0.207 in)   1.60 GB/s  [host-RAM tier]
--- device to device ---
  0 -> 1: no peer access (the copy falls back to the host path)
  1 -> 0: no peer access (the copy falls back to the host path)
```

Depth changes nothing: the links are already saturated at one chunk in flight
(0.047 ms at depth 1, 0.048 ms at depth 8). The link facts behind the asymmetry:
the 5060 Ti negotiates PCIe 5.0 x8, the 1080 sits at **gen3 x4** and is the
bottleneck of every path that touches it (`nvidia-smi`, and `0.200 ms` per
635 KB one way = 3.2 GB/s = gen3 x4).

For reference, this rig's disk serves a fresh 635 KB read in **~0.45 ms**
(1.4 GB/s), and the runtime's own demand path spends 0.49 ms per read.

## The economics

A second tier only pays if the traffic it *saves* costs more than the traffic it
*adds*. One expert:

| path | cost per 635 KB | vs disk |
|---|---|---|
| disk, fresh offsets | 0.45 ms | 1.0x |
| 1080 hit (1080 -> host -> 5060 Ti) | **0.224 ms** | **0.50x** |
| 1080 park (5060 Ti -> host -> 1080) | **0.230 ms** | 0.51x |
| 1080 park **plus** 1080 hit | **0.454 ms** | **1.01x** |
| host-RAM park + hit (5060 Ti -> host -> 5060 Ti) | **0.047 ms** | **0.10x** |

That middle row is the verdict. A cache entry that was ever *parked* costs
0.454 ms to serve once, and the disk would have served it in 0.45 ms. **A 1080
tier is break-even at a 100% hit rate and loses from there**, because it has no
peer access (so every transfer is a full host round trip) and its link is half
the 5060 Ti's width at a third of the generation.

Concretely, per decode token, at the measured rates (247.4 misses, 239 evictions,
121.8 ms of disk):

| policy | disk bytes served | cost of the new path | total | vs today |
|---|---|---|---|---|
| today | 247 misses | - | 121.8 ms | 1.00x |
| every miss served from the 1080, every eviction parked | 0 | 247 x 0.224 + 239 x 0.230 = 110 ms | 110 ms | 0.90x |
| the same, but only 15% of experts can be resident | 210 | 37 x 0.224 + 36 x 0.230 = 16.6 ms | 95 + 16.6 = 112 ms | 0.92x |

The first row of that table is the *impossible* case (an 8 GB tier holding the
whole working set) and it still only breaks even. The realistic case saves ~10 ms
and needs the same residency machinery.

## What is left of the idea

* **1080 as a cache: dead.** The path costs what the disk costs, and it has the
  extra failure mode of a *park* that turns a future hit into no cheaper than a
  fresh read.
* **Host RAM as a cache: real, and cheap, and off-message.** 0.047 ms per expert
  round trip, 13.9 GB/s, 9.6x cheaper than the disk. A ~6 GB pinned expert tier
  would take the working set from 9.8 GB to ~16 GB, which the measured LRU curve
  puts at ~92.5% hit, i.e. ~61 ms of disk instead of 122, plus ~145 x 0.047 =
  7 ms of second-tier traffic and ~239 parks x 0.047 = 11 ms: **~128 ms/token,
  1.5x**. But it is pinned host RAM on a machine this project exists to spare
  (the README's premise is RAM-constrained hardware, and the expert path uses
  `FILE_FLAG_NO_BUFFERING` specifically to keep host RAM free), the machine
  already carries the model's own 28.8 GB PLE mapping, and the RAM budget is the
  user's decision, not the runtime's. It is also, in effect, what the OS page
  cache would do for free if unbuffered I/O were turned off.
* **1080 as a static expert partition (compute offload): the one positive 1080
  shape.** Not caching bytes but *owning* a slice of the expert bank: ~7.5 GB
  usable holds 11.5k of the 73728 `(tensor, expert)` objects, 15.6%, so ~15% of
  lookups would never touch disk or VRAM. Ceiling: 15% of the disk time (~16 ms,
  8%) plus whatever the 1080's own compute overlaps, minus the per-layer
  activation round trips (2 x 8 KB per offloaded expert, 5.5 us, plus a
  cross-device sync per layer). It needs a second-device execution path in the
  bridge, and the win is smaller than the host-RAM tier's. Recorded as a
  candidate; not implemented.

## Reproduce

```bat
:: from a developer command prompt (vcvars64 on PATH):
nvcc -O3 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_120,code=sm_120 ^
     -o second_tier_probe.exe second_tier_probe.cu
second_tier_probe.exe 8
```

The probe is self-contained: it allocates on every device it finds, measures both
directions through pinned host memory and peer-to-peer where the driver allows
it, and prints the disk comparison it should be read against.
