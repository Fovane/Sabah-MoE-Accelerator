"""Record why 30 tok/s is a narrow-prompt number, with the arithmetic that bounds it."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """
## 31. Why 30 tok/s is not reachable for a chat prompt here, arithmetically

The question is fair, so here are the two measurements side by side and the
number each one implies (REPORT 28, same rig, same build, same slab, N=1):

| prompt | reads (256 steps) | **reads/token** | ms/token | tok/s |
|---|---|---|---|---|
| `"Hello"` (53 tokens) | 33,557 | **131** | 30.05 | 33.28 |
| `"Merhaba"` (55 tokens) | 84,767 | **331** | 157.24 | 6.36 |

Same `-c 2048`. The prompt alone moves the answer by 5.2x, so shrinking the
context does not buy the 30 back - at `-c 2048`, where 30 ms is measurable at
all, the chat-style prompt already costs 157 ms.

### What the two rows imply

`"Hello"` is routed onto a set that is already mostly resident: 131 reads/token
at 30.05 ms/token means the model's own per-layer work at batch 1 plus a fetch
that is almost all hits. That is the floor this rig has for this model, and it
is what the README headline reports.

`"Merhaba"` asks for 331 reads/token of 460,800 B each. Served from the SSD that
is measured at 1.05-1.45 GB/s, so 331 reads = 152 MB/token = **105-145 ms/token of
pure fetch**. Measured 157.24 ms, so the fetch is essentially all of it and the
model's own ~30 ms floor is the smaller part. The reads do not fit the tier:
the slab at 9.8e9 evicts continuously (`cache 63024 hits, 21816 misses, 7739
evictions, resident 9.80 GB of 9.8e9`), which is a 74% hit rate - and 74% of
331 reads still leaves 86 misses/token to the SSD.

### What it would take, priced

| lever | what it buys | why it does not reach 30 tok/s |
|---|---|---|
| host RAM as a second tier | 86 misses/token at 0.047 ms instead of 0.45 ms | ~35 ms saved, so ~120 ms = **8 tok/s** (`docs/SECOND_TIER.md`'s own measurement: worth ~1.5x on the fill, nothing on the steady state) |
| a 6 GB/s drive | REPORT 26: 7.1 -> ~10.5 tok/s end to end | the fill improves, the miss count does not |
| `-c` 512 instead of 131072 | already priced above | the prompt is the cause, not the window |
| **fewer bytes per expert** | the only lever that cuts the miss count | the MoE at IQ3_S is 3.4375 bpw; at ~2 bpw the same 9.8 GB slab holds ~1.7x as many experts, so a much larger share of the wide set is resident |

The last row is the only one that addresses the cause: the tier misses because
the expert set is wide and each expert is large. Making experts smaller makes
more of the wide set fit, which is a model-quantisation decision, not something
this runtime can do on its own - and it costs quality, which is the one thing
`gate` does not and cannot prove.

So the honest statement is: **30+ tok/s is the narrow-prompt regime on this rig,
and it is a hardware ceiling for a wide-routing prompt**, not an unoptimised
path. What is left to do is fewer bytes per expert or a faster drive, and both
are recorded out of scope (REPORT 25, REPORT 27).
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 31. Why 30 tok/s is not reachable" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 31")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
