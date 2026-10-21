"""Append the pinned-memory measurement that keeps `sabah serve` from starting."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """
## 29. `sabah serve` cannot start here: the warm tier exceeds the pinned grant

`sabah serve` builds a residency plan and pins the planner's warm blocks into
locked host memory before it launches the backend (`openai_proxy.serve`,
`sabah/server/openai_proxy.py:295-304`, which is `ExpertBank(..., warm_blocks=
list(projection.warm_blocks))` -> `ExpertBank.load_warm` ->
`rt.lib().sabah_host_alloc(per)` per expert).

It fails, every time, with `SabahCudaError: pinned warm alloc 704000 B failed:
cudaHostAlloc: out of memory`. Both ends of that are measured, not inferred
(`tools/probe_pinned.py`, `tools/probe_warm_budget.py`, REPORT 29):

| | value |
|---|---|
| pinned allocs wanted | 16,896 |
| pinned bytes wanted | **9.477 GB** (11 warm blocks x 3 roles x 512 experts x 460,800 B) |
| pinned bytes granted | **5.034 GB** (7,150 x 704,000 B before NULL, `tools/probe_pinned.py`) |
| ratio | **188% of what this driver grants** |

So the gap is 4.4 GB, and there is no flag that reduces `warm_blocks`: the
planner picks them from the hardware profile, and `serve` has no option to skip
or shrink the residency step. This is a rig property (a 16 GB-card WDDM driver
granting ~5 GB of page-locked host memory per process), not a configuration
mistake, and it is independent of the other recorded state of this machine
(REPORT 9's machine state, REPORT 23.1's Shared Usage).

### What still serves

The llama.cpp server started directly with the same environment the proxy would
have set (`SABAH_LLAMA=1`, `SABAH_RT_LIB`, `SABAH_LLAMA_HOT_BYTES`,
`GGML_OP_OFFLOAD_MIN_BATCH=1`, `--cpu-moe`) starts and answers: the expert
streaming path pins nothing in the serving process, so the 5.03 GB grant is not
in play. `tools/start_server.ps1` does exactly that, and
`docs/HOW_TO_RUN.md` records it as the runnable chat path.
`docs/HOW_TO_RUN.md` also carries the proxy path as unavailable until the warm
tier is capped to the grant.

### The fix this implies (not done)

Cap `ExpertBank.load_warm` at the measured grant, or make the pin failure a
warning that leaves the remaining blocks on the streaming path instead of
fatal - the runtime already serves an unpinned block from the tier. Until then,
`serve` is documented as unavailable on this rig rather than silently dropped.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 29. `sabah serve` cannot start here" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 29")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
