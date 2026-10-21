"""Append the environment-change section to results/phase0/REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 19. Environment: the model moved, the pin came back (October 21)

Two machine changes happened between the measurements in sections 1-18 and the
runs after them. Neither invalidates the earlier numbers, and both are recorded
so a later reader does not mistake a different configuration for a regression.

**The model directory moved from `D:` to `C:`**, same shards:

| shard | bytes | GB |
|---|---|---|
| `...-00001-of-00002.gguf` | 54825386081 | 54.8 |
| `...-00002-of-00002.gguf` | 28788186624 | 28.8 |
| total | 83613572705 | **83.6** |

The total matches the 83.6 GB recorded in section 1, so the artifact is
unchanged and the earlier measurements stand. `tools/model_dir.py` resolves the
directory from `SABAH_MODEL_DIR`, the harness prints the resolved directory at
the start of every run, and the launcher exports the same variable, so
`sabah gate`, `sabah bench` and `tools/phase0_measure.py` cannot disagree.
Seven code files that named `D:` directly and four that hardcoded the repo root
were changed; the diff is 11 files and every hunk is a line that built a path.
No evidence file under `results/` was rewritten: those are measurements, not
configuration.

**Clock pinning lost permission and came back.** For part of this session
`nvidia-smi -lgc` answered "the current user does not have permission to change
clocks" and the card idled at 405-585 MHz in P8. It works again now. The
evidence is in the two `sabah gate` runs of this session: the first reported
steady SM 832/**2628**/2767 MHz (VALID), the second 615/1012/2460 (P3,
INVALID). Both PASS the correctness checks, because the gate compares bytes
rather than time - which is exactly why it is the right thing to run when the
clock state is unknown.

The default-token gate run reproduces the earlier numbers exactly: verify
2000/0, id reuse 768/0, mapped fallback 0, over-budget 448, 14.78 GB of disk,
2110.8 MB per decoded token. Same path, same artifact, same counters.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 19. Environment" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 19, %d bytes -> %d" % (len(s), len(s + ADD)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
