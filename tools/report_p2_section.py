"""Append the Phase 2 (-ub sweep) section to results/phase0/REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 20. Phase 2: the chunk read scales with the ubatch count, not with the prompt

One prompt (4486 tokens), `-b 2048` fixed, `SABAH_LLAMA_HOT_BYTES=9.8e9`,
TILE3 and SEG=48 as shipped, `N=1` per setting. The prompt is fed in three
chunks (2048 + 2048 + 390). Each chunk's disk bytes and ms, from the driver log:

| `-ub` | chunk 0 (2048 tok) | chunk 1 (2048 tok) | chunk 2 (390 tok) | total | GB/token |
|---|---|---|---|---|---|
| 512 | 136.2 s / **113.0 GB** | 345.8 s / **115.5 GB** | 30.6 s / 29.0 GB | 257.5 GB | 57.4 |
| 1024 | 109.4 s / **65.9 GB** | 200.1 s / **66.9 GB** | 170.5 s / 29.0 GB | 161.8 GB | 36.1 |

The whole 2048-token chunk is read once per ubatch that sits inside it: 4
ubatches at `-ub 512`, 2 at `-ub 1024`, 1 at `-ub 2048`. That is the mechanism,
and the tail chunk (390 tokens, one ubatch either way) reads the same 29.0 GB
in both settings, which is the control that rules out a per-call effect.

Disk on the full 2048-token chunks: **113.0/115.5 GB -> 65.9/66.9 GB, -42%**.
Total over the prompt: 257.5 -> 161.8 GB, **-37%**. Prefill wall: 512.6 ->
480.0 s (**-6%**), i.e. the disk saving does not translate 1:1 into time,
because the disk read time was already overlapped with compute
(`SABAH_IO read_ms=846.0` of a 512.6 s run). ms/token: 114.3 -> 107.0.
VRAM peak 15993 -> 16001 MiB (+8), host WS peak 5894 -> 6492 MiB (+598, the
larger activation buffers).

### Not byte-identical, so `-ub` larger than 512 is opt-in

Per the brief, the outputs were compared rather than assumed. `-ub 512` vs
`-ub 1024` over the 4 decode rows (`tools/logits_cmp.py`):

| measure | value |
|---|---|
| differing floats | 993280 / 993280 (100%) |
| max abs logit diff | 2.905e+00 |
| max rel diff | 8.493e+04 (on a value of -1.06e-05 vs 0.901, i.e. noise on a small logit) |
| top-1 agreement | **4/4 rows, 100%** |
| max abs KL per row | 2.105e-01 |

So a wider `-ub` is **numerically equivalent in what it samples** and not
byte-identical. It is therefore an opt-in mode (`--flag=-ub N`), never a
default: the default stays 512, which is what every number in sections 1-19 was
measured with. `tools/logits_cmp.py` reports the four quantities above for any
pair of runs, and was checked against a self-comparison (0 differences).

Clock record for both runs: steady SM **2850 / 2857 MHz**, memory 13801 MHz, P1
- valid, above the 2400 MHz floor. `N=1` per setting.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 20. Phase 2" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended phase-2 section, %d bytes" % len(s + ADD))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
