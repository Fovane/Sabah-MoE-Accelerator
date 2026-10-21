"""Record the measured chat-API rates: the served path is not the 33 tok/s path."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """
## 30. Talking to it over the API: measured, and it is not the 33 tok/s number

`tools/start_server.ps1` starts `llama-server.exe` with the same environment the
proxy would have set (`SABAH_LLAMA=1`, `SABAH_RT_LIB`, `SABAH_LLAMA_HOT_BYTES`,
`GGML_OP_OFFLOAD_MIN_BATCH=1`, `--cpu-moe`), so a chat page on
`http://127.0.0.1:8080` and `/v1/chat/completions` both answer. Every number
below is the server's own `print_timing` line, not a stopwatch over HTTP
(`tools/chat_smoke.py`, `tools/chat_long.py`, `results/phase0/srv32k.log.err`,
`results/phase0/direct-smoke.log.err`):

| what | context | prompt eval | decode | decode tok/s |
|---|---|---|---|---|
| first question after load | 131072 | 504.5 ms/tok | 509.3 ms/tok | 1.96 |
| second question | 131072 | 186.5 ms/tok | 467.4 ms/tok | 2.14 |
| third question | 131072 | 208.5 ms/tok | 511.3 ms/tok | 1.96 |
| 200-token generation | 131072 | 246.0 ms/tok | 509.1 ms/tok | 1.96 |
| first question after load | 32768 | - | 471.8 ms/tok | 2.12 |
| second question | 32768 | 149.2 ms/tok | 417.4 ms/tok | 2.40 |

### What the table says

* The 200-token case settles at **1.96 tok/s** and stays there - this is the
  steady regime of the served path, not a warm-up artefact.
* **The context barely matters**: 32K and 128K differ by <20% (417 vs 509
  ms/tok). So the deep-context decode work (REPORT 22) is not what a chat user
  is waiting on.
* What they are waiting on is the expert fetch. During those decodes the GPU is
  at `4 % utilisation, 487 MHz, P8` (`nvidia-smi`) - the card is idle, waiting
  for the host side to bring experts in. That is the same wall REPORT 15.4 and
  REPORT 28 describe: on this rig the tier misses go to the SSD, and the SSD is
  the bound.
* The 33.3 tok/s in the README is the **narrow-prompt** regime: `"Hello"`
  routes 33,557 expert reads (REPORT 28), while a chat-templated Turkish prompt
  routes the wide set (~84,767 reads on the diag path). Same rig, 2.5x the ask,
  and the served path lands ~100x below the narrow-prompt diag number. A number
  that only holds for one prompt's routing is not a headline.

So `docs/HOW_TO_RUN.md` states the served rates as ~2 tok/s per question and
keeps 33.3 tok/s labelled as the short-context, narrow-prompt diag measurement.
Anything faster on this hardware needs fewer bytes per expert (a lower-bit
quantisation), a faster drive, or host RAM as a tier - all three are recorded
out of scope in REPORT 25 and REPORT 27.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 30. Talking to it over the API" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
