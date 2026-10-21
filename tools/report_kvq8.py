"""Append the q8-KV measurement to REPORT.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "results", "phase0", "REPORT.md")

ADD = """

## 28. The quantised KV cache costs decode speed, and it is the routing

A user-facing config change (KV `q8_0` to save VRAM) was measured against the
shipped `f16` on the same prompt, context and build, `N=1`, clocks valid
(steady SM 2827 / 2591 MHz):

| KV type | steady ms/token | tok/s | expert reads | disk bytes |
|---|---|---|---|---|
| f16 (default) | **30.05** | **33.28** | 33,557 | 22.7 GB |
| q8_0 | 188.32 | 5.31 | **84,055** | 56.7 GB |

Same prompt ("Hello", 53 tokens), same `-c 2048`, same slab, only `-ctk q8_0
-ctv q8_0` added. The KV saving is real but the decode gets **6.3x slower**, and
the mechanism is measurable rather than assumed: quantising K/V perturbs the
attention output, which perturbs the hidden state, which changes which experts
the router selects. `lookups` stays identical (441,960 = 48 layers x 10 top-k x
3 tensors x 256 steps) while `reads` goes 33,557 -> 84,055 - the same total ask
landing on a **wider, less cache-resident** expert set, so the tier misses more
and the disk, which is already the wall, does more work.

Reproduced at `-c 131072` with the same shape (q8: 84055 reads, 228.7 ms;
f16: 33557 reads, 133.4 ms), and with a second prompt (Turkish, 55 tokens: q8
84,055-ish reads, f16 84,767 - the *prompt* can also drive the wide set, which is
the same effect REPORT 15.4 records for the coding trace).

### What this means for the config

* **`-ctk q8_0 -ctv q8_0` is an opt-in "numerically equivalent" mode** - the
  outputs are not byte-identical to the f16 path, and on this rig it is a
  6.3x decode-time regression, not a free saving;
* the VRAM it frees is small next to the cause of VRAM pressure (at
  `-c 131072` the whole KV cache is ~384 MB f16, REPORT 23.1), so paying 6x
  decode for it is the wrong trade on this hardware;
* a prompt that routes wide can produce the same regression with f16 KV
  (REPORT 15.4's coding trace), so "q8 is the slow one" is not the whole story -
  **the tier's miss rate is**, and q8 is one more way to raise it.
"""


def main():
    s = open(PATH, encoding="utf-8").read()
    if "## 28. The quantised KV cache costs decode speed" in s:
        print("already present")
        return 0
    if not s.endswith("\n"):
        s += "\n"
    open(PATH, "w", encoding="utf-8", newline="\n").write(s + ADD)
    print("appended section 28")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
