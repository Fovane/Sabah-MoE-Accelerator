"""Apply the review fixes to README.md. See the commit message for the list."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")

BT = chr(96)  # backtick


def rl(old, new, s):
    assert old in s, "NOT FOUND: " + old[:60]
    return s.replace(old, new, 1)


def main():
    s = open(PATH, encoding="utf-8").read()

    # 2. label the "optimised" table's tok/s column as the warm-up regime
    old = (
        "Everything here came from measuring the token, not from guessing. Baseline to\n"
        "current is **4.37 -> 4.84 tok/s (+11%)**, with no change to the outputs.\n"
        "\n"
        "| change | cost before | after | tok/s |\n"
        "|---|---|---|---|\n"
    )
    new = (
        "Everything here came from measuring the token, not from guessing. **Every tok/s\n"
        "figure in this table is the warm-up regime** (the tier still filling, 6-token\n"
        "runs) - not the steady state, and not comparable with the headline table above,\n"
        "whose figures are steady-decode. Baseline to current in that regime is\n"
        "**4.37 -> 4.84 tok/s (+11%)**, with no change to the outputs.\n"
        "\n"
        "| change | cost before | after | tok/s (warm-up) |\n"
        "|---|---|---|---|\n"
    )
    s = rl(old, new, s)

    # 3. lever 4 is stale: TILE3 and the segmented graph moved the floor
    old = (
        "| 4 | **The floor** (the steady state, measured) | 67.3 ms/token with the disk removed entirely = "
        "`30.3` per-position work + `37/B` per-call cost, and the per-call cost is the per-layer router drain "
        "(ids ~50 ms/token at B=1). It *is* the steady state, so it is the only thing that moves 15 tok/s | "
        "medium: needs the batch (lever 3's surviving half) or faster expert kernels |"
    )
    new = (
        "| 4 | ~~**The floor** (the steady state, measured)~~ **measured, and it moved twice** | "
        "The floor was 67.3 ms/token (disk removed) = `30.3` per-position work + `37/B` per-call cost. "
        "**Both halves have since moved on this rig**: a bit-exact quant kernel (TILE3) took the per-token "
        "cost 60.56 -> 44.21 ms, and segmented CUDA graphs removed GPU-side idle time to take the steady "
        "decode 42.74 -> **30.36 ms** (32.93 tok/s) at 128K, byte-identical. The floor is no longer fixed | "
        "done for this rig: kernel + graph, both byte-exact. Remaining: batching (lever 3) |"
    )
    s = rl(old, new, s)

    # 3b. the Status "not done" text repeats the stale floor claim
    old = (
        "Lever 4 is\n"
        "attributed rather than fixed: the steady state is the floor, `30.3 + 37/B`, and\n"
        "moving it needs batching *real* concurrent requests or faster expert kernels."
    )
    new = (
        "Lever 4 is **done for\n"
        "this rig**, and the numbers are REPORT 16: the bit-exact TILE3 kernel took the per-token cost\n"
        "60.56 -> 44.21 ms, and segmented CUDA graphs took the steady decode 42.74 -> 30.36 ms\n"
        "(32.93 tok/s) at 128K, byte-identical. What still moves it is batching real concurrent\n"
        "requests (lever 3's surviving half), not another kernel."
    )
    s = rl(old, new, s)

    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("applied fixes 2, 3, 3b")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
