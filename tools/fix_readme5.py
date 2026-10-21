"""Point the README at docs/HOW_TO_RUN.md and add the q8-KV warning."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")


def rl(old, new, s):
    assert old in s, "NOT FOUND: " + old[:70]
    return s.replace(old, new, 1)


def main():
    s = open(PATH, encoding="utf-8").read()

    old = "## How to reproduce a measured number\n"
    new = "## How to run it\n\nSee [`docs/HOW_TO_RUN.md`](docs/HOW_TO_RUN.md) for the working commands: the\none-time setup, the 128K setting, what to expect per regime, the slab size for\neach context, long prompts, and the clock pinning a measurement needs.\n\n## How to reproduce a measured number\n"
    s = rl(old, new, s)

    old = "- **Deep context is disk-bound, and no policy reaches a tier on this hardware.**"
    new = (
        "- **A quantised KV cache (`-ctk q8_0 -ctv q8_0`) costs 6.3x decode on this rig**\n"
        "  (188.32 vs 30.05 ms/token, REPORT 28). It perturbs the attention output and\n"
        "  with it which experts the router selects: the same ask lands on a wider set,\n"
        "  the misses go to the disk, and the disk is already the wall. The VRAM it\n"
        "  frees is ~384 MB at 128K. Opt-in, and a numerically-equivalent mode, never a\n"
        "  default.\n"
        "- **Deep context is disk-bound, and no policy reaches a tier on this hardware.**"
    )
    s = rl(old, new, s)

    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("applied the HOW_TO_RUN link and the q8 caveat")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
