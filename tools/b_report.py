"""Compare the cached bv-* runs: final artifacts byte equality + grouping active."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GROUPS = {
    "bv-grp650-2ub": ("bv-ref650-2ub", "no"),
    "bv-grp650-2ub-g24": ("bv-ref650-2ub", "no"),
    "bv-grp1888-4ub": ("bv-ref1888-4ub", "no"),
    "bv-grp1888-1ub": ("bv-ref1888-1ub", "yes"),
    "bv-grp1888-1ub-g24": ("bv-ref1888-1ub", "yes"),
}


def blob(label):
    p = os.path.join(ROOT, "results/phase0", label, "main")
    out = b""
    for f in ("fed.i32", "argmax.i32", "logits.f32"):
        out += open(os.path.join(p, f), "rb").read()
    return out


def passes(label):
    dl = os.path.join(ROOT, "results/phase0", label, "driver.log")
    if not os.path.exists(dl):
        return None
    for line in open(dl, errors="replace"):
        if "prefill group" in line:
            return line.split("passes=")[1].split()[0]
    return None


def main():
    print("label                  | reference           | artifacts  | passes (expected)")
    print("-----------------------+---------------------+------------+-----------------")
    for a, (b, expect) in GROUPS.items():
        same = blob(a) == blob(b)
        p = passes(a)
        print(f"{a:22s} | {b:19s} | {'IDENTICAL' if same else 'DIFF       '} | "
              f"{p} ({expect})")


if __name__ == "__main__":
    raise SystemExit(main())
