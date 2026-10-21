"""Document research_registry_v5.csv and the tracked results/ size in the README."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "README.md")


def rl(old, new, s):
    assert old in s, "NOT FOUND: " + old[:70]
    return s.replace(old, new, 1)


def main():
    s = open(PATH, encoding="utf-8").read()

    old = (
        "docs/                               design notes, budgets, validation reports\n"
        "results/                            raw JSON for every number quoted here\n"
        "```"
    )
    new = (
        "research_registry_v5.csv           experiment log: one row per hypothesis, the machine it\n"
        "                                    ran on, the artefact and the verdict (PASS / FAIL)\n"
        "docs/                               design notes, budgets, validation reports\n"
        "results/                            raw JSON for every number quoted here; ~46 MB tracked.\n"
        "                                    The raw per-token logit dump (logits.f32) is git-ignored;\n"
        "                                    its SHA-256 is recorded in REPORT.md instead\n"
        "```"
    )
    s = rl(old, new, s)

    open(PATH, "w", encoding="utf-8", newline="\n").write(s)
    print("applied fix 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
