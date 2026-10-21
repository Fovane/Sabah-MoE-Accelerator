"""Replace the archaeology checkout path in docs/LLAMA_CPP_INTEGRATION_MAP.md."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "docs", "LLAMA_CPP_INTEGRATION_MAP.md")
OLD = "D:" + chr(92) + "llama-glm53"
NEW = "$WORKSPACE"


def main():
    s = open(PATH, encoding="utf-8", newline="").read()
    n = s.count(OLD)
    if n == 0:
        print("nothing to replace")
        return 0
    s = s.replace(OLD, NEW)
    with open(PATH, "w", encoding="utf-8", newline="") as f:
        f.write(s)
    print("replaced %d occurrence(s); %d bytes" % (n, len(s)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
