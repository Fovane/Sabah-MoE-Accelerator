"""Gate M: claim hygiene of README.md (contract 6.10), as an explicit checklist.

Each item is a literal check on the README text, so the result is reproducible.
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))


def main(speedup: float | None, out: str):
    t = open(os.path.join(V5, "README.md"), encoding="utf-8").read()
    low = t.lower()
    items = []

    def item(name, ok, detail=""):
        items.append(dict(check=name, ok=bool(ok), detail=detail))

    item("has a 'What v1.0 means' section", re.search(r"^#+\s*what v1\.0 means", t, re.I | re.M))
    if speedup is not None and speedup < 1.0:
        item("states it is slower than stock llama.cpp", "slower than stock llama.cpp" in low)
        item("states the measured ratio", ("%.2f" % speedup) in t or ("%.3f" % speedup) in t,
             "looking for %.2f" % speedup)
    item("multi-GPU labelled PROJECTED/UNVALIDATED",
         re.search(r"multi-gpu[^\n]*(projected|unvalidated)|(projected|unvalidated)[^\n]*multi-gpu", low))
    bad = [m.group(0) for m in re.finditer(r"[^\n]*bit-exact[^\n]*", low)
           if not any(k in m.group(0) for k in ("decode", "dequant", "quant type", "logits of", "not bit-exact",
                                                "never", "not claimed", "no claim"))]
    item("no end-to-end bit-exact claim", not bad, "; ".join(bad)[:300])
    accel = [m.group(0) for m in re.finditer(r"[^\n]*\b(accelerates|faster than)\b[^\n]*", low)
             if not any(k in m.group(0) for k in ("designed to", "not ", "slower", "projected", "depends", "suitable",
                                                  "if ", "whether", "can "))]
    item("no sentence claims measured acceleration", not accel, "; ".join(accel)[:400])
    res = dict(items=items, ok=all(i["ok"] for i in items),
               summary="%d/%d checks" % (sum(i["ok"] for i in items), len(items)))
    json.dump(res, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for i in items:
        print("%-4s %s %s" % ("ok" if i["ok"] else "FAIL", i["check"], i["detail"]))
    return res


if __name__ == "__main__":
    sp = float(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1] != "-" else None
    main(sp, sys.argv[2] if len(sys.argv) > 2 else os.path.join(V5, "results", "v1_validation", "claims_check.json"))
