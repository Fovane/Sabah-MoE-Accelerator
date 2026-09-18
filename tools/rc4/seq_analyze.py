"""Sequential evidence: greedy token equivalence and per-token error growth."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Run, logit_metrics  # noqa: E402

RUNS = "D:/sabah_rc4/runs/"
OUT = sys.argv[1] if len(sys.argv) > 1 else "D:/sabah_rc4/evidence"


def load(name):
    p = RUNS + name
    return Run(p) if os.path.exists(os.path.join(p, "logits.f32")) else None


def greedy_pair(ref, test, n):
    a, b = ref.argmax[:n], test.argmax[:n]
    n = min(len(a), len(b), n)
    diff = np.where(a[:n] != b[:n])[0]
    first = int(diff[0]) if len(diff) else None
    out = dict(tokens_compared=n, identical=first is None, first_divergence=first,
               matching_prefix=first if first is not None else n)
    if first is not None:
        la, lb = ref.logits[first], test.logits[first]
        ta = np.argsort(-la)[:5]
        out.update(ref_token=int(a[first]), test_token=int(b[first]),
                   ref_margin_at_divergence=float(la[ta[0]] - la[ta[1]]),
                   ref_top5_at_divergence=ta.tolist(),
                   test_token_rank_in_ref=int(np.where(np.argsort(-la) == b[first])[0][0]))
    return out


def bitwise_logits(ref, test, n):
    n = min(len(ref.argmax), len(test.argmax), n)
    same = [bool(np.array_equal(ref.logits[s], test.logits[s])) for s in range(n)]
    return dict(steps=n, bit_identical_steps=int(sum(same)),
                all_bit_identical=all(same))


def per_token(ref, test, n):
    rows = []
    for s in range(min(n, len(ref.argmax), len(test.argmax))):
        if ref.fed[s] != test.fed[s] and s + 1 < n:
            pass  # contexts diverge only after a different token is fed
        m = logit_metrics(ref, test, s)
        rows.append(dict(step=s, context_identical=bool(np.array_equal(ref.fed[:s], test.fed[:s])),
                         logit_rel_l2=m["rel_l2"], logit_cosine=m["cosine"],
                         logit_max_abs=m["max_abs"], top1_agree=m["top1_ref"] == m["top1_test"],
                         top5_overlap=m["top5_overlap"], top10_overlap=m["top10_overlap"],
                         ref_margin=m["margin_ref"], test_margin=m["margin_test"]))
    return rows


def summarize_curve(rows):
    x = np.array([r["logit_rel_l2"] for r in rows])
    agree = [r["top1_agree"] for r in rows]
    flips = [r for r in rows if not r["top1_agree"]]
    half = len(x) // 2
    return dict(steps=len(rows), top1_agreement=float(np.mean(agree)),
                top1_flips=len(flips),
                flip_ref_margins=[round(r["ref_margin"], 4) for r in flips],
                logit_rel_l2_median=float(np.median(x)), logit_rel_l2_p95=float(np.percentile(x, 95)),
                logit_rel_l2_max=float(x.max()),
                logit_rel_l2_first_half_median=float(np.median(x[:half])) if half else None,
                logit_rel_l2_second_half_median=float(np.median(x[half:])) if half else None)


if __name__ == "__main__":
    B1, B2 = load("seq64_cuda_r1"), load("seq64_cuda_r2")
    C1, C2 = load("seq64_sabah_r1"), load("seq64_sabah_r2")
    A1 = load("seq64_cpu_r1")
    TA, TC = load("tf64_cpu"), load("tf64_sabah")
    res = {}
    for n in (16, 64):
        d = {}
        if B1 and C1: d["native_cuda_vs_sabah"] = greedy_pair(B1, C1, n)
        if B1 and A1: d["native_cuda_vs_cpu"] = greedy_pair(B1, A1, n)
        if A1 and C1: d["cpu_vs_sabah"] = greedy_pair(A1, C1, n)
        if B1 and B2: d["native_cuda_repeat"] = dict(greedy_pair(B1, B2, n), **bitwise_logits(B1, B2, n))
        if C1 and C2: d["sabah_repeat"] = dict(greedy_pair(C1, C2, n), **bitwise_logits(C1, C2, n))
        res[n] = d
        json.dump(d, open(os.path.join(OUT, "sequential_%d.json" % n), "w"), indent=1)
        print("==== %d tokens" % n)
        for k, v in d.items():
            print("  %-22s identical=%s first_div=%s %s" % (
                k, v["identical"], v["first_divergence"],
                ("ref_margin=%.3f test_rank=%d" % (v["ref_margin_at_divergence"], v["test_token_rank_in_ref"]))
                if v["first_divergence"] is not None else
                ("bitwise_logits=%s" % v.get("all_bit_identical", "n/a"))))
    curves = {}
    for name, t in (("cpu_teacher_forced", TA), ("sabah_teacher_forced", TC)):
        if B1 and t:
            rows = per_token(B1, t, 64)
            curves[name] = dict(summary=summarize_curve(rows), per_token=rows)
            s = curves[name]["summary"]
            print("  %-22s top1 agree %.3f flips %d at margins %s | logit relL2 med %.3e p95 %.3e max %.3e (1st half %.3e, 2nd half %.3e)"
                  % (name, s["top1_agreement"], s["top1_flips"], s["flip_ref_margins"],
                     s["logit_rel_l2_median"], s["logit_rel_l2_p95"], s["logit_rel_l2_max"],
                     s["logit_rel_l2_first_half_median"] or 0, s["logit_rel_l2_second_half_median"] or 0))
    json.dump(curves, open(os.path.join(OUT, "error_by_token.json"), "w"), indent=1)
