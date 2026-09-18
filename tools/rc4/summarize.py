"""Write the RC4 comparison evidence (router, ladder, logits, experts) as JSON."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import Run, router, ladder, logit_metrics, metrics, id_agree  # noqa: E402

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
R = {k: Run("D:/sabah_rc4/runs/" + v) for k, v in dict(
    cpu="ladder0_cpu", cuda="ladder0_cuda", sabah="ladder1_sabah",
    sabah_prefix="ladder0_sabah_prefix").items()}
PAIRS = [("cpu", "cuda"), ("cuda", "sabah"), ("cpu", "sabah"), ("cuda", "sabah_prefix")]
NAMES = ["ffn_moe_gate_in", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_down_in", "ffn_moe_down",
         "ffn_moe_weighted", "ffn_moe_out", "ffn_out", "l_last"]


def compact(m):
    return None if m is None else {k: m[k] for k in
                                   ("rel_l2", "max_abs", "mean_abs", "rms", "cosine", "p50", "p95", "p99")}


router_out, block_out, logit_out = {}, {}, {}
for a, b in PAIRS:
    key = "%s_vs_%s" % (a, b)
    router_out[key], block_out[key] = {}, {}
    for step in (0, 1):
        rr = router(R[a], R[b], step)
        router_out[key]["step%d" % step] = dict(
            blocks_checked=len(rr),
            selections_checked=sum(x["selections"] for x in rr),
            slot_mismatches=sum(x["slot_mismatches"] for x in rr),
            tokens_with_order_mismatch=sum(x["tokens_order_mismatch"] for x in rr),
            tokens_with_set_mismatch=sum(x["tokens_set_mismatch"] for x in rr),
            first_mismatch_block=next((x["block"] for x in rr if x["slot_mismatches"]), None),
            per_block=[dict(block=x["block"], slot_mismatches=x["slot_mismatches"],
                            tokens_set_mismatch=x["tokens_set_mismatch"],
                            weights=compact(x.get("weights"))) for x in rr])
        lad = ladder(R[a], R[b], step, NAMES)
        block_out[key]["step%d" % step] = [
            dict(block=row["block"], tokens_ids_agree=row["tokens_ids_agree"], tokens=row["tokens"],
                 **{n: compact(row.get(n)) for n in NAMES}) for row in lad]
    if R[a].complete and R[b].complete:
        logit_out[key] = {"step%d" % s: logit_metrics(R[a], R[b], s) for s in range(len(R[a].argmax))}

# per-expert breakdown for block 0, prefill, before aggregation
experts = {}
for a, b in (("cuda", "sabah"), ("cpu", "cuda"), ("cuda", "sabah_prefix")):
    key = "%s_vs_%s" % (a, b)
    ok = id_agree(R[a], R[b], 0, 0)
    ids = R[a].get("ffn_moe_topk_L0", 0).reshape(ok.size, -1)
    w = R[a].get("ffn_moe_weights_norm_L0", 0).reshape(ids.shape) if R[a].has("ffn_moe_weights_norm_L0", 0) else None
    rows = []
    for role in ("gate", "up", "down", "weighted"):
        name = "ffn_moe_%s_L0" % role
        if not (R[a].has(name, 0) and R[b].has(name, 0)):
            continue
        A, B = R[a].get(name, 0), R[b].get(name, 0)
        for slot in range(ids.shape[1]):
            m = metrics(B[ok, slot], A[ok, slot])
            rows.append(dict(role=role, slot=slot, **compact(m),
                             ref_norm=float(np.linalg.norm(A[ok, slot])),
                             test_norm=float(np.linalg.norm(B[ok, slot]))))
    # individual (token, slot) pairs for the down projection: worst and median
    A, B = R[a].get("ffn_moe_down_L0", 0), R[b].get("ffn_moe_down_L0", 0)
    pairs = []
    for t in np.where(ok)[0]:
        for s in range(ids.shape[1]):
            m = metrics(B[t, s], A[t, s])
            pairs.append(dict(token=int(t), slot=s, expert=int(ids[t, s]),
                              router_weight=float(w[t, s]) if w is not None else None,
                              ref_norm=float(np.linalg.norm(A[t, s])),
                              test_norm=float(np.linalg.norm(B[t, s])),
                              max_abs=m["max_abs"], rel_l2=m["rel_l2"], cosine=m["cosine"]))
    pairs.sort(key=lambda d: d["rel_l2"])
    experts[key] = dict(block=0, step=0, tokens_compared=int(ok.sum()), by_slot=rows,
                        down_pairs_median=pairs[len(pairs) // 2], down_pairs_worst=pairs[-3:],
                        down_rel_l2_distribution=dict(
                            p50=float(np.percentile([p["rel_l2"] for p in pairs], 50)),
                            p95=float(np.percentile([p["rel_l2"] for p in pairs], 95)),
                            max=float(max(p["rel_l2"] for p in pairs))))

for name, obj in (("router.json", router_out), ("block_errors.json", block_out),
                  ("logit_errors_ladder.json", logit_out), ("expert_errors.json", experts)):
    json.dump(obj, open(os.path.join(OUT, name), "w"), indent=1)

for k, v in router_out.items():
    for s, d in v.items():
        ws = [x["weights"]["rel_l2"] for x in d["per_block"] if x["weights"]]
        print("router %-22s %s: sel %5d  slot mism %5d  set-mism tokens %4d  first bad %s  weights relL2 med %.2e max %.2e"
              % (k, s, d["selections_checked"], d["slot_mismatches"], d["tokens_with_set_mismatch"],
                 d["first_mismatch_block"], np.median(ws) if ws else -1, max(ws) if ws else -1))
for k, v in logit_out.items():
    for s, m in v.items():
        print("logits %-16s %s: relL2 %.3e cos %.6f max %.3f top1 %d/%d top5ov %d top10ov %d margin %.3f/%.3f"
              % (k, s, m["rel_l2"], m["cosine"], m["max_abs"], m["top1_ref"], m["top1_test"],
                 m["top5_overlap"], m["top10_overlap"], m["margin_ref"], m["margin_test"]))
for k, v in experts.items():
    print("experts %-20s block0 down (token,slot) relL2: p50 %.2e p95 %.2e max %.2e"
          % (k, v["down_rel_l2_distribution"]["p50"], v["down_rel_l2_distribution"]["p95"],
             v["down_rel_l2_distribution"]["max"]))
