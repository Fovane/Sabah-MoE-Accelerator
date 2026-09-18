"""Compare sabah-diag captures between execution paths.

Every metric is computed on dense float64 copies of the captured tensors. Router
ids are compared exactly; weights are compared only where the ids agree, and
always in the router's own slot order, so an id is never separated from its
weight.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import numpy as np


class Run:
    def __init__(self, path):
        self.path = path
        rp = os.path.join(path, "run.json")
        self.run = json.load(open(rp)) if os.path.exists(rp) else {}
        self.nv = self.run.get("n_vocab", 248320)
        self.idx = defaultdict(dict)          # key -> step -> record
        ip = os.path.join(path, "cap", "index.jsonl")
        if os.path.exists(ip):
            for line in open(ip):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue            # truncated tail of a crashed run
                # a key can be written more than once per step (e.g. ids shared
                # by gate and up); keep the first
                self.idx[r["key"]].setdefault(r["step"], r)
        lp = os.path.join(path, "logits.f32")
        self.complete = os.path.exists(lp)
        self.logits = np.fromfile(lp, np.float32).reshape(-1, self.nv) if self.complete else None
        self.argmax = np.fromfile(os.path.join(path, "argmax.i32"), np.int32) if self.complete else np.array([], np.int32)
        self.fed = np.fromfile(os.path.join(path, "fed.i32"), np.int32) if self.complete else np.array([], np.int32)

    def has(self, key, step):
        return step in self.idx.get(key, {})

    def get(self, key, step):
        r = self.idx[key][step]
        dt = np.float32 if r["type"] == "f32" else np.int32
        with open(os.path.join(self.path, "cap", key + ".bin"), "rb") as f:
            f.seek(r["offset"])
            a = np.frombuffer(f.read(r["nbytes"]), dt)
        ne = r["ne"]
        return a.reshape(ne[3], ne[2], ne[1], ne[0])[0] if ne[3] == 1 else a.reshape(ne[::-1])

    def buf(self, key, step):
        return self.idx[key][step].get("buf", "?")


def metrics(a, b):
    """Error of `a` against `b` (b is the reference)."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    nb = np.linalg.norm(b)
    na = np.linalg.norm(a)
    return dict(
        n=int(a.size),
        max_abs=float(d.max()) if d.size else 0.0,
        mean_abs=float(d.mean()) if d.size else 0.0,
        rms=float(np.sqrt((d ** 2).mean())) if d.size else 0.0,
        rel_l2=float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b)),
        cosine=float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0,
        p50=float(np.percentile(d, 50)) if d.size else 0.0,
        p95=float(np.percentile(d, 95)) if d.size else 0.0,
        p99=float(np.percentile(d, 99)) if d.size else 0.0,
        ref_rms=float(np.sqrt((b ** 2).mean())) if b.size else 0.0,
    )


def weight_key(run):
    for k in ("ffn_moe_weights_scaled", "ffn_moe_weights_norm", "ffn_moe_weights"):
        if any(key.startswith(k + "_L") for key in run.idx):
            return k
    return None


def router(ref, test, step, n_layers=48):
    """Exact id comparison per block; weights compared on id-agreeing tokens."""
    wk = weight_key(ref)
    out = []
    for il in range(n_layers):
        k = "ffn_moe_topk_L%d" % il
        if not (ref.has(k, step) and test.has(k, step)):
            continue
        a = ref.get(k, step).reshape(-1, ref.get(k, step).shape[-1])
        b = test.get(k, step).reshape(a.shape)
        tok_mismatch = np.where((a != b).any(axis=1))[0]
        set_mismatch = [t for t in range(a.shape[0]) if set(a[t]) != set(b[t])]
        row = dict(block=il, tokens=int(a.shape[0]), selections=int(a.size),
                   slot_mismatches=int((a != b).sum()),
                   tokens_order_mismatch=int(len(tok_mismatch)),
                   tokens_set_mismatch=int(len(set_mismatch)),
                   first_mismatch_token=int(tok_mismatch[0]) if len(tok_mismatch) else None)
        if wk:
            wkey = "%s_L%d" % (wk, il)
            if ref.has(wkey, step) and test.has(wkey, step):
                wa = ref.get(wkey, step).reshape(a.shape)
                wb = test.get(wkey, step).reshape(a.shape)
                ok = np.ones(a.shape[0], bool)
                ok[tok_mismatch] = False
                if ok.any():
                    row["weights"] = metrics(wb[ok], wa[ok])
        out.append(row)
    return out


PER_SLOT = ("ffn_moe_gate", "ffn_moe_up", "ffn_moe_down_in", "ffn_moe_down", "ffn_moe_weighted")


def id_agree(ref, test, il, step):
    k = "ffn_moe_topk_L%d" % il
    if not (ref.has(k, step) and test.has(k, step)):
        return None
    a = ref.get(k, step)
    a = a.reshape(-1, a.shape[-1])
    b = test.get(k, step).reshape(a.shape)
    return (a == b).all(axis=1)


def ladder(ref, test, step, names, n_layers=48):
    """Per-slot tensors are compared only on tokens whose routed ids agree in
    both paths; comparing different experts slot-by-slot is meaningless."""
    rows = []
    for il in range(n_layers):
        row = dict(block=il)
        ok = id_agree(ref, test, il, step)
        row["tokens_ids_agree"] = int(ok.sum()) if ok is not None else None
        row["tokens"] = int(ok.size) if ok is not None else None
        for n in names:
            k = "%s_L%d" % (n, il)
            if ref.has(k, step) and test.has(k, step):
                a, b = ref.get(k, step), test.get(k, step)
                if a.shape != b.shape:
                    continue
                if n in PER_SLOT and ok is not None and a.shape[0] == ok.size:
                    if not ok.any():
                        continue
                    a, b = a[ok], b[ok]
                row[n] = metrics(b, a)
        rows.append(row)
    return rows


def logit_metrics(ref, test, s):
    a = ref.logits[s].astype(np.float64)
    b = test.logits[s].astype(np.float64)
    m = metrics(b, a)
    ta = np.argsort(-a)[:10]
    tb = np.argsort(-b)[:10]
    m.update(top1_ref=int(ta[0]), top1_test=int(tb[0]),
             top5_ref=ta[:5].tolist(), top5_test=tb[:5].tolist(),
             top10_overlap=len(set(ta) & set(tb)),
             top5_overlap=len(set(ta[:5]) & set(tb[:5])),
             margin_ref=float(a[ta[0]] - a[ta[1]]),
             margin_test=float(b[tb[0]] - b[tb[1]]))
    return m


def fmt(m, k="rel_l2"):
    return "%.2e" % m[k] if m else "   -    "


if __name__ == "__main__":
    ref, test = Run(sys.argv[1]), Run(sys.argv[2])
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    print("argmax ref", ref.argmax.tolist(), "test", test.argmax.tolist(),
          "" if test.complete else "(test run incomplete: no logits)")
    r = router(ref, test, step)
    bad = [x for x in r if x["slot_mismatches"]]
    print("router: blocks %d  selections %d  slot mismatches %d  first bad block %s"
          % (len(r), sum(x["selections"] for x in r), sum(x["slot_mismatches"] for x in r),
             bad[0]["block"] if bad else None))
    names = ["ffn_moe_gate_in", "ffn_moe_gate", "ffn_moe_up", "ffn_moe_down_in",
             "ffn_moe_down", "ffn_moe_out", "ffn_out", "l_last"]
    print("%5s %6s " % ("blk", "idsOK") + " ".join("%10s" % n.replace("ffn_moe_", "")[:10] for n in names))
    for row in ladder(ref, test, step, names):
        print("%5d %3s/%-3s" % (row["block"], row["tokens_ids_agree"], row["tokens"])
              + " ".join("%10s" % fmt(row.get(n)) for n in names))
