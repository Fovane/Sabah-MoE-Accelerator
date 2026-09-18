"""v1 operation-level oracle over llama-sabah-diag captures.

For every captured token row of every captured (step, ubatch), every block and
every expert-projection family, two expert slots are selected by the frozen
rule of docs/V1_CORRECTNESS_CONTRACT.md:

    c      = int(sha256(f"{prompt}|{step}|{occ}|{row}|{op}")[:16], 16) % 10
    slots  = {(il + c) % 10, (il + c + 5) % 10}          (all slots with --all-slots)

The rule never looks at an error value. Because c does not depend on the
block, the 48 blocks of any (step, row, op) cover all ten slots.

Each selected output is recomputed in float64 from the path's own captured
input row (src[1], row slot mod ne11) and ids (src[2]) against gguf-py's
dequantization of the ORIGINAL GGUF bytes. Two exact structural checks run on
every captured row: ids at MUL_MAT_ID equal the router's ffn_moe_topk, and
ffn_moe_weighted equals ffn_moe_down times the router weight of the same slot.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, V5)
GGUF_PY = os.environ.get("SABAH_GGUF_PY", "D:/sabah_scaling/llama-sabah-clean/gguf-py")
sys.path.insert(0, GGUF_PY)

from gguf import quants  # noqa: E402
from gguf.constants import GGMLQuantizationType as Q  # noqa: E402
from sabah.core.model_inspector import inspect_model  # noqa: E402
from sabah.runtime import rt  # noqa: E402
from sabah.runtime.expert_bank import ExpertBank  # noqa: E402

MODEL = os.environ.get("SABAH_MODEL", "D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                                      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
OPS = ("gate", "up", "down")
N_BLOCKS, TOPK = 48, 10


class Capture:
    def __init__(self, cap_dir):
        self.dir = cap_dir
        self.rec = {}
        for line in open(os.path.join(cap_dir, "index.jsonl"), encoding="utf-8"):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            self.rec.setdefault((r["key"], r["step"], r["occ"]), r)
        self.steps = sorted({k[1] for k in self.rec})

    def has(self, key, step, occ):
        return (key, step, occ) in self.rec

    def get(self, key, step, occ):
        """Array shaped [kept_token_rows, ...inner dims], plus the kept row ids."""
        r = self.rec[(key, step, occ)]
        dt = np.float32 if r["type"] == "f32" else np.int32
        with open(os.path.join(self.dir, key + ".bin"), "rb") as f:
            f.seek(r["offset"])
            a = np.frombuffer(f.read(r["nbytes"]), dt)
        ne = list(r["ne"])
        d = r["token_dim"]
        rows = r["rows"]
        if d is not None and d > 0:
            ne[d] = len(rows)
        a = a.reshape(ne[3], ne[2], ne[1], ne[0])[0]            # [ne2, ne1, ne0]
        if d == 1:                                              # [ne1=rows, ne0] -> rows first
            a = a[0] if a.ndim == 3 and a.shape[0] == 1 else a
        return a, rows, r

    def occs(self, key, step, nonempty=True):
        """Ubatch occurrences of a key in a step. Zero-row nodes (the last
        block of an ubatch without output rows exists in the graph with no
        rows and is never dispatched) are skipped unless nonempty=False."""
        return sorted(o for (k, s, o), r in self.rec.items()
                      if k == key and s == step and (not nonempty or r["nbytes"] > 0))


_bank = None
_W = OrderedDict()


def W(block, role, e, rows, cols):
    global _bank
    if _bank is None:
        _bank = ExpertBank(inspect_model(MODEL), mode="mmap")
    key = (block, role, e)
    if key in _W:
        _W.move_to_end(key)
        return _W[key]
    qt = _bank.desc[(block, role)]["qtype"]
    raw = np.asarray(_bank.slice(block, e, role)).reshape(rows, rt.row_bytes(rt.QTYPE[qt], cols))
    m = np.asarray(quants.dequantize(raw, getattr(Q, qt)), np.float64).reshape(rows, cols)
    _W[key] = m
    if len(_W) > 600:
        _W.popitem(last=False)
    return m


def slot_offset(prompt, step, occ, row, op):
    h = hashlib.sha256(("%s|%d|%d|%d|%s" % (prompt, step, occ, row, op)).encode()).hexdigest()
    return int(h[:16], 16) % 10


def rel_l2(a, b):
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(a - b) / nb) if nb > 0 else float(np.linalg.norm(a - b))


def check(cap_dir, prompt_id, all_slots=False, sentinel_label=None):
    cap = Capture(cap_dir)
    samples = []
    structural = dict(ids_rows=0, ids_mismatch=0, weighted_rows=0, weighted_mismatch=0)
    cover_blocks = {op: set() for op in OPS}
    cover_slots = {op: set() for op in OPS}
    missing = []
    for step in cap.steps:
        for il in range(N_BLOCKS):
            for op in OPS:
                name = "ffn_moe_%s" % op
                k_out, k_in, k_ids = "%s_L%d" % (name, il), "%s_in_L%d" % (name, il), "%s_ids_L%d" % (name, il)
                occs = cap.occs(k_out, step)
                if not occs:
                    missing.append((step, il, op))
                    continue
                for occ in occs:
                    out, rows, _ = cap.get(k_out, step, occ)          # [R, K, rows_out]
                    xin, rows_in, _ = cap.get(k_in, step, occ)        # [R, ne11, k]
                    ids, rows_ids, _ = cap.get(k_ids, step, occ)      # [R, K]
                    out = out.reshape(len(rows), TOPK, -1)
                    xin = xin.reshape(len(rows_in), -1, xin.shape[-1])
                    ids = ids.reshape(len(rows_ids), TOPK)
                    assert rows == rows_in == rows_ids, (k_out, step, occ)
                    ne11 = xin.shape[1]
                    # structural: ids at MUL_MAT_ID == router top-k (exact)
                    k_top = "ffn_moe_topk_L%d" % il
                    if op == "gate" and cap.has(k_top, step, occ):
                        top, rows_top, _ = cap.get(k_top, step, occ)
                        top = top.reshape(len(rows_top), TOPK)
                        if rows_top == rows:
                            structural["ids_rows"] += len(rows)
                            structural["ids_mismatch"] += int((top != ids).any(axis=1).sum())
                    # structural: weighted == down * weight (same slot)
                    k_wt, k_w = "ffn_moe_weighted_L%d" % il, "ffn_moe_weights_norm_L%d" % il
                    if op == "down" and cap.has(k_wt, step, occ) and cap.has(k_w, step, occ):
                        wt, rows_wt, _ = cap.get(k_wt, step, occ)
                        wn, rows_w, _ = cap.get(k_w, step, occ)
                        if rows_wt == rows == rows_w:
                            wt = wt.reshape(len(rows), TOPK, -1)
                            wn = wn.reshape(len(rows), TOPK, 1)
                            prod = (out * wn).astype(np.float32)
                            structural["weighted_rows"] += len(rows)
                            structural["weighted_mismatch"] += int((prod != wt).any(axis=(1, 2)).sum())
                    for ri, row in enumerate(rows):
                        if all_slots:
                            slots = list(range(TOPK))
                        else:
                            c = slot_offset(prompt_id, step, occ, row, op)
                            slots = [(il + c) % TOPK, (il + c + 5) % TOPK]
                        for s in slots:
                            e = int(ids[ri, s])
                            x = xin[ri, s % ne11].astype(np.float64)
                            w = W(il, op, e, out.shape[-1], x.size)
                            ref = w @ x
                            samples.append(dict(step=step, occ=occ, row=int(row), block=il, op=op, slot=s,
                                                expert=e, rel_l2=rel_l2(out[ri, s].astype(np.float64), ref)))
                            cover_blocks[op].add(il)
                            cover_slots[op].add(s)
    rels = np.array([s["rel_l2"] for s in samples]) if samples else np.array([np.inf])
    return dict(prompt=prompt_id, cap_dir=cap_dir, steps=cap.steps, n_samples=len(samples),
                max_rel_l2=float(rels.max()), median_rel_l2=float(np.median(rels)),
                n_over_2e6=int((rels > 2e-6).sum()),
                blocks_covered={op: len(v) for op, v in cover_blocks.items()},
                slots_covered={op: sorted(v) for op, v in cover_slots.items()},
                missing_ops=missing[:20], n_missing_ops=len(missing),
                structural=structural,
                worst=sorted(samples, key=lambda s: -s["rel_l2"])[:5])


if __name__ == "__main__":
    r = check(sys.argv[1], sys.argv[2], all_slots=len(sys.argv) > 3 and sys.argv[3] == "all")
    print(json.dumps({k: v for k, v in r.items() if k != "worst"}, indent=1))
    print("worst:", r["worst"][:2])
