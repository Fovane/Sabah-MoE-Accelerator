"""Evaluate the frozen v1 gates (docs/V1_CORRECTNESS_CONTRACT.md) from the raw runs.

Writes machine-readable evidence to results/v1_validation/ and prints the gate
table. Every rule here is the contract's rule; nothing is tuned to the data.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from opcheck_v1 import check as opcheck, Capture  # noqa: E402

RAW = os.environ.get("SABAH_V1_RAW", "D:/sabah_v1val")
OUT = os.path.join(V5, "results", "v1_validation")
CORPUS = os.environ.get("SABAH_V1_CORPUS", os.path.join(V5, "tests", "v1_validation"))
SMOKE = os.environ.get("SABAH_V1_SMOKE") == "1"
TF_STEPS = 4 if SMOKE else 32
SHORT_STEPS = 4 if SMOKE else 8
if SMOKE:
    OUT = os.path.join(RAW, "_evaluation")      # rehearsal output never lands in the repo
UB = 512
OP_TOL = 2e-6
X_MARGIN, DELTA_MARGIN = 1.25, 0.03
BOOT_N, BOOT_SEED, BOOT_Q = 20000, 20260919, 95

os.makedirs(OUT, exist_ok=True)
ROLES = json.load(open(os.path.join(CORPUS, "manifest.json"), encoding="utf-8"))["roles"]


def jload(p, default=None):
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def dump(name, obj):
    with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=float)


def run_dir(name):
    return os.path.join(RAW, name)


def path_rec(name):
    return jload(os.path.join(run_dir(name), "path.json"))


def ids_of(manifest):
    return [json.loads(l)["id"] for l in open(manifest, encoding="utf-8") if l.strip()]


def run_ids(name):
    rec = path_rec(name)
    return ids_of(rec["manifest"]) if rec else []


def n_prompt(name, pid):
    return int(np.fromfile(os.path.join(run_dir(name), pid, "prompt_tokens.i32"), np.int32).size)


def argmax(name, pid):
    return np.fromfile(os.path.join(run_dir(name), pid, "argmax.i32"), np.int32)


def logits(name, pid):
    run = jload(os.path.join(run_dir(name), pid, "run.json"))
    return np.fromfile(os.path.join(run_dir(name), pid, "logits.f32"), np.float32).reshape(-1, run["n_vocab"])


def expected_seq(n_tok, n_steps):
    """Exact coverage for one single-sequence prompt (contract 6.1)."""
    calls = tokens = 0
    rem = n_tok
    while rem > 0:
        n = min(UB, rem)
        rem -= n
        o = 1 if rem == 0 else 0
        calls += 144 if o else 141
        tokens += 141 * n + 3 * o
    calls += 144 * (n_steps - 1)
    tokens += 144 * (n_steps - 1)
    return calls, tokens


def status_delta(name, pid):
    s0 = jload(os.path.join(run_dir(name), pid, "sabah_status_start.json"))
    s1 = jload(os.path.join(run_dir(name), pid, "sabah_status_end.json"))
    if s1 is None:
        return None
    s0 = s0 or {}
    return {k: v - s0.get(k, 0) for k, v in s1.items() if isinstance(v, (int, float))}


# ---------------------------------------------------------------- gate B
SABAH_RUNS = dict(  # run -> (expected sentinel, checks_on)
    R3_C_tf32=(0, True),
    R7_S1=(1, True), R7_S2=(2, True), R7_S3=(3, True),
    R8_C_cache_128M=(0, True), R8_C_cache_512M=(0, True), R8_C_cache_1G=(0, True),
    R9_C_repeat=(0, True), R11_C_multiseq=(0, True), R12_S3_multiseq=(3, True),
    R13_C_ubatch=(0, True), R17_bench_sabah_r1=(0, False), R17_bench_sabah_r2=(0, False))


def gate_b():
    rows = []
    for name, (sent, checks) in SABAH_RUNS.items():
        rec = path_rec(name)
        r = dict(run=name, present=rec is not None)
        if rec is None:
            rows.append(dict(r, ok=False, reason="missing run"))
            continue
        snap = rec.get("snapshot_at_exit") or {}
        r.update(rc=rec["rc"], snapshot=snap)
        ok = rec["rc"] == 0 and bool(snap)
        problems = []
        ids = run_ids(name)
        if rec.get("multiseq"):
            ms = os.path.join(run_dir(name), "_multiseq")
            steps = {pid: len(argmax(name, pid)) for pid in ids}
            rows_all = sum(n_prompt(name, pid) for pid in ids) + sum(s - 1 for s in steps.values())
            outs = len(ids) + sum(s - 1 for s in steps.values())
            exp_tokens = 141 * rows_all + 3 * outs
            r.update(expected_tokens=exp_tokens, calls_checked=False)
            if snap.get("tokens") != exp_tokens:
                problems.append("tokens %s != %s" % (snap.get("tokens"), exp_tokens))
        else:
            per = []
            tc = tt = 0
            for pid in ids:
                c, t = expected_seq(n_prompt(name, pid), len(argmax(name, pid)))
                d = status_delta(name, pid)
                per.append(dict(prompt=pid, expected_calls=c, expected_tokens=t,
                                observed_calls=d and d.get("calls"), observed_tokens=d and d.get("tokens"),
                                selfcheck_fail=d and d.get("selfcheck_fail"),
                                selfcheck_ok=d and d.get("selfcheck_ok")))
                tc += c
                tt += t
                if d is None or d.get("calls") != c or d.get("tokens") != t:
                    problems.append("coverage %s" % pid)
            r.update(per_prompt=per, expected_calls=tc, expected_tokens=tt)
            if snap.get("calls") != tc or snap.get("tokens") != tt:
                problems.append("run totals %s/%s != %s/%s" % (snap.get("calls"), snap.get("tokens"), tc, tt))
        if snap.get("lookups") != 10 * snap.get("tokens", -1):
            problems.append("lookups != 10 x tokens")
        if snap.get("empty_calls", 0) != 0:
            problems.append("empty_calls")
        if snap.get("sentinel") != sent:
            problems.append("sentinel %s != %s" % (snap.get("sentinel"), sent))
        if checks:
            if snap.get("verify_fail", 1) != 0 or snap.get("verify_ok", 0) <= 0:
                problems.append("byte verification")
            if sent == 0 and (snap.get("selfcheck_fail", 1) != 0 or snap.get("selfcheck_ok", 0) <= 0):
                problems.append("selfcheck")
        r.update(problems=problems, ok=ok and not problems)
        rows.append(r)
    return rows


# ---------------------------------------------------------------- gate C (+ structural from captures)
def cap_dir(name, pid):
    return os.path.join(run_dir(name), pid, "cap")


def gate_c():
    res = []
    for pid in run_ids("R3_C_tf32"):
        c = opcheck(cap_dir("R3_C_tf32", pid), pid)
        ok = (c["n_samples"] > 0 and c["n_over_2e6"] == 0 and c["n_missing_ops"] == 0 and
              all(v == 48 for v in c["blocks_covered"].values()) and
              all(len(v) == 10 for v in c["slots_covered"].values()) and
              c["structural"]["ids_mismatch"] == 0 and c["structural"]["weighted_mismatch"] == 0 and
              c["structural"]["ids_rows"] > 0 and c["structural"]["weighted_rows"] > 0)
        res.append(dict(c, ok=ok))
    return res


# ---------------------------------------------------------------- gate D
def gate_d():
    trials = []
    b_arg = {pid: argmax("R1_B_free32", pid) for pid in run_ids("R1_B_free32")}
    for k in (1, 2, 3):
        name = "R7_S%d" % k
        for pid in run_ids(name):
            c = opcheck(cap_dir(name, pid), pid)
            d = status_delta(name, pid) or {}
            a = argmax(name, pid)
            trials.append(dict(sentinel=k, prompt=pid, opcheck_violations=c["n_over_2e6"],
                               opcheck_samples=c["n_samples"], opcheck_max=c["max_rel_l2"],
                               selfcheck_fail=d.get("selfcheck_fail"), selfcheck_max=d.get("selfcheck_max_rel_l2"),
                               greedy_agree_steps=int((a == b_arg[pid][:len(a)]).sum()), greedy_steps=int(len(a)),
                               detected_opcheck=c["n_over_2e6"] > 0, detected_selfcheck=(d.get("selfcheck_fail") or 0) > 0))
    ms = opcheck(os.path.join(run_dir("R12_S3_multiseq"), "_multiseq", "cap"), "multiseq", all_slots=True)
    detected_ms = ms["n_over_2e6"] > 0
    ok = len(trials) == 3 * len(ROLES["sentinel"]) and all(t["detected_opcheck"] and t["detected_selfcheck"] for t in trials) and detected_ms
    return dict(trials=trials, multiseq_s3=dict(violations=ms["n_over_2e6"], samples=ms["n_samples"],
                                                max=ms["max_rel_l2"], detected=detected_ms), ok=ok)


# ---------------------------------------------------------------- gate E
def gate_e():
    out = dict(cache=[], repeat=[])
    ok = True
    for pid in run_ids("R8_C_cache_1G"):
        ref = logits("R8_C_cache_1G", pid)
        row = dict(prompt=pid)
        for tag in ("128M", "512M"):
            other = logits("R8_C_cache_" + tag, pid)
            same = ref.shape == other.shape and bool(np.array_equal(ref, other))
            row[tag + "_bitwise_equal_1G"] = same
            ok &= same
        row["steps"] = int(ref.shape[0])
        out["cache"].append(row)
    for pid in run_ids("R9_C_repeat"):
        a, b = logits("R9_C_repeat", pid), logits("R3_C_tf32", pid)[:SHORT_STEPS]
        same = a.shape == b.shape and bool(np.array_equal(a, b))
        out["repeat"].append(dict(prompt=pid, bitwise_equal_R3_first_steps=same, steps=SHORT_STEPS))
        ok &= same
    out["ok"] = bool(ok and out["cache"] and out["repeat"])
    return out


# ---------------------------------------------------------------- gate F1
def gate_f1():
    name = "R11_C_multiseq"
    capd = os.path.join(run_dir(name), "_multiseq", "cap")
    c = opcheck(capd, "multiseq", all_slots=True)
    cap = Capture(capd)
    shared = []
    for step in cap.steps:
        if step == 0:
            continue
        occs = cap.occs("ffn_moe_down_L0", step)
        r = cap.rec[("ffn_moe_down_L0", step, occs[0])] if occs else None
        shared.append(dict(step=step, ubatches=len(occs), rows=r["ne"][2] if r else None))
    shared_ok = bool(shared) and all(s["ubatches"] == 1 and s["rows"] == len(ROLES["multiseq"]) for s in shared)
    ok = (c["n_over_2e6"] == 0 and c["n_samples"] > 0 and c["n_missing_ops"] == 0 and
          all(v == 48 for v in c["blocks_covered"].values()) and all(len(v) == 10 for v in c["slots_covered"].values())
          and c["structural"]["ids_mismatch"] == 0 and c["structural"]["weighted_mismatch"] == 0 and shared_ok)
    return dict(opcheck={k: v for k, v in c.items() if k != "worst"}, shared_decode_ubatches=shared, ok=ok)


# ---------------------------------------------------------------- gate G
def gate_g():
    res = []
    name = "R13_C_ubatch"
    for pid in run_ids(name):
        n = n_prompt(name, pid)
        U = math.ceil(n / UB)
        c = opcheck(cap_dir(name, pid), pid)
        cap = Capture(cap_dir(name, pid))
        # blocks 0-46: every ubatch; block 47: rows only in the final ubatch
        occ_ok = all(set(cap.occs("ffn_moe_%s_L%d" % (op, il), 0)) == set(range(U))
                     for il in range(47) for op in ("gate", "up", "down"))
        occ_ok &= all(set(cap.occs("ffn_moe_%s_L47" % op, 0)) == {U - 1} for op in ("gate", "up", "down"))
        ok = (c["n_over_2e6"] == 0 and c["n_samples"] > 0 and c["n_missing_ops"] == 0 and occ_ok and
              c["structural"]["ids_mismatch"] == 0 and c["structural"]["weighted_mismatch"] == 0)
        res.append(dict(prompt=pid, n_tokens=n, ubatches=U, all_ubatches_captured=occ_ok,
                        opcheck={k: v for k, v in c.items() if k != "worst"}, ok=ok))
    return res


# ---------------------------------------------------------------- gates H and I
def rel_l2_rows(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)


def gates_h_i():
    ids = run_ids("R1_B_free32")
    per = []
    for pid in ids:
        LB, LA, LC = logits("R1_B_free32", pid), logits("R2_A_tf32", pid), logits("R3_C_tf32", pid)
        assert LB.shape == LA.shape == LC.shape == (TF_STEPS, LB.shape[1]), (pid, LB.shape, LA.shape, LC.shape)
        fed_b = np.fromfile(os.path.join(run_dir("R1_B_free32"), pid, "fed.i32"), np.int32)
        for n in ("R2_A_tf32", "R3_C_tf32"):
            fed = np.fromfile(os.path.join(run_dir(n), pid, "fed.i32"), np.int32)
            assert np.array_equal(fed[:TF_STEPS - 1], fed_b[:TF_STEPS - 1]), ("context mismatch", n, pid)
        eA, eC = rel_l2_rows(LA, LB), rel_l2_rows(LC, LB)
        aB, aA, aC = LB.argmax(1), LA.argmax(1), LC.argmax(1)
        srt = np.sort(LB, axis=1)
        margin = srt[:, -1] - srt[:, -2]
        per.append(dict(prompt=pid, mA=float(eA.mean()), mC=float(eC.mean()),
                        flipsA=int((aA != aB).sum()), flipsC=int((aC != aB).sum()),
                        flipA_margins=[float(margin[s]) for s in np.where(aA != aB)[0]],
                        flipC_margins=[float(margin[s]) for s in np.where(aC != aB)[0]],
                        eA_median=float(np.median(eA)), eC_median=float(np.median(eC))))
    mA = np.array([p["mA"] for p in per])
    mC = np.array([p["mC"] for p in per])
    fA = np.array([p["flipsA"] for p in per])
    fC = np.array([p["flipsC"] for p in per])
    n = len(per)
    R = mC.sum() / mA.sum()
    D = (fC.sum() - fA.sum()) / (n * TF_STEPS)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(BOOT_N, n))
    Rb = mC[idx].sum(1) / mA[idx].sum(1)
    Db = (fC[idx].sum(1) - fA[idx].sum(1)) / (n * TF_STEPS)
    ubR = float(np.percentile(Rb, BOOT_Q))
    ubD = float(np.percentile(Db, BOOT_Q))
    H = dict(statistic="R = sum_p mean_s e_C / sum_p mean_s e_A", R=float(R), upper_95=ubR, margin=X_MARGIN,
             clusters=n, observations_per_path=n * TF_STEPS, bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, percentile=BOOT_Q),
             ok=ubR <= X_MARGIN)
    I = dict(statistic="D = (sum flips_C - sum flips_A) / (n*64)", D=float(D), upper_95=ubD, margin=DELTA_MARGIN,
             flips_A=int(fA.sum()), flips_C=int(fC.sum()), clusters=n, observations_per_path=n * TF_STEPS,
             bootstrap=dict(n=BOOT_N, seed=BOOT_SEED, percentile=BOOT_Q), ok=ubD <= DELTA_MARGIN)
    return H, I, per


def free_running():
    """Free-running greedy agreement, derived EXACTLY from the teacher-forced
    runs: until the first disagreement the free-running and teacher-forced
    contexts are identical, so the first divergence of a free-running run is
    the first step whose argmax differs from native CUDA's."""
    out = []
    for pid in run_ids("R1_B_free32"):
        b = argmax("R1_B_free32", pid)
        row = dict(prompt=pid)
        for tag, name in (("cpu", "R2_A_tf32"), ("sabah", "R3_C_tf32")):
            a = argmax(name, pid)
            diff = np.where(a != b[:len(a)])[0]
            first = int(diff[0]) if len(diff) else None
            row[tag] = dict(first_divergence=first,
                            identical_16=first is None or first >= 16,
                            identical_32=first is None)
        out.append(row)
    summary = {tag: dict(identical_16=sum(r[tag]["identical_16"] for r in out),
                         identical_32=sum(r[tag]["identical_32"] for r in out), prompts=len(out))
               for tag in ("cpu", "sabah")}
    return dict(per_prompt=out, summary=summary, method="derived from teacher-forced runs (exact up to first divergence)")


# ---------------------------------------------------------------- API (F2, J, K)
def gate_api():
    r = jload(run_dir("R15_api") + ".json")
    if r is None:
        return dict(ok_f2=False, ok_j=False, ok_k=False, reason="missing R15")
    groups = {g["name"]: g for g in r["groups"]}
    f2 = []
    for name in ("c1", "c2", "c4", "stream_twin"):
        g = groups.get(name)
        a = (g or {}).get("accounting") or {}
        ok = bool(g) and all(x.get("ok") for x in g["responses"]) and a.get("exact") and a.get("lookups_is_10x") \
            and (a.get("selfcheck_ok") or 0) > 0 and a.get("selfcheck_fail") == 0 and a.get("verify_fail") == 0
        f2.append(dict(group=name, n=g and g["n"], accounting=a, ok=bool(ok)))
    s = r.get("stream") or {}
    sa = s.get("accounting") or {}
    stream_ok = (s.get("chunks", 0) > 1 and s.get("text_equals_twin") and sa.get("exact") and sa.get("lookups_is_10x")
                 and sa.get("selfcheck_fail") == 0 and sa.get("verify_fail") == 0)
    h = r.get("health_start") or {}
    ident_ok = h.get("backend") == "llama.cpp+sabah" and h.get("execution") == "SABAH_NATIVE_MUL_MAT_ID"
    f2_ok = all(x["ok"] for x in f2 if x["group"] in ("c1", "c2", "c4"))
    return dict(f2=f2, stream=dict(chunks=s.get("chunks"), text_equals_twin=s.get("text_equals_twin"),
                                   accounting=sa, ok=bool(stream_ok)),
                identity=dict(backend=h.get("backend"), execution=h.get("execution"), ok=ident_ok),
                ok_f2=f2_ok, ok_j=bool(f2_ok and stream_ok and ident_ok),
                ok_k=bool(all(x["ok"] for x in f2) and stream_ok))


def api_ab():
    a = jload(run_dir("R16_api_ab_sabah") + ".json")
    b = jload(run_dir("R16_api_ab_reference") + ".json")
    if not a or not b:
        return None
    rows = []
    for x, y in zip(b["groups"][0]["responses"], a["groups"][0]["responses"]):
        tx, ty = x.get("token_ids") or [], y.get("token_ids") or []
        n = min(len(tx), len(ty))
        first = next((i for i in range(n) if tx[i] != ty[i]), None)
        margin = None
        if first is not None and x.get("top2") and len(x["top2"][first]) >= 2:
            margin = x["top2"][first][0][1] - x["top2"][first][1][1]
        rows.append(dict(prompt=x["id"], reference_tokens=len(tx), sabah_tokens=len(ty),
                         identical=first is None and len(tx) == len(ty), first_divergence=first,
                         reference_logprob_margin_at_divergence=margin))
    acc = (a["groups"][0].get("accounting") or {})
    return dict(per_prompt=rows, sabah_accounting=acc, sabah_health=a.get("health_start", {}).get("backend"),
                reference_health=b.get("health_start", {}).get("backend"))


# ---------------------------------------------------------------- benchmark
def benchmark():
    rows = {}
    for d in sorted(glob.glob(os.path.join(RAW, "R17_bench_*"))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        rec = path_rec(name)
        per = []
        for pid in run_ids(name):
            ms = np.fromfile(os.path.join(d, pid, "step_ms.f64"), np.float64)
            dec = ms[1:]
            per.append(dict(prompt=pid, prefill_ms=float(ms[0]), prompt_tokens=n_prompt(name, pid),
                            decode_tok_s=float(len(dec) / (dec.sum() / 1000.0)),
                            decode_ms_median=float(np.median(dec)), decode_ms_p95=float(np.percentile(dec, 95))))
        dec_total = sum(len(np.fromfile(os.path.join(d, p["prompt"], "step_ms.f64"), np.float64)) - 1 for p in per)
        dec_ms = sum(np.fromfile(os.path.join(d, p["prompt"], "step_ms.f64"), np.float64)[1:].sum() for p in per)
        pre_tok = sum(p["prompt_tokens"] for p in per)
        pre_ms = sum(p["prefill_ms"] for p in per)
        rows[name] = dict(path=rec["path"], rc=rec["rc"], per_prompt=per,
                          decode_tok_s=dec_total / (dec_ms / 1000.0), prefill_tok_s=pre_tok / (pre_ms / 1000.0),
                          wall_s=rec["wall_s"], sabah=rec.get("snapshot_at_exit"))
    st = [v["decode_tok_s"] for v in rows.values() if v["path"] == "stock"]
    sb = [v["decode_tok_s"] for v in rows.values() if v["path"] == "sabah"]
    out = dict(runs=rows)
    if st and sb:
        out["stock_decode_tok_s_mean"] = float(np.mean(st))
        out["sabah_decode_tok_s_mean"] = float(np.mean(sb))
        out["MEASURED_SPEEDUP_decode"] = float(np.mean(sb) / np.mean(st))
        snaps = [v["sabah"] for v in rows.values() if v["path"] == "sabah" and v["sabah"]]
        if snaps:
            s = snaps[0]
            out["sabah_hit_rate"] = s["hits"] / max(1, s["hits"] + s["misses"])
            out["sabah_bytes_fetched"] = s["bytes_fetched"]
    out["available"] = len(rows) == 4 and all(v["rc"] == 0 for v in rows.values())
    return out


def main():
    gb = gate_b()
    dump("structural_coverage.json", gb)
    gc = gate_c()
    dump("op_level.json", gc)
    gd = gate_d()
    dump("sentinel.json", gd)
    ge = gate_e()
    dump("cache_determinism.json", ge)
    gf1 = gate_f1()
    dump("multiseq.json", gf1)
    gg = gate_g()
    dump("ubatch.json", gg)
    H, I, per = gates_h_i()
    dump("full_graph_statistics.json", dict(H=H, I=I, per_prompt=per))
    fr = free_running()
    dump("greedy_free_running.json", fr)
    api = gate_api()
    dump("api.json", api)
    ab = api_ab()
    dump("api_ab.json", ab)
    bench = benchmark()
    dump("benchmark.json", bench)
    ga = jload(os.path.join(RAW, "R18_build.json"), {}) or {}
    gm = jload(os.path.join(OUT, "claims_check.json"), {}) or {}
    gates = [
        ("A", "Build / reproducibility", bool(ga.get("ok")), ga.get("summary")),
        ("B", "Structural exactness", all(r["ok"] for r in gb) and all(c["ok"] for c in gc),
         "%d runs; captured ids/weights" % len(gb)),
        ("C", "Float64 op-level <= 2e-6", all(c["ok"] for c in gc) and all(
            r["ok"] for r in gb if SABAH_RUNS[r["run"]][0] == 0),
         "max %.3e over %d samples" % (max(c["max_rel_l2"] for c in gc) if gc else float("nan"),
                                       sum(c["n_samples"] for c in gc))),
        ("D", "Wrong-expert discrimination", gd["ok"],
         "%d/%d trials both detectors; multiseq S3 %s" % (sum(t["detected_opcheck"] and t["detected_selfcheck"]
                                                              for t in gd["trials"]), 3 * len(ROLES["sentinel"]),
                                                          gd["multiseq_s3"]["detected"])),
        ("E", "Cache / residency invariance + determinism", ge["ok"], ""),
        ("F", "Multi-sequence / concurrency", gf1["ok"] and api["ok_f2"], ""),
        ("G", ">512 / ubatch boundary", bool(gg) and all(g["ok"] for g in gg), ""),
        ("H", "Full-graph non-inferiority", H["ok"], "R=%.4f UB95=%.4f <= %.2f" % (H["R"], H["upper_95"], X_MARGIN)),
        ("I", "Greedy flip non-inferiority", I["ok"], "D=%.4f UB95=%.4f <= %.2f" % (I["D"], I["upper_95"], DELTA_MARGIN)),
        ("J", "Real Sabah API backend", api["ok_j"], ""),
        ("K", "Evidence-grade counters", api["ok_k"] and all(r["ok"] for r in gb), ""),
        ("L", "Measured benchmark", bench.get("available", False),
         "speedup %.4f" % bench["MEASURED_SPEEDUP_decode"] if "MEASURED_SPEEDUP_decode" in bench else ""),
        ("M", "Claim hygiene", bool(gm.get("ok")), gm.get("summary")),
    ]
    table = [dict(gate=g, name=n, result=("AVAILABLE" if g == "L" and ok else "PASS" if ok else "FAIL"), evidence=e)
             for g, n, ok, e in gates]
    release = all(t["result"] in ("PASS", "AVAILABLE") for t in table)
    dump("release_gates.json", dict(gates=table, all_pass=release, rehearsal=SMOKE, corpus=CORPUS, raw=RAW))
    for t in table:
        print("%s  %-44s %-9s %s" % (t["gate"], t["name"], t["result"], t["evidence"] or ""))
    print("ALL GATES PASS" if release else "NOT ALL GATES PASS")


if __name__ == "__main__":
    main()
