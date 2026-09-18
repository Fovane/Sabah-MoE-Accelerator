"""Execute the frozen v1 validation plan (docs/V1_CORRECTNESS_CONTRACT.md, section 5).

Runs are executed once, in order. A run that has already produced a result
(pass or fail) is never repeated automatically. A single rerun of a run
invalidated by external interference requires `--rerun RUN --reason "..."`, and
is recorded in attempts.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from run_path import run as run_path  # noqa: E402

RAW = os.environ.get("SABAH_V1_RAW", "D:/sabah_v1val")
CORPUS = os.environ.get("SABAH_V1_CORPUS", os.path.join(V5, "tests", "v1_validation"))
# Rehearsal only (dev corpus): shorten every step count. The confirmatory run
# never sets this; evaluate.py records whether it was set.
SMOKE = os.environ.get("SABAH_V1_SMOKE") == "1"


def n(k):
    return min(k, 4) if SMOKE else k
SENTINELS = os.environ.get("SABAH_SENTINEL_DIR", "D:/sabah_rc4/sentinels")
CAP_NAMES = ("ffn_moe_topk,ffn_moe_weights_norm,ffn_moe_gate,ffn_moe_up,ffn_moe_down,ffn_moe_weighted")


def manifest():
    return json.load(open(os.path.join(CORPUS, "manifest.json"), encoding="utf-8"))


def sub_manifest(role, name):
    """A diag manifest with only the prompts of one role, in corpus order."""
    ids = manifest()["roles"][role]
    lines = [l for l in open(os.path.join(CORPUS, "prompts.jsonl"), encoding="utf-8") if l.strip()]
    keep = [l for l in lines if json.loads(l)["id"] in ids]
    assert len(keep) == len(ids), (role, ids)
    p = os.path.join(RAW, "manifests", name + ".jsonl")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8", newline="\n").write("".join(keep))
    return p


def plan():
    P = os.path.join(CORPUS, "prompts.jsonl")
    U = os.path.join(CORPUS, "ubatch.jsonl")
    F = os.path.join(RAW, "R1_B_free32")                     # forced tokens: <F>/<id>/fed.i32
    S = sub_manifest("sentinel", "sentinel")
    CA = sub_manifest("cache_invariance", "cache")
    DR = sub_manifest("determinism_repeat", "determinism")
    MS = sub_manifest("multiseq", "multiseq")
    cap = dict(names=CAP_NAMES, mmid=True, rows="sel")
    # A, B and C carry the SAME instrumentation, so the full-graph comparison
    # (gates H, I) is between identically observed graphs.
    tf_cap = dict(cap, steps="0,1,2,3" if SMOKE else "0,1,16,31")
    runs = [
        ("R1_B_free32", dict(path="cuda", manifest=P, n_predict=n(32), **tf_cap)),
        ("R2_A_tf32", dict(path="cpu", manifest=P, n_predict=n(32), force_dir=F, **tf_cap)),
        ("R3_C_tf32", dict(path="sabah", manifest=P, n_predict=n(32), force_dir=F, **tf_cap)),
    ]
    for k in (1, 2, 3):
        runs.append(("R7_S%d" % k, dict(path="sabah", manifest=S, n_predict=1, force_dir=F, steps="0",
                                         rtlib=os.path.join(SENTINELS, "sabah_rt_sentinel%d.dll" % k), **cap)))
    for tag, b in (("128M", 128 << 20), ("512M", 512 << 20), ("1G", 1 << 30)):
        runs.append(("R8_C_cache_" + tag, dict(path="sabah", manifest=CA, n_predict=n(8), force_dir=F, hot_bytes=b)))
    runs += [
        ("R9_C_repeat", dict(path="sabah", manifest=DR, n_predict=n(8), force_dir=F, **tf_cap)),
        ("R11_C_multiseq", dict(path="sabah", manifest=MS, n_predict=4, multiseq=True,
                                names=CAP_NAMES, mmid=True, rows="sel")),
        ("R12_S3_multiseq", dict(path="sabah", manifest=MS, n_predict=4, multiseq=True,
                                 names=CAP_NAMES, mmid=True, rows="sel",
                                 rtlib=os.path.join(SENTINELS, "sabah_rt_sentinel3.dll"))),
        ("R13_C_ubatch", dict(path="sabah", manifest=U, n_predict=2, steps="0,1", **cap)),
    ]
    return runs


def api_spec(name, backend, groups, stream=None, port=8095):
    m = manifest()
    text = {json.loads(l)["id"]: json.loads(l)["text"]
            for l in open(os.path.join(CORPUS, "prompts.jsonl"), encoding="utf-8") if l.strip()}
    spec = dict(backend=backend, port=port, backend_port=port + 10000, validate=backend == "sabah",
                groups=[dict(name=g, max_tokens=mt, prompts=[dict(id=i, text=text[i]) for i in m["roles"][role]])
                        for g, role, mt in groups])
    if stream:
        sid = m["roles"][stream][0]
        spec["stream"] = dict(id=sid, text=text[sid], max_tokens=n(8))
    p = os.path.join(RAW, "manifests", name + ".json")
    json.dump(spec, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return p


def api_runs():
    r15 = api_spec("R15_api", "sabah",
                   [("c1", "api_concurrency_1", n(8)), ("c2", "api_concurrency_2", n(8)),
                    ("c4", "api_concurrency_4", n(8)), ("stream_twin", "api_stream", n(8))],
                   stream="api_stream")
    r16a = api_spec("R16_api_ab_sabah", "sabah", [("ab", "api_ab", n(16))], port=8096)
    r16b = api_spec("R16_api_ab_reference", "reference", [("ab", "api_ab", n(16))], port=8097)
    return [("R15_api", r15), ("R16_api_ab_sabah", r16a), ("R16_api_ab_reference", r16b)]


def bench_runs():
    B = sub_manifest("benchmark", "benchmark")
    out = []
    for rep in (1, 2):
        out.append(("R17_bench_stock_r%d" % rep, dict(path="stock", manifest=B, n_predict=n(64))))
        out.append(("R17_bench_sabah_r%d" % rep, dict(path="sabah", manifest=B, n_predict=n(64),
                                                      selfcheck=0, verify=False)))
    return out


def log_attempt(name, status, reason=""):
    with open(os.path.join(RAW, "attempts.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(run=name, status=status, reason=reason, t=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")


def gpu_idle_check():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--rerun", default="")
    ap.add_argument("--reason", default="")
    a = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    build = None
    if not a.only and not os.path.exists(os.path.join(RAW, "R18_build.json")) and not SMOKE:
        log_attempt("R18_build", "start (parallel)")
        build = subprocess.Popen([sys.executable, os.path.join(HERE, "build_verify.py"),
                                  os.path.join(RAW, "R18_build.json")], cwd=V5,
                                 stdout=open(os.path.join(RAW, "R18_build.log"), "w"), stderr=subprocess.STDOUT)
    todo = plan() + [(n, dict(api=s)) for n, s in api_runs()] + bench_runs()
    for name, kw in todo:
        if name.startswith("R17") and build is not None and build.poll() is None:
            build.wait()                       # the benchmark runs on an otherwise idle machine
            log_attempt("R18_build", "end rc=%s" % build.returncode)
        if a.only and not any(name.startswith(x) for x in a.only.split(",")):
            continue
        out = os.path.join(RAW, name)
        done = os.path.exists(os.path.join(out, "path.json")) or os.path.exists(out + ".json")
        if done and name != a.rerun:
            continue
        if done and name == a.rerun:
            if not a.reason:
                sys.exit("a rerun needs --reason (external interference only)")
            log_attempt(name, "rerun", a.reason)
        busy = gpu_idle_check()
        log_attempt(name, "start", "gpu_apps_before=%r" % busy)
        t0 = time.time()
        if "api" in kw:
            rc = subprocess.run([sys.executable, os.path.join(HERE, "api_validate.py"), kw["api"], out + ".json"],
                                cwd=V5).returncode
        else:
            rc = run_path(out=out, **kw)["rc"]
        log_attempt(name, "end rc=%s wall=%.0fs" % (rc, time.time() - t0))
        print("%-24s rc=%s  %.0fs" % (name, rc, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
