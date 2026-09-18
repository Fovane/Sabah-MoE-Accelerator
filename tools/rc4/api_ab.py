"""Deterministic A/B through the OpenAI-compatible API: backend=reference vs backend=sabah.

Both backends are the same patched llama-server with identical graph and
placement; only the expert MUL_MAT_ID executor differs. Each request is
temperature 0 with a fixed seed and asks for per-token logprobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

V5 = "D:/sabah_scaling/v5"
MODEL = ("D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
         "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")
PROMPTS = [
    "Reply with exactly: SABAH_OK",
    "What is 17 times 23? Answer with the number only.",
    "Türkiye'nin başkenti neresidir? Tek kelimeyle cevap ver.",
]
MAX_TOKENS = 32
PORT, BPORT = 8091, 18091


def get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post(url, body, timeout=1800):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def run_backend(backend):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SABAH_", "GGML_OP_OFFLOAD"))}
    env["SABAH_LLAMA_TRACE"] = "1"
    log = open("D:/sabah_rc4/api_%s.log" % backend, "w", encoding="utf-8")
    cmd = [sys.executable, "-m", "sabah.server.openai_proxy", MODEL, "--port", str(PORT),
           "--backend-port", str(BPORT), "--context", "1024", "--backend", backend]
    if backend == "reference":
        cmd.append("--allow-reference")
    p = subprocess.Popen(cmd, cwd=V5, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 900
        health = None
        while time.time() < deadline:
            if p.poll() is not None:
                raise RuntimeError("server exited: see api_%s.log" % backend)
            try:
                health = get("http://127.0.0.1:%d/health" % PORT)
                break
            except OSError:
                time.sleep(2)
        out = dict(health=health, responses=[])
        for prompt in PROMPTS:
            t0 = time.time()
            r = post("http://127.0.0.1:%d/v1/chat/completions" % PORT, dict(
                messages=[{"role": "user", "content": prompt}], max_tokens=MAX_TOKENS,
                temperature=0, seed=1, logprobs=True, top_logprobs=5))
            ch = r["choices"][0]
            lp = (ch.get("logprobs") or {}).get("content") or []
            out["responses"].append(dict(
                prompt=prompt, wall_s=time.time() - t0,
                content=ch["message"].get("content"),
                reasoning=ch["message"].get("reasoning_content"),
                finish_reason=ch.get("finish_reason"),
                tokens=[t.get("token") for t in lp], logprobs=[t.get("logprob") for t in lp],
                top5=[[(c.get("token"), c.get("logprob")) for c in t.get("top_logprobs", [])] for t in lp],
                usage=r.get("usage"), timings=r.get("timings"),
                runtime_after=(time.sleep(1.0), get("http://127.0.0.1:%d/health" % PORT))[1].get("sabah_runtime")))
        return out
    finally:
        p.terminate()
        try:
            p.wait(30)
        except subprocess.TimeoutExpired:
            p.kill()
        log.close()
        # llama-server is a grandchild; make sure it is gone before the next backend starts
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(3)


if __name__ == "__main__":
    res = {b: run_backend(b) for b in ("reference", "sabah")}
    cmp = []
    for a, b in zip(res["reference"]["responses"], res["sabah"]["responses"]):
        n = min(len(a["tokens"]), len(b["tokens"]))
        first = next((i for i in range(n) if a["tokens"][i] != b["tokens"][i]), None)
        dlp = [abs(a["logprobs"][i] - b["logprobs"][i]) for i in range(first if first is not None else n)]
        cmp.append(dict(prompt=a["prompt"], tokens_ref=len(a["tokens"]), tokens_sabah=len(b["tokens"]),
                        identical_text=(a["content"], a["reasoning"]) == (b["content"], b["reasoning"]),
                        identical_tokens=first is None and len(a["tokens"]) == len(b["tokens"]),
                        first_divergence=first,
                        max_abs_logprob_diff_before_divergence=max(dlp) if dlp else None))
    res["comparison"] = cmp
    json.dump(res, open(sys.argv[1] if len(sys.argv) > 1 else "D:/sabah_rc4/evidence/api_ab.json", "w", encoding="utf-8"),
              indent=1, ensure_ascii=False)
    for c in cmp:
        print(json.dumps(c, ensure_ascii=False))
    print("health.backend:", res["reference"]["health"]["backend"], "|", res["sabah"]["health"]["backend"])
    for b in ("reference", "sabah"):
        print(b, "runtime after each request:", [x["runtime_after"] and {k: x["runtime_after"][k] for k in ("calls", "tokens", "lookups")} for x in res[b]["responses"]])
