"""Validate the user-facing API: `sabah serve --backend sabah --validate`.

For each request group (1, 2 or 4 simultaneous requests) the runtime's exact
counters are read from /health before and after, and checked against the graph:

  d_tokens  == 141 * sum(new prompt tokens) + 3 * n_requests + 144 * sum(decoded tokens)
  d_lookups == 10 * d_tokens
  d_selfcheck_ok > 0, d_selfcheck_fail == 0, d_verify_fail == 0

new prompt tokens = prompt_tokens - cached_tokens (usage); decoded tokens =
completion_tokens - 1 (the last sampled token is never fed back). The factor
141/144 follows from qwen4exp: 48 blocks x 3 expert ops, and the last block
runs only on output rows.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL = os.environ.get("SABAH_MODEL", "D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                                      "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def post(url, body, timeout=3600):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def post_stream(url, body, timeout=3600):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
    chunks, reasoning, content = 0, [], []
    usage = None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            ev = json.loads(data)
            chunks += 1
            for ch in ev.get("choices", []):
                d = ch.get("delta") or {}
                reasoning.append(d.get("reasoning_content") or "")
                content.append(d.get("content") or "")
            usage = ev.get("usage") or usage
    return chunks, "".join(reasoning), "".join(content), usage


class Server:
    def __init__(self, backend, port, bport, log, validate=True, extra=()):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("SABAH_", "GGML_OP_OFFLOAD"))}
        cmd = [sys.executable, "-m", "sabah.server.openai_proxy", MODEL, "--port", str(port),
               "--backend-port", str(bport), "--context", "4096", "--parallel", "4", "--backend", backend]
        if backend == "reference":
            cmd.append("--allow-reference")
        elif validate:
            cmd.append("--validate")
        cmd += list(extra)
        self.port = port
        self.log = open(log, "w", encoding="utf-8")
        self.p = subprocess.Popen(cmd, cwd=V5, env=env, stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.time() + 1200
        while time.time() < deadline:
            if self.p.poll() is not None:
                raise RuntimeError("server exited, see %s" % log)
            try:
                self.health()
                return
            except OSError:
                time.sleep(2)
        raise TimeoutError("server did not start")

    def base(self):
        return "http://127.0.0.1:%d" % self.port

    def health(self):
        return get(self.base() + "/health")

    def stop(self):
        self.p.terminate()
        try:
            self.p.wait(30)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.log.close()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(3)


def chat(server, prompt, max_tokens, results, key):
    t0 = time.time()
    try:
        r = post(server.base() + "/v1/chat/completions", dict(
            messages=[{"role": "user", "content": prompt}], max_tokens=max_tokens,
            temperature=0, seed=1, logprobs=True, top_logprobs=2))
        ch = r["choices"][0]
        lp = (ch.get("logprobs") or {}).get("content") or []
        results[key] = dict(ok=True, wall_s=time.time() - t0, usage=r.get("usage"),
                            content=ch["message"].get("content"),
                            reasoning=ch["message"].get("reasoning_content"),
                            finish_reason=ch.get("finish_reason"),
                            token_ids=[t.get("id") for t in lp], tokens=[t.get("token") for t in lp],
                            logprobs=[t.get("logprob") for t in lp],
                            top2=[[(c.get("id"), c.get("logprob")) for c in t.get("top_logprobs", [])] for t in lp])
    except Exception as e:  # recorded, and counted as a failure
        results[key] = dict(ok=False, error=repr(e), wall_s=time.time() - t0)


def group(server, prompts, max_tokens):
    before = server.health().get("sabah_runtime")
    results = {}
    th = [threading.Thread(target=chat, args=(server, p, max_tokens, results, i)) for i, (pid, p) in enumerate(prompts)]
    [t.start() for t in th]
    [t.join() for t in th]
    after = server.health().get("sabah_runtime")
    out = dict(n=len(prompts), ids=[pid for pid, _ in prompts],
               responses=[dict(id=prompts[i][0], **results[i]) for i in range(len(prompts))])
    if before is not None and after is not None:
        d = {k: after[k] - before[k] for k in after if isinstance(after[k], (int, float)) and k in before}
        new_p = sum(r["usage"]["prompt_tokens"] - ((r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                    for r in out["responses"] if r.get("ok"))
        dec = sum(r["usage"]["completion_tokens"] - 1 for r in out["responses"] if r.get("ok"))
        expected = 141 * new_p + 3 * len(prompts) + 144 * dec
        out["counters_delta"] = d
        out["accounting"] = dict(new_prompt_tokens=new_p, decoded_tokens=dec, expected_tokens=expected,
                                 observed_tokens=d.get("tokens"), exact=d.get("tokens") == expected,
                                 lookups_is_10x=d.get("lookups") == 10 * d.get("tokens", -1),
                                 selfcheck_ok=d.get("selfcheck_ok"), selfcheck_fail=d.get("selfcheck_fail"),
                                 verify_fail=d.get("verify_fail"))
    return out


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1], encoding="utf-8"))
    out_path = sys.argv[2]
    res = dict(spec=spec, groups=[])
    srv = Server(spec.get("backend", "sabah"), spec.get("port", 8093), spec.get("backend_port", 18093),
                 out_path + ".server.log", validate=spec.get("validate", True))
    try:
        res["health_start"] = srv.health()
        for g in spec["groups"]:
            res["groups"].append(dict(name=g["name"], **group(srv, [(p["id"], p["text"]) for p in g["prompts"]],
                                                              g["max_tokens"])))
            print(json.dumps({k: v for k, v in res["groups"][-1].items() if k in ("name", "accounting")}))
        if spec.get("stream"):
            s = spec["stream"]
            before = srv.health().get("sabah_runtime")
            chunks, s_reasoning, s_content, usage = post_stream(srv.base() + "/v1/chat/completions", dict(
                messages=[{"role": "user", "content": s["text"]}], max_tokens=s["max_tokens"], temperature=0, seed=1))
            after = srv.health().get("sabah_runtime")
            d = ({k: after[k] - before[k] for k in after if isinstance(after[k], (int, float)) and k in before}
                 if before and after else None)
            twin = next((r for g in res["groups"] if g["name"] == "stream_twin" for r in g["responses"]), None)
            acc = None
            if usage and d:
                new_p = usage["prompt_tokens"] - ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
                expected = 141 * new_p + 3 + 144 * (usage["completion_tokens"] - 1)
                acc = dict(new_prompt_tokens=new_p, decoded_tokens=usage["completion_tokens"] - 1,
                           expected_tokens=expected, observed_tokens=d.get("tokens"),
                           exact=d.get("tokens") == expected, lookups_is_10x=d.get("lookups") == 10 * d.get("tokens", -1),
                           selfcheck_ok=d.get("selfcheck_ok"), selfcheck_fail=d.get("selfcheck_fail"),
                           verify_fail=d.get("verify_fail"))
            res["stream"] = dict(id=s["id"], chunks=chunks, reasoning=s_reasoning, content=s_content, usage=usage,
                                 counters_delta=d, accounting=acc,
                                 text_equals_twin=bool(twin and twin.get("ok") and
                                                       (twin.get("reasoning") or "") == s_reasoning and
                                                       (twin.get("content") or "") == s_content))
            print(json.dumps({"stream_chunks": chunks, "usage": usage, "accounting": acc,
                              "text_equals_twin": res["stream"]["text_equals_twin"]}))
        res["health_end"] = srv.health()
    finally:
        srv.stop()
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
