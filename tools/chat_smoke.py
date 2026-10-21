# Two questions against the running local server: cold, then warm.
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
QUESTIONS = [
    "Merhaba. Turkce ve kisa cevap ver: 2+2 kacar?",
    "Bir onceki cevabin kac cikti? Tek kelimeyle yaz.",
]


def ask(q, n):
    body = json.dumps({
        "model": "sabah",
        "messages": [{"role": "user", "content": q}],
        "max_tokens": 48,
        "temperature": 0.7,
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    dt = time.time() - t0
    msg = r["choices"][0]["message"]["content"].strip()
    ct, pt = r["usage"]["completion_tokens"], r["usage"]["prompt_tokens"]
    print("Q%d: %s" % (n, q))
    print("  answer  : %r" % msg[:300])
    print("  tokens  : prompt %d, completion %d" % (pt, ct))
    print("  wall    : %.1fs  ->  %.2f tok/s (prompt eval included)"
          % (dt, ct / dt))
    print()
    return msg


for i, q in enumerate(QUESTIONS, 1):
    ask(q, i)

try:
    models = json.loads(urllib.request.urlopen(BASE + "/v1/models", timeout=10).read())
    print("/v1/models:", json.dumps(models, ensure_ascii=False)[:400])
except Exception as e:                                   # noqa: BLE001
    print("/v1/models failed:", e)
