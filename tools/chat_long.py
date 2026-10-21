# A 200-token generation: does the server settle into the steady regime?
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8080"
body = json.dumps({
    "model": "sabah",
    "messages": [{"role": "user",
                  "content": "En sevdigin 3 sehri ve nedenlerini kisa kisa yaz."}],
    "max_tokens": 200,
    "temperature": 0.7,
}).encode()
req = urllib.request.Request(BASE + "/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
t0 = time.time()
r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
dt = time.time() - t0
ct = r["usage"]["completion_tokens"]
print("completion tokens : %d" % ct)
print("wall              : %.1fs" % dt)
print("end-to-end        : %.2f tok/s (prompt eval + decode)" % (ct / dt))
print()
print(r["choices"][0]["message"]["content"][:600])
