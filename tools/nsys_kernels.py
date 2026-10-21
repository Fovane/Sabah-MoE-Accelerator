#!/usr/bin/env python3
"""Per-kernel breakdown of a decode window in an exported nsys SQLite file.

Reports ms/token, kernels/token and us/call per kernel name, plus GPU busy and
host API time. The decode window is taken from the first `k_mul_mat_id` shorter
than 1 ms (the Sabah bridge runs once per expert tensor per layer, so the
prefill calls are the long ones) to the end of the trace.

Usage:
    nsys export --type sqlite -o trace.sqlite report.nsys-rep
    python tools/nsys_kernels.py trace.sqlite
"""
import collections
import sqlite3
import sys

db = sys.argv[1] if len(sys.argv) > 1 else r"$DRIVE\Temp\nsys\ctx128.sqlite"
c = sqlite3.connect(db)
c.text_factory = bytes
cur = c.cursor()
cur.execute("select id,value from StringIds")
s = {k: v.decode("utf-8", "replace") for k, v in cur.fetchall()}

cur.execute("select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL order by start")
rows = cur.fetchall()

mmid = [(a, e) for a, e, n in rows if s.get(n, "").startswith("k_mul_mat_id")]
small = [a for a, e in mmid if e - a < 1_000_000]
t_start = small[0] if small else rows[0][0]
n_tok = max(len(small) / 144.0, 1.0)

sel = [r for r in rows if r[0] >= t_start]
print(f"decode window {n_tok:.2f} tokens, {len(sel)} kernels, "
      f"{len(sel)/n_tok:.0f} kernels/token")

tot = collections.Counter()
cnt = collections.Counter()
for a, e, n in sel:
    key = s.get(n, str(n)).split("(")[0].split("<")[0].strip()
    tot[key] += e - a
    cnt[key] += 1

print(f"gpu busy/token = {sum(tot.values())/n_tok/1e6:.2f} ms")
print(f"{'ms/tok':>8} {'cnt/tok':>8} {'us/call':>8}  kernel")
for k, v in tot.most_common(25):
    print(f"{v/n_tok/1e6:8.2f} {cnt[k]/n_tok:8.1f} {v/cnt[k]/1000:8.2f}  {k[:58]}")

agg = collections.defaultdict(lambda: [0, 0])
cur.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where start>=?", (t_start,))
for a, e, n in cur.fetchall():
    k = s.get(n, str(n))
    agg[k][0] += 1
    agg[k][1] += e - a
print(f"{'calls/tok':>9} {'ms/tok':>7} {'us/call':>8}  host api")
for k, (n, d) in sorted(agg.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"{n/n_tok:9.1f} {d/n_tok/1e6:7.2f} {d/1000/n:8.2f}  {k[:48]}")
