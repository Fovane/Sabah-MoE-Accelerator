import sqlite3, sys
db = sys.argv[1]
c = sqlite3.connect(db)
c.text_factory = bytes
tabs = [r[0].decode("utf-8", "replace") for r in c.execute("select name from sqlite_master where type='table'")]
want = [t for t in tabs if "KERNEL" in t or "RUNTIME" in t or "MEMCPY" in t]
print("kernel/runtime tables:", want)
for t in want:
    if "KERNEL" in t:
        try:
            n = c.execute("select count(*) from %s" % t).fetchone()[0]
            print(t, "rows:", n)
        except Exception as e:
            print(t, "err", e)
