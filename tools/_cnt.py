import sqlite3, sys
db = sys.argv[1]
c = sqlite3.connect(db); c.text_factory = bytes
for t in ["CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"]:
    try:
        n = c.execute("select count(*) from %s" % t).fetchone()[0]
        print(t, n)
    except Exception as e:
        print(t, "MISSING")
