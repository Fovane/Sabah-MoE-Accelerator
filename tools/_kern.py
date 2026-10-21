import sqlite3, sys, collections
db = sys.argv[1]
c = sqlite3.connect(db); c.text_factory = bytes

def names():
    return {k: v.decode("utf-8", "replace") for k, v in c.execute("select id,value from StringIds")}

s = names()

print("=== processes / streams ===")
try:
    for row in c.execute("select * from PROCESSES"):
        print(row)
except Exception as e:
    print("PROCESSES err", e)

print("\n=== top runtime API by count ===")
try:
    for row in c.execute("select nameId, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId order by count(*) desc limit 15"):
        print("%-45s n=%d sum_us=%.0f" % (s.get(row[0], "?"), row[1], row[2] / 1000))
except Exception as e:
    print("runtime err", e)

print("\n=== top kernels by count ===")
try:
    for row in c.execute("select demangledName, count(*), sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by demangledName order by count(*) desc limit 20"):
        nm = s.get(row[0], "?")
        print("%-55s n=%d sum_ms=%.1f" % (nm[:55], row[1], row[2] / 1e6))
except Exception as e:
    print("kernel err", e)
