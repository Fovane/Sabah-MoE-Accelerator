import csv, sys, os, collections

d = sys.argv[1]
raw = os.path.join(d, "engines.csv")
mp = os.path.join(d, "pidmap.csv")
names = {}
if os.path.exists(mp):
    with open(mp) as f:
        for r in csv.reader(f):
            if len(r) >= 2:
                names[r[0]] = r[1]

rows = []
with open(raw) as f:
    for r in csv.DictReader(f):
        try:
            rows.append((float(r["t_s"]), r["pid"], r["engtype"], float(r["util"])))
        except (ValueError, KeyError):
            pass

# decode window: fill ends ~11.3s, steady starts ~13s, run ends ~47s
dec = [x for x in rows if 1.0 <= x[0] < 47.0]
steady = [x for x in rows if 14.0 <= x[0] < 45.0]
print("samples with util>0.05 in decode window:", len(dec))

def agg(win, title):
    print("\n" + title)
    per = collections.defaultdict(list)
    for t, pid, eng, u in win:
        per[(pid, eng)].append(u)
    items = sorted(per.items(), key=lambda kv: -sum(kv[1]))
    print("%-10s %-10s %-22s %8s %8s %8s" % ("pid", "process", "engine", "samples", "mean%", "max%"))
    for (pid, eng), vals in items[:20]:
        print("%-10s %-10s %-22s %8d %8.1f %8.1f" % (
            pid, names.get(pid, "?")[:10], eng, len(vals),
            sum(vals)/len(vals), max(vals)))

agg(dec, "=== engine utilization by process (fill+steady, 1-47s) ===")
agg(steady, "=== engine utilization by process (steady only, 14-45s) ===")

# total per engine type in steady
tot = collections.defaultdict(float)
cnt = collections.defaultdict(int)
for t, pid, eng, u in steady:
    tot[eng] += u
    cnt[eng] += 1
print("\n=== steady: engine totals ===")
for eng in sorted(tot, key=lambda k: -tot[k]):
    print("%-22s sum_of_pct=%10.1f  active_samples=%d" % (eng, tot[eng], cnt[eng]))
