import csv, sys, statistics, json, os

path = sys.argv[1]
rows = []
with open(path, newline="") as f:
    for r in csv.DictReader(f):
        try:
            rows.append({
                "t": float(r["t_s"]),
                "sm": float(r["sm_mhz"]),
                "mem": float(r["mem_mhz"]),
                "pstate": r["pstate"].strip(),
                "pw": float(r["power_w"]),
                "temp": float(r["temp_c"]),
                "util": float(r["util_pct"]),
                "thr": r["throttle"].strip(),
            })
        except (ValueError, KeyError):
            pass

def win(lo, hi):
    return [r for r in rows if lo <= r["t"] < hi]

def summ(name, w):
    if not w:
        print(f"{name}: no samples")
        return
    print(f"{name}: n={len(w)}  sm {min(x['sm'] for x in w):.0f}/{statistics.median(x['sm'] for x in w):.0f}/{max(x['sm'] for x in w):.0f}"
          f"  mem {min(x['mem'] for x in w):.0f}/{statistics.median(x['mem'] for x in w):.0f}/{max(x['mem'] for x in w):.0f}"
          f"  pw {min(x['pw'] for x in w):.1f}/{statistics.median(x['pw'] for x in w):.1f}/{max(x['pw'] for x in w):.1f}W"
          f"  util {statistics.median(x['util'] for x in w):.0f}%  temp {statistics.median(x['temp'] for x in w):.0f}C"
          f"  pstate {sorted(set(x['pstate'] for x in w))}  throttle {sorted(set(x['thr'] for x in w))}")

print("total samples", len(rows), "span %.1fs" % (rows[-1]["t"] if rows else 0))
summ("idle(0-1s)", win(0, 1))
summ("fill(1-13s)", win(1, 13))
summ("steady(13-40s)", win(13, 40))
summ("steady(20-40s)", win(20, 40))

d = os.path.dirname(path)
mj = os.path.join(d, "measurement.json")
if os.path.exists(mj):
    m = json.load(open(mj))
    print("steady_ms_median=%.3f  tok/s=%.3f  fill_s=%.2f  warm_ms=%.3f" % (
        m.get("steady_ms_median", 0), m.get("steady_tok_per_s", 0),
        m.get("fill_s", 0), m.get("warm_ms_median", 0) or 0))
