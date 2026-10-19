import csv
import statistics
import sys

labels = sys.argv[1:] or [
    "gpu-mem-6.0", "gpu-mem-7.0", "gpu-mem-8.0", "gpu-mem-9.8", "gpu-mem-c512",
]
for l in labels:
    rows = []
    for r in csv.DictReader(open(f"results/phase0/{l}/gpu_mem.csv")):
        try:
            vals = {k: float(v) for k, v in r.items() if k != "t_s" and v not in (None, "")}
        except ValueError:
            continue
        if len(vals) == 7:
            rows.append(vals)
    if not rows:
        print(f"{l:14s} no samples")
        continue
    mid = rows[len(rows) // 2:]

    def med(k):
        return statistics.median(x[k] for x in mid)

    print(f"{l:14s} ded={med('adapter_ded_mib'):7.0f} shared={med('adapter_shared_mib'):6.0f} "
          f"ours_ded={med('ours_ded_mib'):7.0f} ours_shared={med('ours_shared_mib'):6.0f} "
          f"other_ded={med('other_ded_mib'):5.0f}")
