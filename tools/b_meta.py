import json, os, sys

RUNS = [
    ("b-eq-ref3", "short prompt 53 tok, compare"),
    ("b-eq-grp3", "same prompt, group"),
    ("b-eq-grp3-g24", "same prompt, group G=24"),
    ("b-eq-ref777", "777 tok prompt, plain"),
    ("b-eq-grp777", "777 tok prompt, group"),
    ("b-eq-650-g24", "777 tok prompt, group G=24"),
    ("b-eq-1630-ref", "1888 tok prompt (1630 file), plain"),
    ("b-eq-1630-grp", "1888 tok prompt, group"),
    ("b-eq-1630-grp-ub2k", "1888 tok prompt, group, ub=2048"),
    ("b-eq-1630-grp2", "1888 tok prompt, group, sync fix"),
    ("b-eq-cap1888-ref", "1888 tok, plain, cap hc_init"),
    ("b-eq-cap1888-grp", "1888 tok, group, cap hc_init"),
    ("b-eq-cap1888b-ref", "1888 tok, plain ub=2048, cap hc_mixed L47"),
    ("b-eq-cap1888b-grp", "1888 tok, group ub=2048, cap hc_mixed L47"),
    ("b-hc47-ref2", "1888 tok, plain ub=512, cap hc_mixed L47"),
    ("b-hc47-grp2", "1888 tok, group ub=512, cap hc_mixed L47"),
    ("b-hc3456-0", "1888 tok, plain ub=512, cap L3/4/5/6"),
    ("b-hc3456-1", "1888 tok, group ub=512, cap L3/4/5/6"),
    ("b-dbg7", "1888 tok, group ub=512, dev dump"),
    ("b-dbg8", "1888 tok, group ub=512, dev dump all ub"),
]


def val(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv else "-"


def env_of(rec):
    # measurement.json argv is the diag argv; SABAH_* came from --set, recorded nowhere
    return rec.get("env", {})


for name, note in RUNS:
    p = os.path.join("results/phase0", name, "measurement.json")
    if not os.path.exists(p):
        print(f"{name:22s} MISSING")
        continue
    rec = json.load(open(p))
    argv = rec.get("argv", [])
    print(f"{name:22s} argv: -b {val(argv,'-b'):>5} -ub {val(argv,'-ub'):>5} "
          f"-c {val(argv,'-c'):>6} -n {val(argv,'-n'):>3}  | argv_env: {env_of(rec)}  | {note}")
