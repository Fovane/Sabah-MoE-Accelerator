#!/usr/bin/env python3
"""Compare per-node fusion decisions between two SABAH_FUSE_TRACE logs.

The trace has one FUSE line per visited node (name, op, nodes_to_skip) and one
FUSE_SPLIT line per scheduler split. Node indices are relative to a split, so
the comparison uses the node name, which is stable.

    python tools/fuse_diff.py a.log b.log
"""
import collections
import sys


def blocks(path):
    out, cur = [], None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("FUSE_SPLIT"):
                cur = []
                out.append(cur)
            elif line.startswith("FUSE ") and cur is not None:
                parts = line.split()
                if len(parts) < 4 or not parts[-1].lstrip("-").isdigit():
                    continue
                # FUSE <name, may contain spaces> <op> <skip>
                cur.append((" ".join(parts[1:-2]), parts[-2], int(parts[-1])))
    return out


def main(a_path, b_path):
    a, b = blocks(a_path), blocks(b_path)
    print("blocks: %s=%d %s=%d" % (a_path, len(a), b_path, len(b)))
    n = min(len(a), len(b))
    bad = 0
    for k in range(n):
        if a[k] == b[k]:
            continue
        bad += 1
        ca = collections.Counter(a[k])
        cb = collections.Counter(b[k])
        only_a = list((ca - cb).elements())
        only_b = list((cb - ca).elements())
        print("block %d: %d vs %d nodes, %d only in A, %d only in B" %
              (k, len(a[k]), len(b[k]), len(only_a), len(only_b)))
        for name, op, skip in only_a[:6]:
            print("  A only: %s %s skip=%d" % (name, op, skip))
        for name, op, skip in only_b[:6]:
            print("  B only: %s %s skip=%d" % (name, op, skip))
    print("%d/%d blocks identical" % (n - bad, n))
    return 0 if bad == 0 and len(a) == len(b) else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
