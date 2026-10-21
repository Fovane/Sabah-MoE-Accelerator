"""Offline replay of a routing trace against a fixed-size slab, per policy.

This is Phase 2 of the long-context work: it answers whether *any* replacement
policy can reach a hit rate, without touching the GPU. It replays the same
access sequence at several slab sizes and policies and reports the hit rate and
the misses per decode token.

Policies:
  lru   recency, victim = least recently used (what the runtime does today)
  lfu   frequency, victim = least frequently used
  2q    scan resistant: a small FIFO probation queue feeding an LRU main list
  arc   adaptive replacement cache (t1/t2/b1/b2)
  perlayer  per-layer budgets, each layer getting slab/layers slots, LRU inside
  belady  offline optimum: evict the entry used furthest in the future

    python tools/slab_replay.py trace.bin --slabs 2.0,2.4,3,4,6,8 --expert-kb 682
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
from collections import OrderedDict, defaultdict, deque

REC = struct.Struct("<IIII")


class LRU:
    name = "lru"

    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()

    def access(self, k):
        d = self.d
        if k in d:
            d.move_to_end(k)
            return True
        if len(d) >= self.cap:
            d.popitem(last=False)
        d[k] = True
        return False


class LFU:
    name = "lfu"

    def __init__(self, cap):
        self.cap = cap
        self.d = {}
        self.f = defaultdict(int)

    def access(self, k):
        if k in self.d:
            self.f[k] += 1
            return True
        if len(self.d) >= self.cap:
            victim = min(self.d, key=lambda x: (self.f[x], x))
            del self.d[victim]
            del self.f[victim]
        self.d[k] = True
        self.f[k] = 1
        return False


class TwoQ:
    name = "2q"

    def __init__(self, cap):
        self.cap = cap
        self.kin = cap // 4
        self.kout = max(1, cap)
        self.a1in = deque()
        self.a1out = deque()
        self.am = OrderedDict()

    def access(self, k):
        if k in self.am:
            self.am.move_to_end(k)
            return True
        if len(self.a1in) >= self.kin:
            old = self.a1in.popleft()
            if len(self.a1out) >= self.kout:
                self.a1out.popleft()
            self.a1out.append(old)
        self.a1in.append(k)
        if len(self.am) < self.cap - self.kin:
            self.am[k] = True
            return False
        # promote from out-list instead of evicting main
        if k in self.a1out:
            self.am[k] = True
            return False
        self.am.popitem(last=False)
        self.am[k] = True
        return False


class ARC:
    name = "arc"

    def __init__(self, cap):
        self.c = max(2, cap)
        self.p = 0
        self.t1 = OrderedDict()
        self.t2 = OrderedDict()
        self.b1 = OrderedDict()
        self.b2 = OrderedDict()

    def _replace(self, in_b2):
        if self.t1 and (len(self.t1) > self.p or (in_b2 and len(self.t1) == self.p)):
            k, _ = self.t1.popitem(last=False)
            self.b1[k] = True
        else:
            k, _ = self.t2.popitem(last=False)
            self.b2[k] = True

    def access(self, k):
        if k in self.t1:
            del self.t1[k]
            self.t2[k] = True
            return True
        if k in self.t2:
            self.t2.move_to_end(k)
            return True
        if k in self.b1:
            d = 1 if len(self.b1) >= len(self.b2) else 0
            self.p = min(self.p + d, self.c)
            self._replace(False)
            del self.b1[k]
            self.t2[k] = True
            return False
        if k in self.b2:
            d = 1 if len(self.b2) >= len(self.b1) else 0
            self.p = max(self.p - d, 0)
            self._replace(True)
            del self.b2[k]
            self.t2[k] = True
            return False
        if len(self.t1) + len(self.b1) == self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
                self._replace(False)
            else:
                self.t1.popitem(last=False)
        else:
            if len(self.t1) + len(self.b1) + len(self.t2) + len(self.b2) >= self.c:
                if len(self.t2) + len(self.b2) < self.c:
                    self._replace(False)
                else:
                    self.t2.popitem(last=False)
        self.t1[k] = True
        return False


class PerLayer:
    name = "perlayer"

    def __init__(self, cap, n_layers):
        self.cap = max(1, cap // max(1, n_layers))
        self.d = defaultdict(lambda: OrderedDict())

    def access(self, k):
        layer = k[1]
        d = self.d[layer]
        if k in d:
            d.move_to_end(k)
            return True
        if len(d) >= self.cap:
            d.popitem(last=False)
        d[k] = True
        return False


def belady(trace, cap):
    """Offline optimum: evict the resident entry used furthest in the future.

    O(n log n) with a lazy heap: pushed entries may be stale, so the eviction
    loop pops until it finds one whose stored next use still matches the
    current one. Anything O(n*cap) does not finish on a 2.9M-access trace.
    """
    import heapq

    n = len(trace)
    next_use = [0] * n
    last = {}
    for i in range(n - 1, -1, -1):
        k = trace[i]
        next_use[i] = last.get(k, n)
        last[k] = i

    resident = {}   # key -> its current next-use position
    heap = []       # (-next_use, key), lazy
    misses = 0
    for i, k in enumerate(trace):
        cur = next_use[i]
        if k in resident:
            resident[k] = cur
            heapq.heappush(heap, (-cur, k))
            continue
        misses += 1
        if len(resident) >= cap:
            while heap:
                neg, vk = heapq.heappop(heap)
                if vk in resident and resident[vk] == -neg:
                    del resident[vk]
                    break
        resident[k] = cur
        heapq.heappush(heap, (-cur, k))
    return misses


def read_trace(path):
    trace = []
    with open(path, "rb") as f:
        while True:
            b = f.read(REC.size)
            if len(b) < REC.size:
                break
            trace.append(REC.unpack(b))
    return trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--slabs", default="2.0,2.4,3,4,6,8")
    ap.add_argument("--expert-kb", type=float, default=682.0)
    ap.add_argument("--steps-per-token", type=int, default=1440,
                    help="accesses per decode token (48 layers x 10 topk x 3 roles)")
    ap.add_argument("--policies", default="lru,lfu,2q,arc,perlayer,belady")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    trace = read_trace(args.trace)
    if not trace:
        print("empty trace")
        return 1
    n_layers = len({k[2] for k in trace})
    steps = len({k[0] for k in trace})
    print("trace: %d accesses, %d steps, %d layers" % (len(trace), steps, n_layers))
    per_tok = len(trace) / max(1, steps)
    print("accesses per step: %.0f (assumed %d)" % (per_tok, args.steps_per_token))
    if abs(per_tok - args.steps_per_token) > 0.3 * args.steps_per_token:
        args.steps_per_token = int(per_tok)

    keys = [(r, l, e) for _s, r, l, e in trace]
    rows = []
    for slab_gb in [float(x) for x in args.slabs.split(",")]:
        cap = int(slab_gb * 1e9 / (args.expert_kb * 1024.0))
        for pol in args.policies.split(","):
            if pol == "belady":
                misses = belady(keys, cap)
            else:
                cls = {"lru": LRU, "lfu": LFU, "2q": TwoQ, "arc": ARC,
                       "perlayer": PerLayer}[pol]
                sim = cls(cap, n_layers) if pol == "perlayer" else cls(cap)
                misses = 0
                for k in keys:
                    if not sim.access(k):
                        misses += 1
            hits = len(keys) - misses
            rows.append((slab_gb, pol, cap, hits / len(keys), misses / max(1, steps)))

    print("\nslab GB | policy   | slots | hit rate | misses/token")
    print("--------+----------+-------+----------+-------------")
    for slab_gb, pol, cap, hr, mt in rows:
        print("%7.1f | %-8s | %5d | %7.2f%% | %11.1f" % (slab_gb, pol, cap, hr * 100, mt))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("slab_gb,policy,slots,hit_rate,misses_per_token\n")
            for slab_gb, pol, cap, hr, mt in rows:
                f.write("%.1f,%s,%d,%.6f,%.2f\n" % (slab_gb, pol, cap, hr, mt))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
