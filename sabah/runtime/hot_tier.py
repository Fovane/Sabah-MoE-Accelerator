"""
Sabah / runtime / hot_tier

The VRAM residency layer.

This is where Sabah is allowed to be clever, and where it is forbidden to be
clever: it decides WHERE an expert lives and WHEN it moves, never WHICH expert
runs. A miss stalls and fetches the expert the router asked for. There is no
path in this file that substitutes a resident expert for an absent one.

Layout
------
The tier is partitioned per transformer block, because within a block every
expert has the identical byte size (the quant mix varies across blocks, not
within one). That gives fixed-size slots, zero fragmentation and O(1) slot
management. Each slot holds one expert's gate|up|down back to back, so a miss
is a single contiguous transfer.

States
------
    RAM_ONLY      not in VRAM; the backing store is authoritative
    TRANSFERRING  a copy is in flight on the copy stream
    GPU_RESIDENT  usable by the compute stream
    EVICT_PENDING selected as a victim, but its slot is still referenced by
                  work already submitted this step

Telemetry
---------
`gpu_wait_expert_ms` is the primary KPI: wall time the COMPUTE stream spent
blocked on expert transfers. It is measured with events on the compute stream
itself, not inferred from byte counts.
"""
from __future__ import annotations

import time
import ctypes
import collections
import threading
import numpy as np

from sabah.runtime import rt

RAM_ONLY      = "RAM_ONLY"
TRANSFERRING  = "TRANSFERRING"
GPU_RESIDENT  = "GPU_RESIDENT"
EVICT_PENDING = "EVICT_PENDING"

STAGING_SLOTS = 32
STALL_RING    = 512


class Telemetry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.hits = 0
        self.misses = 0
        self.requests = 0
        self.byte_hits = 0
        self.byte_misses = 0
        self.duplicate_transfers_suppressed = 0
        self.evictions = 0
        self.bytes_fetched = 0
        self.gpu_wait_expert_ms = 0.0
        self.blocks_executed = 0
        self.tokens = 0
        self.host_stage_ms = 0.0      # backing store -> pinned staging, on the
                                      # calling thread, i.e. on the critical path
        self.host_stage_bytes = 0
        self.per_block = collections.defaultdict(lambda: {
            "requests": 0, "hits": 0, "misses": 0, "bytes_fetched": 0,
        })

    @property
    def hit_rate(self) -> float:
        n = self.hits + self.misses
        return (self.hits / n) if n else 0.0

    @property
    def byte_hit_rate(self) -> float:
        n = self.byte_hits + self.byte_misses
        return (self.byte_hits / n) if n else 0.0

    def as_dict(self) -> dict:
        return dict(hits=self.hits, misses=self.misses,
                    evictions=self.evictions,
                    requests=self.requests,
                    byte_hits=self.byte_hits,
                    byte_misses=self.byte_misses,
                    byte_hit_rate=round(self.byte_hit_rate, 6),
                    duplicate_transfers_suppressed=self.duplicate_transfers_suppressed,
                    hit_rate=round(self.hit_rate, 6),
                    bytes_fetched=self.bytes_fetched,
                    gpu_wait_expert_ms=round(self.gpu_wait_expert_ms, 3),
                    host_stage_ms=round(self.host_stage_ms, 3),
                    host_stage_bytes=self.host_stage_bytes,
                    blocks_executed=self.blocks_executed,
                    tokens=self.tokens,
                    per_block={str(block): dict(values)
                               for block, values in self.per_block.items()})

    def summary(self) -> str:
        return ("hits %d  misses %d  hit_rate %.4f  evictions %d\n"
                "fetched %.3f GB  gpu_wait_expert %.1f ms  blocks %d"
                % (self.hits, self.misses, self.hit_rate, self.evictions,
                   self.bytes_fetched / 1e9, self.gpu_wait_expert_ms,
                   self.blocks_executed))


class BlockPool:
    """Fixed-size expert slots for one transformer block."""

    def __init__(self, block: int, slot_bytes: int, n_slots: int,
                 role_offsets: dict):
        self.block = block
        self.slot_bytes = slot_bytes
        self.n_slots = n_slots
        self.role_offsets = role_offsets      # role -> byte offset inside a slot
        self.base = None
        if n_slots > 0:
            self.base = rt.check_ptr(
                rt.lib().sabah_dev_alloc(slot_bytes * n_slots),
                "VRAM pool for block %d (%d slots x %d B)" % (block, n_slots, slot_bytes))
        self.free_slots = list(range(n_slots))
        self.slot_of = {}                     # expert -> slot
        self.expert_of = {}                   # slot -> expert
        self.state = {}                       # expert -> state
        self.lru = collections.OrderedDict()  # expert -> None, oldest first

    def slot_ptr(self, slot: int, role: str) -> int:
        return self.base + slot * self.slot_bytes + self.role_offsets[role]

    def free(self):
        if self.base:
            rt.lib().sabah_dev_free(ctypes.c_void_p(self.base))
            self.base = None

    def bytes_used(self) -> int:
        return self.slot_bytes * self.n_slots


class HotTier:
    """VRAM residency across all resident blocks of one device."""

    def __init__(self, bank, capacity_bytes: int, blocks=None,
                 device: int = 0):
        self.bank = bank
        self.device = device
        self.tel = Telemetry()
        self._lock = threading.RLock()
        self.blocks = list(blocks if blocks is not None else bank.blocks)

        # ---- size the per-block pools -------------------------------------
        # Capacity is split evenly across blocks by slot count, because every
        # block routes exactly top-k experts per token: an uneven split would
        # starve some blocks while others held experts they cannot use.
        per = {b: bank.block_expert_bytes(b) for b in self.blocks}
        biggest = max(per.values())
        slots_each = int(capacity_bytes // (biggest * len(self.blocks))) if self.blocks else 0
        slots_each = max(0, min(slots_each, bank.n_experts))

        self.pools = {}
        for b in self.blocks:
            offs, o = {}, 0
            for r in bank.ROLES:
                offs[r] = o
                o += bank.expert_bytes(b, r)
            self.pools[b] = BlockPool(b, per[b], slots_each, offs)

        self.slots_per_block = slots_each
        self.capacity_bytes = sum(p.bytes_used() for p in self.pools.values())

        # ---- pinned staging ring ------------------------------------------
        # Each staging buffer carries its own event, so reusing one waits for
        # that buffer's transfer only, instead of draining the copy stream.
        # A pinned RAM bank can be DMA'd in place: no staging copy exists to
        # sit on the critical path.
        self._direct = bool(getattr(bank, "mode", "") == "ram"
                            and getattr(bank, "pinned", False))
        self._stage_bytes = biggest
        self._stage = []
        self._stage_ev = []
        if slots_each > 0 and not self._direct:
            for _ in range(STAGING_SLOTS):
                p = rt.check_ptr(rt.lib().sabah_host_alloc(self._stage_bytes),
                                 "pinned staging buffer")
                self._stage.append(p)
                self._stage_ev.append(rt.check_ptr(rt.lib().sabah_event_create(),
                                                   "staging event"))
        self._stage_i = 0
        self._stage_inflight = [False] * STAGING_SLOTS

        # ---- stall measurement ---------------------------------------------
        # A ring of event PAIRS recorded on the compute stream. Reading elapsed
        # time synchronises, so nothing is read during the run: pairs are
        # drained afterwards. Measuring the stall must not create one.
        self._ev_fetch = rt.check_ptr(rt.lib().sabah_event_create(), "fetch event")
        self._stall_ring = []
        if slots_each > 0:
            for _ in range(STALL_RING):
                self._stall_ring.append((
                    rt.check_ptr(rt.lib().sabah_event_create(), "wait-begin event"),
                    rt.check_ptr(rt.lib().sabah_event_create(), "wait-end event")))
        self._stall_i = 0
        self._stall_pending = 0
        self._last_compute_submission = False

    # ------------------------------------------------------------------
    def state_of(self, block: int, expert: int) -> str:
        return self.pools[block].state.get(expert, RAM_ONLY)

    def resident(self, block: int) -> list:
        return sorted(self.pools[block].slot_of)

    # ------------------------------------------------------------------
    def preload(self, block: int, experts) -> int:
        """Seed a block's pool synchronously (popularity ordering, warm-up).

        Returns how many experts actually fit."""
        p = self.pools[block]
        n = 0
        for e in experts:
            if len(p.free_slots) == 0:
                break
            if e in p.slot_of:
                continue
            slot = p.free_slots.pop(0)
            self._copy_expert_sync(block, e, slot)
            p.slot_of[e] = slot
            p.expert_of[slot] = e
            p.state[e] = GPU_RESIDENT
            p.lru[e] = None
            n += 1
        return n

    def _copy_expert_sync(self, block: int, expert: int, slot: int):
        p = self.pools[block]
        L = rt.lib()
        for r in self.bank.ROLES:
            src = self.bank.slice(block, expert, r)
            rt.check(L.sabah_memcpy_h2d(
                ctypes.c_void_p(p.slot_ptr(slot, r)),
                src.ctypes.data_as(ctypes.c_void_p), src.nbytes), "preload h2d")
        self.tel.bytes_fetched += self.bank.block_expert_bytes(block)

    # ------------------------------------------------------------------
    def _acquire_stage(self):
        i = self._stage_i
        self._stage_i = (self._stage_i + 1) % STAGING_SLOTS
        if self._stage_inflight[i]:
            # this buffer's own transfer must have landed before it is
            # overwritten; other in-flight copies are left alone
            rt.check(rt.lib().sabah_event_sync(
                ctypes.c_void_p(self._stage_ev[i])), "staging event sync")
            self._stage_inflight[i] = False
        return i, self._stage[i]

    def _fetch_async(self, block: int, expert: int, slot: int):
        """mmap/RAM -> pinned staging -> one contiguous async H2D.

        gate|up|down are adjacent in both the staging buffer and the VRAM
        slot, so a miss costs exactly one transfer, not three."""
        p = self.pools[block]
        L = rt.lib()

        if self._direct:
            # The bank already lives in pinned RAM, so the DMA engine can read
            # it in place. There is no staging copy to pay for; the cost is
            # three transfers instead of one, because gate/up/down are separate
            # tensors in the artifact and stay separate in the bank.
            n = 0
            for r in self.bank.ROLES:
                src = self.bank.slice(block, expert, r)
                rt.check(L.sabah_h2d_async(
                    ctypes.c_void_p(p.slot_ptr(slot, r)),
                    src.ctypes.data_as(ctypes.c_void_p), src.nbytes),
                    "expert h2d async (direct)")
                n += src.nbytes
            self.tel.bytes_fetched += n
            return

        si, stage = self._acquire_stage()
        dst = np.ctypeslib.as_array(
            ctypes.cast(stage, ctypes.POINTER(ctypes.c_uint8)), (self._stage_bytes,))
        t0 = time.perf_counter()
        o = 0
        for r in self.bank.ROLES:
            src = self.bank.slice(block, expert, r)
            dst[o:o + src.nbytes] = src
            o += src.nbytes
        self.tel.host_stage_ms += (time.perf_counter() - t0) * 1000.0
        self.tel.host_stage_bytes += o
        rt.check(L.sabah_h2d_async(ctypes.c_void_p(p.slot_ptr(slot, self.bank.ROLES[0])),
                                   ctypes.c_void_p(stage), o), "expert h2d async")
        rt.check(L.sabah_event_record_copy(ctypes.c_void_p(self._stage_ev[si])),
                 "record staging event")
        self._stage_inflight[si] = True
        self.tel.bytes_fetched += o

    # ------------------------------------------------------------------
    def ensure(self, block: int, experts) -> list:
        """Make every requested expert resident, and return their slots.

        This is the only admission path, and it admits exactly what it was
        asked for. `experts` comes from the router; it is never filtered,
        reordered or substituted here.
        """
        with self._lock:
            return self._ensure_locked(block, experts)

    def _ensure_locked(self, block: int, experts) -> list:
        p = self.pools[block]
        if p.n_slots == 0:
            raise RuntimeError(
                "block %d has no VRAM slots; this tier cannot execute it" % block)

        want = list(experts)
        distinct_want = list(dict.fromkeys(want))
        self.tel.duplicate_transfers_suppressed += max(0, len(want) - len(distinct_want))
        if len(set(want)) > p.n_slots:
            raise RuntimeError(
                "block %d routes %d distinct experts but the pool holds %d slots; "
                "Sabah will not drop an expert to make it fit"
                % (block, len(set(want)), p.n_slots))

        misses = []
        for e in want:
            if e in p.slot_of:
                self.tel.hits += 1
                self.tel.per_block[block]["hits"] += 1
                p.lru.move_to_end(e)
            else:
                self.tel.misses += 1
                self.tel.per_block[block]["misses"] += 1
                misses.append(e)

        self.tel.requests += len(want)
        self.tel.per_block[block]["requests"] += len(want)
        self.tel.byte_hits += sum(self.bank.block_expert_bytes(block)
                                  for e in want if e in p.slot_of)
        self.tel.byte_misses += sum(self.bank.block_expert_bytes(block)
                                    for e in misses)

        # protect everything needed this step from eviction
        protected = set(want)
        for e in dict.fromkeys(misses):
            if p.free_slots:
                slot = p.free_slots.pop(0)
            else:
                # The previous compute submission may still be reading the
                # victim slot.  Overwriting it before that work completes is
                # a silent correctness bug.  This conservative synchronization
                # is only needed on an actual eviction; it is preferable to
                # an occasional wrong token and can later be replaced by
                # per-slot CUDA fences without changing the contract.
                if self._last_compute_submission:
                    rt.check(rt.lib().sabah_sync_compute(),
                             "sync before expert eviction")
                    self._last_compute_submission = False
                slot = self._evict(p, protected)
            p.slot_of[e] = slot
            p.expert_of[slot] = e
            p.state[e] = TRANSFERRING
            p.lru[e] = None
            self._fetch_async(block, e, slot)
            self.tel.per_block[block]["bytes_fetched"] += self.bank.block_expert_bytes(block)

        if misses:
            L = rt.lib()
            ev = ctypes.c_void_p(self._ev_fetch)
            w0, w1 = self._stall_ring[self._stall_i]
            if self._stall_pending >= len(self._stall_ring):
                self.drain_stalls()
                w0, w1 = self._stall_ring[self._stall_i]
            rt.check(L.sabah_event_record_copy(ev), "record fetch")
            rt.check(L.sabah_event_record_compute(ctypes.c_void_p(w0)), "record wait0")
            rt.check(L.sabah_compute_wait_event(ev), "compute wait")
            rt.check(L.sabah_event_record_compute(ctypes.c_void_p(w1)), "record wait1")
            self._stall_i = (self._stall_i + 1) % len(self._stall_ring)
            self._stall_pending += 1
            for e in misses:
                p.state[e] = GPU_RESIDENT
            self._last_compute_submission = True

        return [p.slot_of[e] for e in want]

    def drain_stalls(self):
        """Read back the recorded stall intervals.

        This synchronises, so it is called between measured runs - never
        inside one."""
        L = rt.lib()
        n = min(self._stall_pending, len(self._stall_ring))
        for i in range(n):
            w0, w1 = self._stall_ring[i]
            ms = L.sabah_event_elapsed_ms(ctypes.c_void_p(w0), ctypes.c_void_p(w1))
            if ms >= 0:
                self.tel.gpu_wait_expert_ms += float(ms)
        self._stall_pending = 0
        self._stall_i = 0

    def _evict(self, p: BlockPool, protected: set) -> int:
        for e in list(p.lru):
            if e in protected:
                continue
            p.state[e] = EVICT_PENDING
            slot = p.slot_of.pop(e)
            p.expert_of.pop(slot, None)
            p.lru.pop(e, None)
            p.state[e] = RAM_ONLY
            self.tel.evictions += 1
            return slot
        raise RuntimeError("every slot in block %d is protected this step" % p.block)

    # ------------------------------------------------------------------
    def slot_ptrs(self, block: int, slots) -> dict:
        p = self.pools[block]
        return {r: [p.slot_ptr(s, r) for s in slots] for r in self.bank.ROLES}

    def free(self):
        with self._lock:
            L = rt.lib()
            for p in self.pools.values():
                p.free()
            for s in self._stage:
                L.sabah_host_free(ctypes.c_void_p(s))
            evs = list(self._stage_ev) + [self._ev_fetch]
            for a, b in self._stall_ring:
                evs += [a, b]
            for ev in evs:
                L.sabah_event_destroy(ctypes.c_void_p(ev))
            self._stage = []
            self._stage_ev = []
            self._stall_ring = []

    def describe(self) -> str:
        return ("hot tier   : %d blocks x %d slots = %.3f GB VRAM\n"
                "slot size  : %s bytes (gate|up|down contiguous)\n"
                "coverage   : %.1f%% of those blocks' experts"
                % (len(self.blocks), self.slots_per_block,
                   self.capacity_bytes / 1e9,
                   "{:,}".format(self.pools[self.blocks[0]].slot_bytes) if self.blocks else 0,
                   100.0 * self.slots_per_block / max(1, self.bank.n_experts)))
