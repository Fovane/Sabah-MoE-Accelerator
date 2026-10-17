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
        self.prefetch_issued = 0
        self.prefetch_hits = 0
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
                    prefetch_issued=self.prefetch_issued,
                    prefetch_hits=self.prefetch_hits,
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
        self.inflight = {}                    # expert -> copy event, in flight

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
                 device: int = 0, prefetch_slots: int = 0):
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
        # Slots kept free for prefetch (advice), so a prefetch never has to
        # evict something the current step may still read. 0 = prefetch off.
        #
        # Clamped so the live working set always fits: a single step routes to
        # top-k experts and every one of them must be resident at the same time,
        # so reserving more than slots_each - top_k would make the step
        # unservable ("every slot is protected this step"). Prefetch is advice,
        # so the tier degrades to fewer (or zero) reserved slots instead of
        # failing the engine.
        live = int(getattr(bank, "n_experts_used", 0) or 0)
        self.prefetch_slots = max(0, min(int(prefetch_slots), slots_each - live))

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

        # ---- per-transfer events -------------------------------------------
        # Every miss and every prefetch gets its own completion event, so the
        # compute stream can wait on exactly the copies it needs instead of
        # draining the copy stream. Events are pooled; the pool never grows
        # beyond the number of simultaneously-outstanding transfers.
        self._ev_free = []
        self._ev_all = []
        # One "slot readers are done" event per block, recorded on the compute
        # stream right after that block's kernel is submitted. Before any copy
        # overwrites a slot, the copy stream waits on it -- the precise
        # dependency that replaces draining the whole compute stream on every
        # eviction. See `mark_submitted`.
        self._read_ev = {}

        # ---- stall measurement ---------------------------------------------
        # A ring of event PAIRS recorded on the compute stream. Reading elapsed
        # time synchronises, so nothing is read during the run: pairs are
        # drained afterwards. Measuring the stall must not create one.
        self._stall_ring = []
        if slots_each > 0:
            for _ in range(STALL_RING):
                self._stall_ring.append((
                    rt.check_ptr(rt.lib().sabah_event_create(), "wait-begin event"),
                    rt.check_ptr(rt.lib().sabah_event_create(), "wait-end event")))
        self._stall_i = 0
        self._stall_pending = 0

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

    def _ev_get(self):
        if self._ev_free:
            return self._ev_free.pop()
        ev = rt.check_ptr(rt.lib().sabah_event_create(), "transfer event")
        self._ev_all.append(ev)
        return ev

    def _ev_put(self, ev):
        self._ev_free.append(ev)

    def mark_submitted(self, block: int):
        """Record that `block`'s kernel is now enqueued on the compute stream.

        The executor calls this right after submitting `sabah_moe_block`. It
        anchors the event that every later copy into this block's slots waits on
        (`_start_copy`), so a slot is never overwritten while a submitted kernel
        reads it -- without draining the compute stream.
        """
        ev = self._read_ev.get(block)
        if ev is None:
            ev = rt.check_ptr(rt.lib().sabah_event_create(), "block read event")
            self._read_ev[block] = ev
        rt.check(rt.lib().sabah_event_record_compute(ctypes.c_void_p(ev)),
                 "record block read event")

    def _start_copy(self, block: int, expert: int, slot: int) -> int:
        """Start the expert->VRAM copy on the copy stream.

        Returns a completion event recorded AFTER the copy. The caller decides
        when to make the compute stream wait on it; a prefetch can hold it, a
        miss waits immediately. gate|up|down are adjacent in both the staging
        buffer and the VRAM slot, so a miss costs one contiguous transfer, not
        three -- except in direct-DMA mode, where the three live tensors are
        read in place and cost three transfers.
        """
        p = self.pools[block]
        L = rt.lib()

        # Ordering: a copy must never overwrite a slot that a *submitted* kernel
        # may still be reading. The block's read event was recorded on the
        # compute stream right after that kernel, so waiting on it here is the
        # exact dependency -- not a full compute-stream drain.
        rv = self._read_ev.get(block)
        if rv is not None:
            rt.check(L.sabah_copy_wait_event(ctypes.c_void_p(rv)), "copy wait read")

        # If this block's experts already live in PINNED RAM -- either the whole
        # bank was opened mode="ram", or a warm tier pinned this block -- the DMA
        # engine can read them in place. There is no staging copy to pay for, so
        # the warm tier removes the host hop as well as the SSD read.
        pinned_block = getattr(self.bank, "pinned_block", None)
        direct = self._direct or (pinned_block is not None
                                  and bool(pinned_block(block)))
        if direct:
            n = 0
            for r in self.bank.ROLES:
                src = self.bank.slice(block, expert, r)
                rt.check(L.sabah_h2d_async(
                    ctypes.c_void_p(p.slot_ptr(slot, r)),
                    src.ctypes.data_as(ctypes.c_void_p), src.nbytes),
                    "expert h2d async (direct)")
                n += src.nbytes
            ev = self._ev_get()
            rt.check(L.sabah_event_record_copy(ctypes.c_void_p(ev)),
                     "record fetch event")
            self.tel.bytes_fetched += n
            return ev

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
        # Two records, deliberately: the staging buffer must not be overwritten
        # until ITS copy lands (that is `_stage_ev`, checked in _acquire_stage),
        # and the compute stream must not read the slot until the same copy
        # lands (that is the pooled event, waited on where it is needed).
        rt.check(L.sabah_event_record_copy(ctypes.c_void_p(self._stage_ev[si])),
                 "record staging event")
        self._stage_inflight[si] = True
        ev = self._ev_get()
        rt.check(L.sabah_event_record_copy(ctypes.c_void_p(ev)),
                 "record fetch event")
        self.tel.bytes_fetched += o
        return ev

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

        L = rt.lib()
        misses = []
        for e in want:
            if e in p.slot_of:
                if e in p.inflight:
                    # A prefetch already copied (or is copying) this expert's
                    # bytes. It is only safe to use once the copy lands, so the
                    # compute stream waits on that transfer's own event. This is
                    # still a hit: the router asked and got its exact expert.
                    ev = p.inflight.pop(e)
                    rt.check(L.sabah_compute_wait_event(ctypes.c_void_p(ev)),
                             "prefetch wait")
                    self._ev_put(ev)
                    p.state[e] = GPU_RESIDENT
                    self.tel.prefetch_hits += 1
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
        miss_events = []
        for e in dict.fromkeys(misses):
            if p.free_slots:
                slot = p.free_slots.pop(0)
            else:
                slot = self._evict(p, protected)
            p.slot_of[e] = slot
            p.expert_of[slot] = e
            p.state[e] = TRANSFERRING
            p.lru[e] = None
            miss_events.append(self._start_copy(block, e, slot))
            self.tel.per_block[block]["bytes_fetched"] += self.bank.block_expert_bytes(block)

        # Keep the prefetch reserve free: shed the oldest non-protected entries
        # so the next prefetch has somewhere to land without evicting live work.
        reserve = min(self.prefetch_slots, max(0, p.n_slots - len(protected)))
        soft_cap = p.n_slots - reserve
        while len(p.slot_of) > soft_cap:
            try:
                self._evict(p, protected)
            except RuntimeError:
                break

        if misses:
            w0, w1 = self._stall_ring[self._stall_i]
            if self._stall_pending >= len(self._stall_ring):
                self.drain_stalls()
                w0, w1 = self._stall_ring[self._stall_i]
            rt.check(L.sabah_event_record_compute(ctypes.c_void_p(w0)), "record wait0")
            for ev in miss_events:
                rt.check(L.sabah_compute_wait_event(ctypes.c_void_p(ev)),
                         "compute wait")
                self._ev_put(ev)
            rt.check(L.sabah_event_record_compute(ctypes.c_void_p(w1)), "record wait1")
            self._stall_i = (self._stall_i + 1) % len(self._stall_ring)
            self._stall_pending += 1
            for e in misses:
                p.state[e] = GPU_RESIDENT

        return [p.slot_of[e] for e in want]

    # ------------------------------------------------------------------
    def prefetch(self, block: int, experts) -> int:
        """Warm the tier with experts the router has NOT asked for yet.

        This is ADVICE, and it is fenced as such: it only ever fills slots that
        are already free (the reserve `_ensure_locked` keeps open), never evicts
        a slot that submitted or upcoming work may read, and never changes which
        expert runs. If the prediction is wrong the bytes are simply evicted
        later; if capacity is short the call is a no-op.

        Returns how many experts were queued.
        """
        with self._lock:
            p = self.pools[block]
            if p.n_slots == 0 or self.prefetch_slots <= 0:
                return 0
            n = 0
            for e in dict.fromkeys(int(x) for x in experts):
                if e in p.slot_of:
                    if e not in p.inflight:
                        p.lru.move_to_end(e)
                    continue
                if not p.free_slots:
                    break
                slot = p.free_slots.pop(0)
                p.slot_of[e] = slot
                p.expert_of[slot] = e
                p.state[e] = TRANSFERRING
                p.lru[e] = None
                p.inflight[e] = self._start_copy(block, e, slot)
                n += 1
            self.tel.prefetch_issued += n
            return n

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
            if e in p.inflight:
                self._ev_put(p.inflight.pop(e))
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
            evs = list(self._stage_ev) + list(self._ev_all) \
                + list(self._read_ev.values())
            for a, b in self._stall_ring:
                evs += [a, b]
            for ev in evs:
                L.sabah_event_destroy(ctypes.c_void_p(ev))
            self._stage = []
            self._stage_ev = []
            self._ev_free = []
            self._ev_all = []
            self._read_ev = {}
            self._stall_ring = []

    def describe(self) -> str:
        return ("hot tier   : %d blocks x %d slots = %.3f GB VRAM\n"
                "slot size  : %s bytes (gate|up|down contiguous)\n"
                "coverage   : %.1f%% of those blocks' experts\n"
                "prefetch   : %d slot(s) reserved per block"
                % (len(self.blocks), self.slots_per_block,
                   self.capacity_bytes / 1e9,
                   "{:,}".format(self.pools[self.blocks[0]].slot_bytes) if self.blocks else 0,
                   100.0 * self.slots_per_block / max(1, self.bank.n_experts),
                   self.prefetch_slots))
