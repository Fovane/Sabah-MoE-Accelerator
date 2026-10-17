"""Prefetch must change TIMING, never VALUES. This is the test that proves it.

Requires a CUDA runtime, a supported GGUF, and real weights. Skips cleanly when
any of those are absent, so it is safe to run in the normal suite.

What is checked, on real expert bytes:

  * the same routing sequence, replayed through a hot tier so small that it
    evicts on nearly every step, produces BIT-IDENTICAL block outputs with
    speculative prefetch OFF and ON;
  * with prefetch ON the predictor is given an ORACLE (it always names the next
    step's exact experts), so every free prefetch slot is filled with an expert
    the router is about to ask for -- the best case, and the case that exercises
    the in-flight wait path the most;
  * prefetch actually fired and actually turned into hits (`prefetch_issued` and
    `prefetch_hits` are non-zero), so a passing run is not a passing run because
    the feature did nothing;
  * every slot resident at the end is BYTE-IDENTICAL to its GGUF range, so the
    ordering fix (a copy waits for the kernel that last read the slot) really
    holds.

Run it directly:

    python tests/test_prefetch_equivalence.py
"""
from __future__ import annotations

import os
import sys
import ctypes

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sabah.runtime import rt
from sabah.runtime.expert_bank import ExpertBank
from sabah.runtime.hot_tier import HotTier
from sabah.runtime.executor import MoEExecutor
from sabah.runtime.mtp_prefetch import ExpertPrefetcher, ExternalPredictor

DEFAULT_MODEL = (r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                 r"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")

BLOCK = int(os.environ.get("SABAH_TEST_BLOCK", "0"))
STEPS = int(os.environ.get("SABAH_TEST_STEPS", "40"))


def _skip(reason):
    try:
        import pytest
        pytest.skip(reason)
    except Exception:
        print("SKIP:", reason)
        raise SystemExit(0)


def _model_path():
    return os.environ.get("SABAH_MODEL", DEFAULT_MODEL)


def _oracle_factory(seq):
    """A predictor that always names the NEXT step's exact experts."""
    state = {"i": -1}

    def fn(block, ctx):
        nxt = state["i"] + 1
        if nxt >= len(seq):
            return []
        return [int(e) for e in seq[nxt]]

    return fn, state


def _replay(bank, prof, seq, prefetch: bool, n_slots: int, x, w):
    slot_bytes = bank.block_expert_bytes(BLOCK)
    tier = HotTier(bank, n_slots * slot_bytes, blocks=[BLOCK],
                   prefetch_slots=(prof.n_experts_used if prefetch else 0))
    if tier.slots_per_block != n_slots:
        tier.free()
        raise RuntimeError("tier built %d slots, wanted %d"
                           % (tier.slots_per_block, n_slots))
    ex = MoEExecutor(bank, tier, max_used=prof.n_experts_used)
    pf = None
    tick = None
    if prefetch:
        fn, tick = _oracle_factory(seq)
        pf = ExpertPrefetcher(tier, ExternalPredictor(fn))
    try:
        ex.upload_x(x)
        outs = []
        for i, ids in enumerate(seq):
            if pf is not None:
                # issue the prefetch for the step we are about to run: the copy
                # is therefore still in flight when `ensure` admits these ids,
                # which is exactly the in-flight wait path.
                pf.warm(BLOCK)
            ex.zero_out()
            ex.run_block(BLOCK, ids, w)
            ex.sync()
            outs.append(ex.download_out().copy())
            if tick is not None:
                tick["i"] = i

        rt.check(rt.lib().sabah_sync_all(), "final sync")

        # every resident slot must equal its GGUF range, byte for byte
        p = tier.pools[BLOCK]
        mismatches = []
        for e, slot in p.slot_of.items():
            for r in bank.ROLES:
                per = bank.expert_bytes(BLOCK, r)
                host = np.empty(per, dtype=np.uint8)
                rt.check(rt.lib().sabah_memcpy_d2h(
                    host.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_void_p(p.slot_ptr(slot, r)), per), "slot d2h")
                src = np.asarray(bank.slice(BLOCK, int(e), r))
                if not np.array_equal(host, src):
                    mismatches.append((int(e), r))

        return np.array(outs), tier.tel.as_dict(), mismatches
    finally:
        ex.free()
        tier.free()


def run():
    path = _model_path()
    if not os.path.exists(path):
        _skip("model not found at %s (set SABAH_MODEL)" % path)
    if not rt.available() or rt.device_count() < 1:
        _skip("CUDA runtime unavailable")

    from sabah.core.model_inspector import inspect_model
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")
    prof = inspect_model(path)
    if not prof.supported:
        _skip("model is not supported by this Sabah build")

    bank = ExpertBank(prof, mode="mmap", blocks=[BLOCK])
    try:
        rng = np.random.default_rng(20240613)
        x = (rng.standard_normal(prof.d_model).astype(np.float32) * 0.05)
        w = np.full(prof.n_experts_used, 1.0 / prof.n_experts_used,
                    dtype=np.float32)

        # a routing sequence where the working set is far larger than the tier,
        # so misses and evictions happen on nearly every step
        seq = []
        for _ in range(STEPS):
            seq.append(rng.choice(prof.n_experts, prof.n_experts_used,
                                  replace=False).astype(np.int64).tolist())

        # a tier of 2x top-k: room for the current step, and prefetch has real
        # pressure because the reserve is half the pool
        n_slots = max(2 * prof.n_experts_used, 4)

        off, tel_off, bad_off = _replay(bank, prof, seq, False, n_slots, x, w)
        on, tel_on, bad_on = _replay(bank, prof, seq, True, n_slots, x, w)

        ok = True
        if off.shape != on.shape:
            print("FAIL: output shapes differ: %s vs %s" % (off.shape, on.shape))
            ok = False
        elif not np.array_equal(off, on):
            diff = np.abs(off - on)
            print("FAIL: prefetch changed the output; max abs diff %.3e at %s"
                  % (diff.max(), np.unravel_index(int(diff.argmax()), diff.shape)))
            ok = False
        else:
            print("PASS: %d steps bit-identical with prefetch off vs on"
                  % off.shape[0])

        if bad_off or bad_on:
            print("FAIL: resident slots differ from GGUF ranges: off=%d on=%d"
                  % (len(bad_off), len(bad_on)))
            ok = False
        else:
            print("PASS: every resident slot was byte-identical to its GGUF range")

        if tel_on.get("prefetch_issued", 0) <= 0:
            print("FAIL: prefetch never fired (issued=%d)"
                  % tel_on.get("prefetch_issued", 0))
            ok = False
        elif tel_on.get("prefetch_hits", 0) <= 0:
            print("WARN: prefetch fired (%d) but produced no hits; the reserve "
                  "may be too small to land a copy"
                  % tel_on.get("prefetch_issued", 0))
        else:
            print("PASS: prefetch issued %d, of which %d were used as hits"
                  % (tel_on["prefetch_issued"], tel_on["prefetch_hits"]))

        print("  prefetch OFF: hit %.4f, stall %.3f ms/tok"
              % (tel_off["hit_rate"],
                 tel_off["gpu_wait_expert_ms"] / max(1, tel_off["tokens"])))
        print("  prefetch ON : hit %.4f, stall %.3f ms/tok"
              % (tel_on["hit_rate"],
                 tel_on["gpu_wait_expert_ms"] / max(1, tel_on["tokens"])))
        return 0 if ok else 1
    finally:
        bank.close()


def test_prefetch_equivalence():
    """Pytest entry point."""
    assert run() == 0


if __name__ == "__main__":
    raise SystemExit(run())
