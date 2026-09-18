"""
End-to-end Phase B chain, on real weights:

    real GGUF -> RAM/mmap expert bank -> VRAM hot tier -> GPU expert execution
                 -> compared against a CPU reference

Three things are checked, in increasing strictness:

  1. AGREEMENT  the GPU block matches a numpy reference built from gguf-py's
                dequantizer, to fp32 reduction tolerance.
  2. DISCRIMINATION  the same comparison, with ONE routed expert swapped for a
                different one, must fail loudly. A test that cannot detect a
                wrong expert cannot certify a right one.
  3. RESIDENCE INDEPENDENCE  the same block, same ids, run once with the
                experts preloaded and once after eviction forced a refetch,
                must produce bit-identical output. Sabah's caching is only
                legitimate if residency cannot change the answer.
"""
from __future__ import annotations

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from sabah.core.model_inspector import inspect_model
from sabah.runtime import rt
from sabah.runtime.expert_bank import ExpertBank
from sabah.runtime.hot_tier import HotTier
from sabah.runtime.executor import MoEExecutor, reference_block

DEFAULT_MODEL = (r"D:/lmstudio/models/qwen/qwen3.8-Flash-Next-MoE/"
                 r"Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf")

# blocks chosen to cover the quant mix: 0 is Q4_K/Q5_1, 2 is Q5_K/Q8_0
TEST_BLOCKS = [0, 2]
TOPK = 10
SLOTS = 24
TOL_REL_L2 = 1e-5


def rel_l2(a, b):
    d = np.linalg.norm(a - b)
    n = np.linalg.norm(b)
    return float(d / n) if n > 0 else float(d)


def main(model_path=None):
    model_path = model_path or DEFAULT_MODEL
    print("=" * 78)
    print("SABAH Phase B - exact expert execution on real weights")
    print("=" * 78)

    if not rt.available() or rt.device_count() < 1:
        print("CUDA runtime unavailable.")
        return 2
    rt.check(rt.lib().sabah_rt_init(0), "rt_init")

    try:
        prof = inspect_model(model_path)
    except Exception as e:
        print("cannot inspect %s: %s" % (model_path, e))
        return 2
    if not prof.supported:
        print("unsupported model"); return 2

    bank = ExpertBank(prof, mode="mmap", blocks=TEST_BLOCKS)
    print(bank.describe())
    print()

    cap = SLOTS * max(bank.block_expert_bytes(b) for b in TEST_BLOCKS) * len(TEST_BLOCKS)
    tier = HotTier(bank, cap, blocks=TEST_BLOCKS)
    print(tier.describe())
    print()

    ex = MoEExecutor(bank, tier, max_used=TOPK)
    rng = np.random.default_rng(20260918)
    failures = []

    print("%-6s %-9s %-11s %11s   %s"
          % ("block", "quant", "check", "rel L2", "verdict"))
    print("-" * 64)

    for blk in TEST_BLOCKS:
        qmix = "%s/%s" % (bank.desc[(blk, "gate")]["qtype"],
                          bank.desc[(blk, "down")]["qtype"])
        x = rng.standard_normal(prof.d_model).astype(np.float32) * 0.05
        ids = sorted(rng.choice(prof.n_experts, TOPK, replace=False).tolist())
        w = rng.random(TOPK).astype(np.float32)
        w /= w.sum()

        ref = reference_block(bank, blk, x, ids, w)

        # ---- 1. agreement ------------------------------------------------
        ex.upload_x(x)
        ex.run_block(blk, ids, w, accumulate=False)
        ex.sync()
        got = ex.download_out()
        r = rel_l2(got, ref)
        ok = r < TOL_REL_L2
        if not ok:
            failures.append("block %d agreement rel_l2=%.3e" % (blk, r))
        print("%-6d %-9s %-11s %11.3e   %s"
              % (blk, qmix, "agreement", r, "OK" if ok else "FAIL"))

        # ---- 2. discrimination -------------------------------------------
        wrong = list(ids)
        alt = next(e for e in range(prof.n_experts) if e not in ids)
        wrong[0] = alt
        ex.upload_x(x)
        ex.run_block(blk, wrong, w, accumulate=False)
        ex.sync()
        got_wrong = ex.download_out()
        rw = rel_l2(got_wrong, ref)
        discriminating = rw > TOL_REL_L2 * 100
        if not discriminating:
            failures.append("block %d: swapping an expert changed nothing "
                            "(rel_l2=%.3e) - the test is blind" % (blk, rw))
        print("%-6d %-9s %-11s %11.3e   %s"
              % (blk, qmix, "wrong-expert", rw,
                 "detected" if discriminating else "BLIND"))

        # ---- 3. residence independence ------------------------------------
        # force every routed expert out of VRAM, then rerun the same request
        p = tier.pools[blk]
        filler = [e for e in range(prof.n_experts) if e not in ids][:p.n_slots]
        for e in list(p.slot_of):
            p.state[e] = "RAM_ONLY"
            s = p.slot_of.pop(e); p.expert_of.pop(s, None); p.lru.pop(e, None)
            p.free_slots.append(s)
        tier.preload(blk, filler)
        misses_before = tier.tel.misses
        ex.upload_x(x)
        ex.run_block(blk, ids, w, accumulate=False)
        ex.sync()
        got2 = ex.download_out()
        refetched = tier.tel.misses - misses_before
        identical = np.array_equal(got, got2)
        if not identical:
            failures.append("block %d: output changed after eviction/refetch" % blk)
        print("%-6d %-9s %-11s %11s   %s (%d refetched)"
              % (blk, qmix, "residency", "bit-exact" if identical else "DIFFERS",
                 "OK" if identical else "FAIL", refetched))

    tier.drain_stalls()
    print()
    print(tier.tel.summary())
    print()

    ex.free(); tier.free(); bank.close()

    if failures:
        print("FAILURES:")
        for f in failures:
            print("  - " + f)
        return 1
    print("PASSED - the GPU executes the routed experts exactly, the test can "
          "tell a wrong expert from a right one, and residency does not affect "
          "the result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
