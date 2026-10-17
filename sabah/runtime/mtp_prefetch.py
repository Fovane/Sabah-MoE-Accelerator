"""
Sabah / runtime / mtp_prefetch

Cross-token expert prefetch, driven by the model's own prediction head.

Why this is the only prefetch that can work
-------------------------------------------
Within one token, block L's router consumes block L-1's output, so the experts
for L are not known until L-1 has run. There is nothing to prefetch there -- it
is a chain. What the Qwen3.8-Flash-Next family adds is an MTP / Prediction Head:
it predicts the NEXT token from the current hidden state. If that prediction is
good enough to guess which experts token t+1 will route to, those experts can be
fetched while token t is still computing, and the transfer latency stops sitting
on the critical path.

The contract
------------
Prefetch is ADVICE. A predictor may be wrong; it may return nothing; it may be
turned off. None of that changes a single output value:

  * `ensure` still admits exactly the ids the router produced;
  * a prefetched expert the router did not ask for is simply evicted later;
  * a prefetched expert the router DID ask for is used only after its copy is
    awaited on the compute stream (see `HotTier._ensure_locked`);
  * a prefetch never evicts a slot that live work may read.

So the worst a bad predictor can do is waste bandwidth, never corrupt a token.

This module is the scheduling skeleton. Three predictors are provided:
`NullPredictor` (baseline, no prefetch), `FrequencyPredictor` (recent routing
frequency per block -- a real, if weak, signal) and `ExternalPredictor` (a hook
for candidate ids produced outside, e.g. by an MTP head exposed through the
bridge). The MTP head itself lives in the model/llama.cpp, not here.
"""
from __future__ import annotations

import collections
from typing import Callable, Dict, Iterable, List, Optional, Sequence


class Predictor:
    """Maps a block (and optional context) to candidate expert ids."""

    def observe(self, block: int, ids: Sequence[int]) -> None:
        """Record the experts the router actually chose for `block`."""

    def candidates(self, block: int, context=None) -> List[int]:
        """Return expert ids worth prefetching for `block`. May be empty."""
        return []

    def reset(self) -> None:
        pass


class NullPredictor(Predictor):
    """No prediction. Used to measure the cost of prefetch being off."""

    def candidates(self, block: int, context=None) -> List[int]:
        return []


class FrequencyPredictor(Predictor):
    """Prefetch the experts most often routed recently, per block.

    Not an oracle: this is the same signal a static hot set uses, so it mainly
    helps the first tokens after a context switch. It is cheap, needs no MTP
    head, and makes the plumbing testable on any trace.
    """

    def __init__(self, top_k: int, blocks: Optional[Iterable[int]] = None,
                 decay: float = 0.0):
        self.top_k = int(top_k)
        self.decay = float(decay)
        self._counts: Dict[int, collections.Counter] = {}
        if blocks is not None:
            for b in blocks:
                self._counts[int(b)] = collections.Counter()

    def observe(self, block: int, ids: Sequence[int]) -> None:
        c = self._counts.setdefault(int(block), collections.Counter())
        if self.decay > 0:
            for k in list(c):
                c[k] *= (1.0 - self.decay)
        for e in ids:
            c[int(e)] += 1

    def candidates(self, block: int, context=None) -> List[int]:
        c = self._counts.get(int(block))
        if not c:
            return []
        return [e for e, _ in c.most_common(self.top_k)]

    def reset(self) -> None:
        for c in self._counts.values():
            c.clear()


class ExternalPredictor(Predictor):
    """Candidates supplied from outside -- the hook for an MTP head.

    `fn(block, context) -> Sequence[int]`. `context` is whatever the caller has
    (e.g. the predicted next token's hidden state, or the ids an MTP head
    proposed). This is the seam where a real multi-token-prediction signal
    plugs in without changing anything else.
    """

    def __init__(self, fn: Callable[[int, object], Sequence[int]],
                 top_k: Optional[int] = None):
        self.fn = fn
        self.top_k = top_k
        self.observe = lambda block, ids: None   # external source learns nothing

    def candidates(self, block: int, context=None) -> List[int]:
        out = [int(x) for x in (self.fn(int(block), context) or [])]
        return out[:self.top_k] if self.top_k else out


class ExpertPrefetcher:
    """Drives `HotTier.prefetch` from a predictor and reports what it bought."""

    def __init__(self, tier, predictor: Optional[Predictor] = None,
                 enabled: bool = True):
        self.tier = tier
        self.predictor = predictor or NullPredictor()
        self.enabled = bool(enabled)
        self.calls = 0
        self.issued = 0

    def observe(self, block: int, ids: Sequence[int]) -> None:
        self.predictor.observe(block, ids)

    def warm(self, block: int, context=None) -> int:
        if not self.enabled:
            return 0
        cand = self.predictor.candidates(block, context)
        self.calls += 1
        if not cand:
            return 0
        n = self.tier.prefetch(block, cand)
        self.issued += n
        return n

    def warm_ahead(self, block: int, context=None) -> int:
        """Warm `block` for the NEXT token.

        Same block index: token t+1 re-enters the same 48 blocks, so block L of
        the next token is the thing worth warming while token t finishes.
        """
        return self.warm(block, context)

    def stats(self) -> dict:
        return dict(enabled=self.enabled, calls=self.calls, issued=self.issued,
                    prefetch_issued=self.tier.tel.prefetch_issued,
                    prefetch_hits=self.tier.tel.prefetch_hits,
                    predictor=type(self.predictor).__name__)


def run_blocks_with_prefetch(executor, prefetcher: ExpertPrefetcher,
                             x, block_plan):
    """Run one token's blocks, warming the next token's experts behind them.

    `block_plan` is [(block, ids, weights), ...]; `x` is the d_model input.
    After each block runs, the prefetcher is asked for that same block's
    candidates, and any reserved-and-free slots are filled. Those copies ride
    the copy stream while the remaining blocks compute, which is the point.

    The output is identical to `executor.run_model(x, block_plan)`: prefetch
    changes timing, never values.
    """
    executor.upload_x(x)
    for block, ids, weights in block_plan:
        executor.run_block(block, ids, weights, accumulate=False)
        prefetcher.observe(block, ids)
        prefetcher.warm_ahead(block)
    return executor.download_out()
