"""Compare future_items_pct (per-user loop) vs future_items_pct_batched.

Uses a stub model with a deterministic score function so we can construct
a scenario where top-K contains items with future first-seen timestamps,
making the future-item count non-zero. Both functions should return
identical integer counts.
"""

import sys

import numpy as np

import cornac
from cornac.eval_methods import TimestampSplit
from cornac.models.recommender import Recommender

from ..lib.causal_sampling import (
    future_items_pct, future_items_pct_batched,
)


class StubModel(Recommender):
    """score(u) returns a deterministic per-(u,i) vector."""

    def __init__(self, num_items):
        super().__init__(name="stub", trainable=False, verbose=False)
        self._num_items = num_items
        # Deterministic scores: distinct per (u, i) so no ties.
        rng = np.random.default_rng(7)
        self._S = rng.standard_normal((10000, num_items)).astype(np.float32)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        return self

    def score(self, user_idx, item_idx=None):
        s = self._S[user_idx % self._S.shape[0], :self._num_items]
        if item_idx is None:
            return s
        return float(s[item_idx])


K = 10


def build_case():
    """Synthetic UIRT log plus a stub model, as `(model, eval_method, item_first)`.

    200 users over 100 items in three time blocks, so later items have a larger
    first-seen and can look "future" to an earlier test timestamp — without that
    the future-item count is zero and comparing the two implementations proves
    nothing.
    """
    rng = np.random.default_rng(0)
    rows = []
    for t_block in range(3):
        base_ts = 1_600_000_000_000 + t_block * 10_000_000_000  # ~115d apart
        items_avail = (t_block + 1) * 30  # block 0: 30 items, block 1: 60, block 2: 90
        for _ in range(2000):
            u = int(rng.integers(0, 200))
            i = int(rng.integers(0, items_avail))
            ts = base_ts + int(rng.integers(0, 5_000_000_000))
            rows.append((f"u{u}", f"i{i}", 1.0, ts))

    item_first = {}
    for u, i, _, ts in rows:
        if i not in item_first or ts < item_first[i]:
            item_first[i] = ts

    eval_method = TimestampSplit(
        data=rows,
        val_timestamp=1_605_000_000_000,
        test_timestamp=1_615_000_000_000,
        fmt="UIRT",
        exclude_unknowns=False,
        verbose=False,
    )

    model = StubModel(num_items=eval_method.test_set.num_items)
    model.fit(eval_method.train_set, eval_method.val_set)
    return model, eval_method, item_first


def compare(k=K, chunk=64):
    """Run both implementations over the same case; returns `(old, new)`."""
    model, eval_method, item_first = build_case()
    old = future_items_pct(model, eval_method, item_first, k=k)
    new = future_items_pct_batched(model, eval_method, item_first, k=k, chunk=chunk)
    return old, new


def agree(old, new):
    return (
        old["evaluated"] == new["evaluated"]
        and abs(old["global_pct"] - new["global_pct"]) < 1e-9
        and abs(old["mean_pct_per_instance"] - new["mean_pct_per_instance"]) < 1e-9
    )


def main():
    old, new = compare()
    print("OLD:", old)
    print("NEW:", new)
    ok = agree(old, new)
    print("MATCH:", ok)
    if old["global_pct"] == 0.0:
        print("(test inconclusive: global_pct is 0)")
    sys.exit(0 if ok else 1)


# ---------------------------------------------------------------------------
# pytest entry point. See the note in test_cornac_causal.py — no import-time
# pytest dependency, and `python -m` keeps working unchanged.
# ---------------------------------------------------------------------------

def test_batched_matches_per_user_loop():
    old, new = compare()
    assert old["global_pct"] > 0.0, (
        "degenerate case: no top-K recommendation is a future item, so the "
        "comparison below would hold trivially for both implementations")
    assert agree(old, new), f"OLD={old} NEW={new}"


if __name__ == "__main__":
    main()
