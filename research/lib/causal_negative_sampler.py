"""Per-fit negative-sampling executor for our BPR backbone.

`TimeAwareDataset` (see `timeaware_data.py`) owns the *definition* of causal
sampling — per-item first-seen and the sorted causal index — and serves it to
cornac's own models through `uij_iter` / `uir_iter`. This module is the
*execution* path for the one model that never asks the data loader for
negatives: our NumPy BPR (`bpr_cpu.py`), which builds a sampler in `fit()` and
calls `.sample(pos_ts, num_neg)`.

The sampler also tracks the counterfactual rate (fraction of drawn negatives
that did not yet exist — a Mechanism-1 faithfulness probe, ~0 for the causal
sampler by construction). For cornac's models the same probe is read off the
dataset instead; see `TimeAwareDataset.counterfactual_rate`.

If handed a plain `Dataset` (not a `TimeAwareDataset`), the index is computed
on the fly, so the model still works under a stock `TimestampSplit`.
"""

import numpy as np

from .timeaware_data import (
    build_observed_keys,
    causal_draw,
    compute_item_first_seen,
    reject_collisions,
    uniform_draw,
)


def ensure_causal_arrays(train_set):
    """Return `(item_first_seen, sorted_first_seen, sorted_item_order)` for a
    training set, reading them off a `TimeAwareDataset` or computing them from
    a plain `Dataset`."""
    ifs = getattr(train_set, "item_first_seen", None)
    if ifs is not None:
        return (np.asarray(train_set.item_first_seen),
                np.asarray(train_set.sorted_first_seen),
                np.asarray(train_set.sorted_item_order))
    ifs = compute_item_first_seen(
        train_set.uir_tuple[1], train_set.timestamps, train_set.num_items
    )
    order = np.argsort(ifs, kind="stable")
    return ifs, ifs[order], order.astype(np.int64)


class NumpyCausalSampler:
    """NumPy negative sampler (`sampler_kind` ∈ {"uniform", "causal"})."""

    def __init__(self, train_set, sampler_kind, rng):
        self.kind = sampler_kind
        self.rng = rng
        (self.item_first_seen,
         self.sorted_first_seen,
         self.sorted_item_order) = ensure_causal_arrays(train_set)
        self.num_items = len(self.item_first_seen)
        self.observed_keys = getattr(train_set, "observed_keys", None)
        if self.observed_keys is None:
            self.observed_keys = build_observed_keys(
                train_set.uir_tuple[0], train_set.uir_tuple[1], self.num_items
            )
        self._counterfactual_negs = 0
        self._total_negs = 0
        self._residual_collisions = 0

    def sample(self, pos_ts, users, num_neg=1):
        """`users` is required: rejecting collisions needs to know whose
        positives a drawn negative might be. Both arms reject identically, so
        they still differ only in the item pool."""
        pos_ts = np.asarray(pos_ts, dtype=np.int64)
        users = np.asarray(users, dtype=np.int64)
        m = len(pos_ts)
        ts_rep = pos_ts if num_neg == 1 else np.repeat(pos_ts, num_neg)
        u_rep = users if num_neg == 1 else np.repeat(users, num_neg)
        uniform = self.kind == "uniform"

        def draw(n, ts):
            if uniform:
                return uniform_draw(self.num_items, n, self.rng)
            return causal_draw(self.sorted_first_seen, self.sorted_item_order,
                               ts, self.rng)

        negs = draw(len(ts_rep), ts_rep)
        negs, residual = reject_collisions(
            u_rep, negs, self.observed_keys, self.num_items,
            redraw=lambda mask: draw(int(mask.sum()), ts_rep[mask]),
        )
        self._residual_collisions += residual
        if uniform:
            self._counterfactual_negs += int((self.item_first_seen[negs] > ts_rep).sum())
        self._total_negs += len(ts_rep)
        return negs if num_neg == 1 else negs.reshape(m, num_neg)

    @property
    def counterfactual_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._counterfactual_negs / self._total_negs

    @property
    def collision_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._residual_collisions / self._total_negs
