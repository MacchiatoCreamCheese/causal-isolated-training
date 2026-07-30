"""Generic causal negative sampling at cornac's data-loader layer.

This is the reusable form of Mechanism 1 (causal negative sampling). Instead
of every model re-deriving per-item first-seen timestamps and hand-rolling a
sampler in its own training loop, the *definition* lives here once, on a
`cornac.data.Dataset` subclass, and is exposed through cornac's own negative
sampling entry point `Dataset.uij_iter(..., neg_sampling=...)`.

cornac already ships `neg_sampling ∈ {"uniform", "popularity"}`
(`cornac/data/dataset.py`). `TimeAwareDataset` adds a third option,
`"causal"`: a negative item `j` for a `(user, pos)` interaction observed at
time `t` may only be drawn from items whose first appearance in the training
data is at-or-before `t`. Items that did not yet exist are never sampled, so
the model is never pushed to rank a not-yet-existent item below a positive.

Any cornac model that trains by consuming `train_set.uij_iter(...)` gets
causal sampling for free — it is not welded into BPR/NeuMF/LightGCN. Our own
GPU models keep an on-device sampler for speed (see
`causal_negative_sampler.py`) but read the *same* arrays computed here, so
there is a single source of truth for what "causal" means.

`CausalTimestampSplit` is a drop-in `TimestampSplit` whose training split is a
`TimeAwareDataset`; validation/test splits stay stock (negatives are only
drawn during training).
"""

import numpy as np

from cornac.data import Dataset
from cornac.eval_methods import TimestampSplit


# Sentinel first-seen for items that never appear in training. It is larger
# than any real timestamp, so such items sit at the end of the sorted order
# and are never included in a causal prefix.
_NEVER_SEEN = np.iinfo(np.int64).max


def compute_item_first_seen(item_indices, timestamps, num_items):
    """Per-item earliest training timestamp, aligned to item index.

    Returns an int64 array of length `num_items`; items absent from the given
    interactions get `_NEVER_SEEN`.
    """
    first_seen = np.full(num_items, _NEVER_SEEN, dtype=np.int64)
    np.minimum.at(first_seen, np.asarray(item_indices, dtype=np.int64),
                  np.asarray(timestamps, dtype=np.int64))
    return first_seen


def uniform_draw(num_items, size, rng):
    """`size` uniform item indices over the whole catalog (vanilla BPR)."""
    return rng.integers(0, num_items, size=size)


def causal_draw(sorted_first_seen, sorted_items, pos_timestamps, rng):
    """Vectorized causal negatives, one per element of `pos_timestamps`.

    Items are pre-sorted by first-seen. For a positive observed at time `t`,
    the eligible items are exactly the prefix `sorted_items[:L]` where
    `L = searchsorted(sorted_first_seen, t, side="right")`; we draw uniformly
    from that prefix. This is the numpy twin of the on-device sampler in
    `causal_negative_sampler.TorchCausalSampler`.

    Like vanilla BPR (Rendle 2009) and unlike cornac's base `uij_iter`, this
    does *not* reject a drawn item that the user has actually observed; the
    collision probability is ~O(interactions/items) (<0.1% on our datasets) and
    rejecting it would break bit-parity with the original research sampler. The
    guarantee this sampler makes is the causal one: a drawn negative never
    post-dates the positive.
    """
    pos_timestamps = np.asarray(pos_timestamps, dtype=np.int64)
    prefix_len = np.searchsorted(sorted_first_seen, pos_timestamps, side="right")
    prefix_len = np.maximum(prefix_len, 1)
    idx_in_prefix = (rng.random(len(pos_timestamps)) * prefix_len).astype(np.int64)
    return sorted_items[idx_in_prefix]


class TimeAwareDataset(Dataset):
    """`cornac.data.Dataset` that also knows each item's first-seen timestamp
    and supports `neg_sampling="causal"` in `uij_iter`.

    Construct via `TimeAwareDataset.from_dataset(existing_dataset)`.
    """

    def _init_causal_index(self):
        if self.timestamps is None:
            raise ValueError(
                "TimeAwareDataset requires timestamps (UIRT data) for causal "
                "negative sampling."
            )
        self.item_first_seen = compute_item_first_seen(
            self.uir_tuple[1], self.timestamps, self.num_items
        )
        order = np.argsort(self.item_first_seen, kind="stable")
        self.sorted_item_order = order.astype(np.int64)
        self.sorted_first_seen = self.item_first_seen[order]

    @classmethod
    def from_dataset(cls, dataset):
        """Wrap an already-built `Dataset` (same interactions, maps, seed),
        adding the causal index. Only interaction data is carried over, which
        is all a UIRT training split needs."""
        obj = cls(
            num_users=dataset.num_users,
            num_items=dataset.num_items,
            uid_map=dataset.uid_map,
            iid_map=dataset.iid_map,
            uir_tuple=dataset.uir_tuple,
            timestamps=dataset.timestamps,
            seed=dataset.seed,
        )
        obj._init_causal_index()
        return obj

    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling="uniform"):
        """As `Dataset.uij_iter`, with an added `neg_sampling="causal"` option.

        Yields `(batch_users, batch_pos_items, batch_neg_items)`. `uniform` and
        `popularity` are delegated to cornac's base implementation unchanged.
        """
        if neg_sampling.lower() != "causal":
            yield from super().uij_iter(batch_size, shuffle, neg_sampling)
            return

        timestamps = np.asarray(self.timestamps, dtype=np.int64)
        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_pos_items = self.uir_tuple[1][batch_ids]
            batch_ts = timestamps[batch_ids]
            batch_neg_items = causal_draw(
                self.sorted_first_seen, self.sorted_item_order, batch_ts, self.rng
            )
            yield batch_users, batch_pos_items, batch_neg_items


class CausalTimestampSplit(TimestampSplit):
    """`TimestampSplit` whose training split is a `TimeAwareDataset`, so any
    consumer can request `train_set.uij_iter(neg_sampling="causal")`."""

    def build(self, train_data, val_data=None, test_data=None, **kwargs):
        super().build(
            train_data=train_data, val_data=val_data, test_data=test_data, **kwargs
        )
        if self.train_set is not None:
            self.train_set = TimeAwareDataset.from_dataset(self.train_set)
        return self
