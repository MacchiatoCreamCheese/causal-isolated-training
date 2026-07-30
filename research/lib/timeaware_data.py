"""Generic causal negative sampling at cornac's data-loader layer.

This is the reusable form of Mechanism 1 (causal negative sampling). The
*definition* lives here once, on a `cornac.data.Dataset` subclass, and is
served to models through cornac's own negative-sampling entry points. A
negative item `j` for a `(user, pos)` interaction observed at time `t` may
only be drawn from items whose first appearance in the training data is
at-or-before `t`. Items that did not yet exist are never sampled, so the
model is never pushed to rank a not-yet-existent item below a positive.

Because the rule lives at the data layer, cornac's stock models pick it up
with no model-side code:

  - **LightGCN** (`cornac/models/lightgcn/recom_lightgcn.py`) trains from
    `train_set.uij_iter(...)` → served by `uij_iter` below.
  - **NeuMF / GMF / MLP** (`cornac/models/ncf/recom_ncf_base.py`) train from
    `train_set.uir_iter(..., num_zeros=num_neg)` → served by `uir_iter` below.
    cornac's `uir_iter` has no `neg_sampling` parameter at all; it draws
    negatives inline, so we override the whole method.

Neither caller passes a sampling mode, so the arm is selected by the
dataset's own `neg_sampling` attribute ("causal" or "uniform") rather than by
an argument. That attribute is what the ablation toggles.

The one model that cannot be served this way is cornac's BPR: it trains in
compiled Cython over `train_set.matrix` (a CSR carrying no timestamps) and
never calls either iterator. Our NumPy BPR in `bpr_cpu.py` covers that case
via `causal_negative_sampler.NumpyCausalSampler`, reading the *same* arrays
computed here, so there is a single source of truth for what "causal" means.

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

NEG_SAMPLING_MODES = ("causal", "uniform")


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
    """`size` uniform item indices over the whole catalog (vanilla BPR).

    Accepts either a modern `numpy.random.Generator` (`.integers`, used by our
    BPR) or the legacy `numpy.random.RandomState` that cornac's `Dataset.rng`
    is (`.randint`). Dispatching on the method keeps the Generator draw
    sequence bit-identical to what it was before the dataset became a caller.
    """
    draw = getattr(rng, "integers", None)
    if draw is None:
        draw = rng.randint
    return draw(0, num_items, size=size)


def causal_draw(sorted_first_seen, sorted_items, pos_timestamps, rng):
    """Vectorized causal negatives, one per element of `pos_timestamps`.

    Items are pre-sorted by first-seen. For a positive observed at time `t`,
    the eligible items are exactly the prefix `sorted_items[:L]` where
    `L = searchsorted(sorted_first_seen, t, side="right")`; we draw uniformly
    from that prefix.

    Like vanilla BPR (Rendle 2009) — and unlike cornac's base `uij_iter` /
    `uir_iter` — this does *not* reject a drawn item that the user has
    actually observed; the collision probability is ~O(interactions/items)
    (<0.1% on our datasets) and rejecting it would break bit-parity with the
    original research sampler. The guarantee this sampler makes is the causal
    one: a drawn negative never post-dates the positive.

    `rng.random` is used rather than an integer draw so the same code path
    works for both `Generator` and `RandomState`.
    """
    pos_timestamps = np.asarray(pos_timestamps, dtype=np.int64)
    prefix_len = np.searchsorted(sorted_first_seen, pos_timestamps, side="right")
    prefix_len = np.maximum(prefix_len, 1)
    idx_in_prefix = (rng.random(len(pos_timestamps)) * prefix_len).astype(np.int64)
    return sorted_items[idx_in_prefix]


class TimeAwareDataset(Dataset):
    """`cornac.data.Dataset` that knows each item's first-seen timestamp and
    serves causal negatives through both `uij_iter` and `uir_iter`.

    Construct via `TimeAwareDataset.from_dataset(existing_dataset)`.

    `neg_sampling` ("causal" | "uniform") selects the arm for callers that do
    not pass a mode — which is every cornac model, since none of them expose
    the option. Both arms use the same no-rejection draw, so the only
    difference between them is the item pool: the full catalog versus the
    causal prefix. That keeps the ablation a one-variable comparison.
    """

    def _init_causal_index(self, neg_sampling="causal"):
        if self.timestamps is None:
            raise ValueError(
                "TimeAwareDataset requires timestamps (UIRT data) for causal "
                "negative sampling."
            )
        self.neg_sampling = _check_mode(neg_sampling)
        self.item_first_seen = compute_item_first_seen(
            self.uir_tuple[1], self.timestamps, self.num_items
        )
        order = np.argsort(self.item_first_seen, kind="stable")
        self.sorted_item_order = order.astype(np.int64)
        self.sorted_first_seen = self.item_first_seen[order]
        self._ts_array = np.asarray(self.timestamps, dtype=np.int64)
        self.reset_counterfactual_counters()

    @classmethod
    def from_dataset(cls, dataset, neg_sampling="causal"):
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
        obj._init_causal_index(neg_sampling)
        return obj

    # -- Faithfulness probe -------------------------------------------------
    # Counterfactual-negative rate: the fraction of drawn negatives whose
    # first-seen timestamp post-dates their paired positive. Zero by
    # construction under causal sampling; the uniform arm is what it measures.
    # It lives here rather than on the model because cornac's models have no
    # idea negatives are being sampled for them.

    def reset_counterfactual_counters(self):
        """Zero the probe. Call before each `fit()` — otherwise counts
        accumulate across the cells of a multi-recipe run."""
        self._counterfactual_negs = 0
        self._total_negs = 0

    @property
    def counterfactual_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._counterfactual_negs / self._total_negs

    def _draw_negatives(self, batch_ts):
        """One negative per element of `batch_ts`, under the active arm,
        updating the counterfactual counters."""
        if self.neg_sampling == "uniform":
            negs = uniform_draw(self.num_items, len(batch_ts), self.rng)
            self._counterfactual_negs += int(
                (self.item_first_seen[negs] > batch_ts).sum()
            )
        else:
            negs = causal_draw(
                self.sorted_first_seen, self.sorted_item_order, batch_ts, self.rng
            )
        self._total_negs += len(batch_ts)
        return negs

    # -- cornac iterator overrides -----------------------------------------

    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling=None):
        """As `Dataset.uij_iter`, but the default mode comes from the dataset.

        `neg_sampling=None` (what cornac's LightGCN effectively passes, since
        it omits the argument) means "use `self.neg_sampling`". `popularity`
        is delegated to cornac's base implementation unchanged.
        """
        mode = self.neg_sampling if neg_sampling is None else neg_sampling.lower()
        if mode not in NEG_SAMPLING_MODES:
            yield from super().uij_iter(batch_size, shuffle, mode)
            return

        prev, self.neg_sampling = self.neg_sampling, mode
        try:
            for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
                batch_users = self.uir_tuple[0][batch_ids]
                batch_pos_items = self.uir_tuple[1][batch_ids]
                batch_neg_items = self._draw_negatives(self._ts_array[batch_ids])
                yield batch_users, batch_pos_items, batch_neg_items
        finally:
            self.neg_sampling = prev

    def uir_iter(self, batch_size=1, shuffle=False, binary=False, num_zeros=0):
        """As `Dataset.uir_iter`, with time-aware negatives.

        This is the entry point cornac's NCF family uses
        (`recom_ncf_base.py`: `uir_iter(..., binary=True, num_zeros=num_neg)`).
        The yielded layout matches cornac's exactly — `num_zeros` negatives per
        positive, appended after the positives with rating 0 — so the models
        need no changes. Only the choice of negative differs, and the draw is
        vectorized instead of cornac's per-element `randint` + rejection loop.
        """
        if num_zeros <= 0:
            yield from super().uir_iter(batch_size, shuffle, binary, num_zeros)
            return

        for batch_ids in self.idx_iter(len(self.uir_tuple[0]), batch_size, shuffle):
            batch_users = self.uir_tuple[0][batch_ids]
            batch_items = self.uir_tuple[1][batch_ids]
            if binary:
                batch_ratings = np.ones_like(batch_items)
            else:
                batch_ratings = self.uir_tuple[2][batch_ids]

            repeated_users = batch_users.repeat(num_zeros)
            repeated_ts = self._ts_array[batch_ids].repeat(num_zeros)
            neg_items = self._draw_negatives(repeated_ts).astype(batch_items.dtype)

            yield (
                np.concatenate((batch_users, repeated_users)),
                np.concatenate((batch_items, neg_items)),
                np.concatenate((batch_ratings, np.zeros_like(neg_items))),
            )


def _check_mode(neg_sampling):
    mode = str(neg_sampling).lower()
    if mode not in NEG_SAMPLING_MODES:
        raise ValueError(
            "neg_sampling must be one of {}, got {!r}".format(
                NEG_SAMPLING_MODES, neg_sampling
            )
        )
    return mode


class CausalTimestampSplit(TimestampSplit):
    """`TimestampSplit` whose training split is a `TimeAwareDataset`.

    `neg_sampling` ("causal" | "uniform") is stamped onto that split, which is
    how a runner picks the ablation arm without touching the model.
    """

    def __init__(self, *args, neg_sampling="causal", **kwargs):
        # Set before super().__init__, which ends by calling _split() -> build().
        self.neg_sampling = _check_mode(neg_sampling)
        super().__init__(*args, **kwargs)

    def build(self, train_data, val_data=None, test_data=None, **kwargs):
        super().build(
            train_data=train_data, val_data=val_data, test_data=test_data, **kwargs
        )
        if self.train_set is not None:
            self.train_set = TimeAwareDataset.from_dataset(
                self.train_set, neg_sampling=self.neg_sampling
            )
        return self
