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

Collision rejection is *per model*, matching cornac
--------------------------------------------------

Restricting the item pool is only half of what a negative sampler does; the
other half is refusing to hand back an item the user already interacted with.
cornac has three different answers to that, and each is correct for the loss it
feeds:

  ===========================  ===============================  ==============
  cornac site                  Filter                           On collision
  ===========================  ===============================  ==============
  BPR, Cython                  any observed, rating ignored     skip the sample
  (`recom_bpr.pyx`,
  `has_non_zero`)

  `Dataset.uij_iter`           `dok[u,j] >= pos_rating`         redraw
  -> LightGCN                  (rating-aware)

  `Dataset.uir_iter`           `dok[u,j] > 0`                   redraw
  -> NeuMF / GMF / MLP
  ===========================  ===============================  ==============

The rule tracks the *loss*, not the dataset. `uij_iter` feeds a pairwise BPR
loss, which asserts only an ordering — so an item the user rated *below* the
positive is a legitimate, indeed stronger, negative: the preference is observed
rather than assumed. `uir_iter` feeds a pointwise binary cross-entropy, where
that same item would be handed to the model carrying a label of 0, which is
simply false. Hence rating-aware there, any-observed here.

This matters on explicit-rating data. `data.load_uirt` reads raw Amazon 1-5
stars with no binarization, and the ratings skew hard to 5, so the two rules
genuinely diverge: for a 5-star positive cornac admits the user's own 1-4 star
items as negatives. On implicit data (all ratings 1) `>= pos_rating` collapses
to `> 0` and the distinction vanishes.

So `_draw_negatives` takes `pos_ratings`: `uij_iter` supplies them and gets the
rating-aware rule, `uir_iter` omits them and gets any-observed. What must *not*
vary is the filter within a single model — see `_draw_negatives`.

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


# How many times to redraw a negative that collided with one of the user's own
# positives. Bounded on purpose: cornac's samplers use an unbounded `while`,
# which cannot be used here. The earliest interactions in the log have a causal
# prefix of exactly one item -- that item being the positive itself -- so for
# them no non-colliding negative exists and an unbounded loop would spin
# forever. After the last round we accept whatever remains and count it.
MAX_REJECT_ROUNDS = 4


def build_observed_index(users, items, ratings, num_items):
    """Sorted `(keys, ratings)` for every observed interaction.

    `keys` are `user * num_items + item`, sorted; `ratings` carries each key's
    rating under the same permutation, which is what lets the rating-aware
    filter answer "what did this user give item j?" with a single
    `searchsorted`. Together they are a vectorized stand-in for cornac's
    `dok_matrix` lookup, which is what keeps collision rejection a whole-batch
    operation instead of cornac's per-element Python loop.

    Duplicates need no reduction: `cornac.data.Dataset.build` already drops
    repeated `(uid, iid)` pairs (its `ui_set`, which is what the "N duplicated
    observations are removed!" warning reports), so each key appears once. The
    sort is nonetheless by key *then descending rating*, so that if a future
    cornac ever stopped deduplicating, `searchsorted`'s left-hand match would
    land on the highest rating for the pair -- the strictest, most-rejecting
    choice -- instead of an arbitrary one. That costs nothing here and fails
    safe there.

    Memory matters on the larger datasets (healthcare is ~7.2M rows): this
    holds two arrays of `n`, and `lexsort` a third. Deliberately not
    `np.unique(..., return_inverse=True)` plus `np.maximum.at`, which would
    allocate an extra index array and run a scatter-reduce that is far slower
    than the sort it follows.
    """
    keys = (np.asarray(users, dtype=np.int64) * np.int64(num_items)
            + np.asarray(items, dtype=np.int64))
    ratings = np.asarray(ratings, dtype=np.float64)
    order = np.lexsort((-ratings, keys))
    return keys[order], ratings[order]


def find_collisions(users, negs, observed_keys, num_items,
                    observed_ratings=None, pos_ratings=None):
    """Boolean mask: which drawn negatives must be rejected.

    With `observed_ratings`/`pos_ratings` omitted this is cornac's `uir_iter`
    rule -- reject any item the user has observed at all. Supply both and it
    becomes cornac's `uij_iter` rule: reject only where the user rated the drawn
    item at least as highly as the positive, so genuinely less-preferred items
    stay eligible. See the module docstring for why the two differ.
    """
    if observed_keys.size == 0:
        return np.zeros(len(negs), dtype=bool)
    keys = (np.asarray(users, dtype=np.int64) * np.int64(num_items)
            + np.asarray(negs, dtype=np.int64))
    idx = np.searchsorted(observed_keys, keys)
    np.clip(idx, 0, observed_keys.size - 1, out=idx)
    hit = observed_keys[idx] == keys
    if observed_ratings is None or pos_ratings is None:
        return hit
    # `hit` guards the clipped `idx`, so the gathered rating is only trusted
    # where the key genuinely matched.
    return hit & (observed_ratings[idx] >= np.asarray(pos_ratings))


def reject_collisions(users, negs, observed_keys, num_items, redraw,
                      observed_ratings=None, pos_ratings=None,
                      max_rounds=MAX_REJECT_ROUNDS):
    """Redraw negatives that collide with the user's own positives.

    `redraw(mask)` returns replacement items for the masked positions. Returns
    `(negs, n_residual)` where `n_residual` is how many collisions survived all
    rounds -- non-zero only where no valid negative exists (see the note on
    MAX_REJECT_ROUNDS).

    `observed_ratings`/`pos_ratings` select the filter and are forwarded
    unchanged to `find_collisions` on every round. They need no masking:
    `negs` and `pos_ratings` stay full-length throughout, and only `redraw`
    sees the subset.
    """
    negs = np.array(negs, dtype=np.int64, copy=True)
    kw = dict(observed_ratings=observed_ratings, pos_ratings=pos_ratings)
    hit = find_collisions(users, negs, observed_keys, num_items, **kw)
    for _ in range(max_rounds):
        if not hit.any():
            return negs, 0
        negs[hit] = redraw(hit)
        hit = find_collisions(users, negs, observed_keys, num_items, **kw)
    return negs, int(hit.sum())


def causal_draw(sorted_first_seen, sorted_items, pos_timestamps, rng):
    """Vectorized causal negatives, one per element of `pos_timestamps`.

    Items are pre-sorted by first-seen. For a positive observed at time `t`,
    the eligible items are exactly the prefix `sorted_items[:L]` where
    `L = searchsorted(sorted_first_seen, t, side="right")`; we draw uniformly
    from that prefix.

    The guarantee made here is the causal one: a drawn negative never
    post-dates the positive. It says nothing about whether the user has
    actually observed the drawn item -- that is handled separately by
    `reject_collisions`, applied identically to both sampling arms.

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
        self.observed_keys, self.observed_ratings = build_observed_index(
            self.uir_tuple[0], self.uir_tuple[1], self.uir_tuple[2],
            self.num_items,
        )
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
    # first-seen timestamp post-dates their paired positive. It lives here
    # rather than on the model because cornac's models have no idea negatives
    # are being sampled for them.
    #
    # Measured in **both** arms, though it is zero by construction under causal
    # sampling (a positive's own item is always in its causal prefix, so the
    # prefix is never empty and no future item is reachable). Skipping the
    # causal count looks like a free optimization and was how this started out,
    # but it makes the probe unfalsifiable: `rho == 0` would hold because the
    # counter never incremented, not because the draw was correct. Verified by
    # sabotage -- with `causal_draw` replaced by a uniform draw, the counting
    # version reports rho ~= 31% and four smoke checks fail; the gated version
    # reports 0% and passes.
    #
    # Cost is 1.5-2.9 ns per draw (a vectorized gather and compare, ~0.5% of
    # the sampling step, ~1s per 1000 epochs on musical), so gating it to save
    # time is a false economy.
    #
    # `_total_negs` is load-bearing too: `counterfactual_rate` returns 0.0 when
    # nothing was drawn, so callers asserting rho == 0 must also assert that
    # draws happened -- otherwise a bypassed loader passes. See
    # `smoke/test_cornac_causal.py:Results.check`.

    def reset_counterfactual_counters(self):
        """Zero the probes. Call before each `fit()` — otherwise counts
        accumulate across the cells of a multi-recipe run."""
        self._counterfactual_negs = 0
        self._total_negs = 0
        self._residual_collisions = 0

    @property
    def counterfactual_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._counterfactual_negs / self._total_negs

    @property
    def collision_rate(self):
        """Fraction of drawn negatives still rejected after all redraw rounds.

        Non-zero only where the causal prefix offers no alternative — see
        MAX_REJECT_ROUNDS. Note the denominator is arm- *and model*-specific:
        which draws count as collisions depends on which filter the calling
        iterator selected, so a LightGCN figure (rating-aware, rejects less) is
        not directly comparable to a NeuMF one (any-observed).
        """
        if self._total_negs == 0:
            return 0.0
        return self._residual_collisions / self._total_negs

    def _draw_negatives(self, users, batch_ts, pos_ratings=None):
        """One negative per element of `batch_ts`, under the active arm.

        `pos_ratings` selects which of cornac's rejection filters to apply:
        supplied (from `uij_iter`) gives the rating-aware `>= pos_rating` rule,
        omitted (from `uir_iter`) gives any-observed. See the module docstring.

        Whichever filter is in play, **both arms get the same one**, so
        `uniform` and `causal` still differ in exactly one thing: the item pool.
        Without that the causal arm would collide more often purely because its
        pool is smaller, which would confound the very comparison the ablation
        is making. Filters differing across *models* is harmless — no claim is
        made across models.
        """
        uniform = self.neg_sampling == "uniform"

        def draw(n, ts):
            if uniform:
                return uniform_draw(self.num_items, n, self.rng)
            return causal_draw(
                self.sorted_first_seen, self.sorted_item_order, ts, self.rng
            )

        negs = draw(len(batch_ts), batch_ts)
        negs, residual = reject_collisions(
            users, negs, self.observed_keys, self.num_items,
            redraw=lambda mask: draw(int(mask.sum()), batch_ts[mask]),
            observed_ratings=None if pos_ratings is None else self.observed_ratings,
            pos_ratings=pos_ratings,
        )
        self._residual_collisions += residual

        # rho is measured on the negatives actually used, after rejection, and
        # in both arms -- see the probe note above.
        self._counterfactual_negs += int(
            (self.item_first_seen[negs] > batch_ts).sum()
        )
        self._total_negs += len(batch_ts)
        return negs

    # -- cornac iterator overrides -----------------------------------------

    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling=None):
        """As `Dataset.uij_iter`, but the default mode comes from the dataset.

        `neg_sampling=None` (what cornac's LightGCN effectively passes, since
        it omits the argument) means "use `self.neg_sampling`". `popularity`
        is delegated to cornac's base implementation unchanged.

        Rejection here is rating-aware, matching cornac's own `uij_iter`: the
        batch's positive ratings are passed down to `_draw_negatives`.
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
                batch_neg_items = self._draw_negatives(
                    batch_users, self._ts_array[batch_ids],
                    pos_ratings=self.uir_tuple[2][batch_ids],
                )
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

        Rejection is any-observed, matching cornac's own `uir_iter`: no ratings
        are passed to `_draw_negatives`. A pointwise loss labels these rows 0,
        so an item the user did interact with would be a wrong label at any
        rating — unlike the pairwise case in `uij_iter` above.
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
            neg_items = self._draw_negatives(
                repeated_users, repeated_ts
            ).astype(batch_items.dtype)

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
