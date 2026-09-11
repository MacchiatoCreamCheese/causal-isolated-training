"""Mechanism 2: temporal mini-batching.

Mechanism 1 constrains *which items* may be negatives. Mechanism 2 constrains
*which rows share a gradient step*: a mini-batch is one contiguous band of the
training timeline instead of a random sample of it.

Why the batch and not the training phase
----------------------------------------

The earlier form of this mechanism (`lib/temporal_batching.py:BPRWindowed` on the
`main` branch) partitioned training into K windows and trained them one after
another::

    for w in range(n_windows):                 # K windows, sequentially
        for epoch in range(epochs_per_window):
            perm = rng.permutation(n_rows_in_window)   # shuffled *inside*
            ...
        # window w is never revisited

Two things are wrong with that. It **forgets**: each window is exhausted and
abandoned, so the finished model spent its last epochs entirely on the most
recent slice of the timeline and the early slices have been written over. And the
mini-batches are **not actually temporal** -- `perm` shuffles within the window,
so every gradient step still sees a random sample. The temporal structure lived
at the *phase* level while the mini-batch, which is what forms a gradient, stayed
time-blind.

GRU4Rec (Hidasi et al., ICLR 2016) §3.1.1 rejects the same shape for sessions: it
never exhausts one session before starting the next, and it makes the *mini-batch*
the unit that carries temporal meaning. Adopting that here means every epoch still
covers the whole timeline -- so nothing is forgotten -- while each individual
batch is temporally coherent.

Why this needs no model-side code
---------------------------------

`TimeAwareDataset.uij_iter` and `.uir_iter` both draw their batch indices from
`self.idx_iter(...)`. Overriding that single method therefore reaches **both**
entry points, and so both cornac models -- LightGCN through `uij_iter`, the NCF
family through `uir_iter` -- with nothing changed inside either. That is the same
property Mechanism 1 relies on, and it is what the `BPRWindowed` version could
never have: subclassing the BPR model meant the mechanism only ever existed for
BPR.

The three orders
----------------

`batch_order` takes three values, and the middle one is a control rather than a
proposal:

    shuffle    random rows, random order    cornac's own behaviour (baseline)
    coherent   one time band, random order  isolates *batch coherence*
    temporal   one time band, chronological coherence + the chronological march

`coherent` exists because `temporal` changes two things at once -- batches become
time-coherent *and* the model marches forward through time. Without a cell that
has the first without the second, a measured effect cannot be attributed to
either. It costs one branch, and it is available but not part of the default
ablation ladder (see `runner.py`).

The ablation is a cumulative ladder rather than a factorial cross::

    uniform                     baseline
    causal                      + Mechanism 1   (tuned on the causal arm)
    causal + temporal           + Mechanism 2   (reusing causal's winners)

`uniform + temporal` is deliberately not run. The consequence is worth stating
once: the gain attributed to Mechanism 2 is measured *given* Mechanism 1, so it
answers "does temporal batching improve our method" rather than "does temporal
batching help in general". That is the claim being made, but it does mean the
missing cell cannot be recovered later without running it.

What this does **not** change
-----------------------------

The negative sampler. `_draw_negatives` is untouched, so rho and `collision_rate`
are unaffected by `batch_order` and Mechanism 1 remains a separate axis -- which
is what makes the M1 x M2 cross a clean 2x2 rather than two entangled knobs.
`test_batching.py` asserts this rather than trusting it.
"""

import numpy as np

from ..lib.timeaware_data import NEG_SAMPLING_MODES, TimeAwareDataset

#: Ordering strategies, in ablation order (baseline first).
BATCH_ORDERS = ("shuffle", "coherent", "temporal")


def _check_order(batch_order):
    order = str(batch_order).lower()
    if order not in BATCH_ORDERS:
        raise ValueError(
            "batch_order must be one of {}, got {!r}".format(
                BATCH_ORDERS, batch_order))
    return order


class TemporalBatchDataset(TimeAwareDataset):
    """`TimeAwareDataset` whose mini-batches are bands of the training timeline.

    Construct via `TemporalBatchDataset.from_dataset(existing_dataset, ...)`, or
    upgrade a built split with `use_temporal_batching` below.

    `batch_order` is a plain attribute so a runner can switch arms in place
    between cells, exactly as `ablation_harness.set_recipe` switches
    `neg_sampling` -- no rebuild, and the split's RNG reset keeps the arms a
    paired comparison.
    """

    @classmethod
    def from_dataset(cls, dataset, neg_sampling="causal", batch_order="temporal"):
        obj = super().from_dataset(dataset, neg_sampling=neg_sampling)
        obj.batch_order = _check_order(batch_order)
        return obj

    @property
    def time_order(self):
        """Row indices sorted by timestamp, computed once.

        `kind="stable"` so rows sharing a timestamp keep their original relative
        order: ties are common in these datasets (timestamps are coarse), and an
        unstable sort would make batch contents depend on numpy's internals
        rather than on the data.
        """
        cached = getattr(self, "_time_order", None)
        if cached is None:
            cached = np.argsort(self._ts_array, kind="stable")
            self._time_order = cached
        return cached

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        """Batch indices, ordered by `self.batch_order`.

        Mirrors `cornac.data.Dataset.idx_iter`'s contract -- yields arrays of row
        indices -- but replaces `np.arange(idx_range)` with the time-sorted order
        and chunks that instead.

        The caller's `shuffle` argument is deliberately **ignored** outside
        `shuffle` mode. cornac's LightGCN passes `shuffle=True` unconditionally,
        so honouring it would undo the mechanism at the one call site that
        matters most.

        Falls back to the base implementation whenever `idx_range` is not the
        full row count. Nothing in cornac does that today, but the base method's
        contract permits it, and a partial range has no meaningful time order.
        """
        order = _check_order(getattr(self, "batch_order", "shuffle"))
        rows = self.time_order
        if order == "shuffle" or idx_range != len(rows):
            yield from super().idx_iter(idx_range, batch_size, shuffle)
            return

        n_batches = int(np.ceil(len(rows) / batch_size))
        batches = np.arange(n_batches)
        if order == "coherent":
            # Shuffle which band is visited when, NOT the rows inside a band --
            # that is the whole distinction from `shuffle`.
            self.rng.shuffle(batches)

        for b in batches:
            yield rows[b * batch_size:(b + 1) * batch_size]


def use_temporal_batching(eval_method, batch_order="temporal",
                          neg_sampling=None):
    """Upgrade a built split's training set to `TemporalBatchDataset`, in place.

    Mirrors what `CausalTimestampSplit.build` does for Mechanism 1: only the
    *training* split is upgraded, since batching is a training-time concern and
    the evaluation path never iterates batches.

    Returns the new training set so a caller can read the probes off it, matching
    `ablation_harness.set_recipe`'s contract.

    Re-runs `_init_causal_index` (a sort plus the observed-key index), which is
    tenths of a second even on the largest dataset and happens once per cell --
    cheap enough not to warrant mutating `__class__` in place.
    """
    train_set = eval_method.train_set
    if neg_sampling is None:
        neg_sampling = getattr(train_set, "neg_sampling", "causal")
    eval_method.train_set = TemporalBatchDataset.from_dataset(
        train_set, neg_sampling=neg_sampling, batch_order=batch_order)
    return eval_method.train_set


#: Models that can receive Mechanism 2, and how each gets it.
#:
#: `neumf` and `lightgcn` are loader-driven: they batch through
#: `uij_iter` / `uir_iter`, which read `idx_iter`, so they need no model-side
#: code at all.
#:
#: `bpr` batches itself -- `lib/bpr_cpu.py:fit` draws its own
#: `rng.permutation(n_samples)` and never consults the split -- so it needs
#: `mecha2/bpr.py:TemporalBPR`, a subclass overriding the one `_epoch_batches`
#: seam. Same exception, same reason, as Mechanism 1: our BPR does its own work
#: where cornac's models delegate. A runner that passed a plain `BPRMiniBatch`
#: here would produce a cell labelled "temporal" that trained on shuffled
#: batches, which is the silent no-op this list exists to prevent.
M2_MODELS = ("bpr", "neumf", "lightgcn")


def set_arm(train_set, neg_sampling, batch_order):
    """Point a training split at one cell of the ablation, in place.

    The sibling of `ablation_harness.set_recipe`, extended with `batch_order`.
    Switching in place rather than rebuilding matters for the same reasons: a
    rebuild costs minutes on the larger datasets, and `reset()` restoring the
    split's RNG to its seed is what makes the arms a paired comparison -- same
    users, same positives, same draws -- so results cannot depend on the order
    the cells ran in.
    """
    neg = str(neg_sampling).lower()
    if neg not in NEG_SAMPLING_MODES:
        raise ValueError(
            "neg_sampling must be one of {}, got {!r}".format(
                NEG_SAMPLING_MODES, neg_sampling))
    train_set.neg_sampling = neg
    train_set.batch_order = _check_order(batch_order)
    train_set.reset()
    train_set.reset_counterfactual_counters()
    return train_set
