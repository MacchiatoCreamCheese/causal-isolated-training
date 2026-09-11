"""Mechanism 2 for our NumPy BPR.

cornac's LightGCN and NCF family receive temporal batching for free: they draw
their batches from `uij_iter` / `uir_iter`, which draw indices from
`Dataset.idx_iter`, which `TemporalBatchDataset` overrides. Nothing model-side
changes.

`lib/bpr_cpu.py:BPRMiniBatch` is the exception, for the same reason it was the
exception for Mechanism 1: it does its own work rather than asking the loader.
For Mechanism 1 that was the negative sampler (cornac's BPR samples inside
compiled Cython over a timestamp-free CSR, so we wrote a NumPy BPR that reads the
same first-seen index). For Mechanism 2 it is the batching -- `fit` shuffles its
own `rng.permutation(n_samples)` each epoch and never consults the split.

So BPR gets Mechanism 2 through the one seam `BPRMiniBatch` exposes,
`_epoch_batches`. Overriding that single method routes batching through the
split's `idx_iter`, which means BPR and the cornac models end up sharing **one**
definition of what a temporal batch is -- exactly as `NumpyCausalSampler` and
`TimeAwareDataset` share one definition of what a causal negative is. Subclassing
`fit` instead would have copied the Rendle 2009 gradient update into a second
place, and the two copies would drift.
"""

from ..lib.bpr_cpu import BPRMiniBatch


class TemporalBPR(BPRMiniBatch):
    """`BPRMiniBatch` whose epoch batches come from the training split.

    With a `TemporalBatchDataset` training split this yields time bands; with
    `batch_order="shuffle"`, or with any split that has no `batch_order` at all,
    it falls back to the parent's own permutation and is bit-identical to a plain
    `BPRMiniBatch`.

    That fallback is deliberate rather than defensive. Delegating to
    `Dataset.idx_iter` unconditionally would switch the shuffle from the *model's*
    `numpy.random.Generator` to the *split's* legacy `RandomState`, silently
    changing every baseline BPR number for no reason. The baseline arm has to stay
    the same model it was.
    """

    def _epoch_batches(self, train_set, n_samples):
        order = getattr(train_set, "batch_order", None)
        if order is None or order == "shuffle":
            yield from super()._epoch_batches(train_set, n_samples)
            return
        # `shuffle=True` is what cornac's own callers pass; TemporalBatchDataset
        # ignores it outside "shuffle" mode, which is the point.
        yield from train_set.idx_iter(n_samples, self.batch_size, shuffle=True)
