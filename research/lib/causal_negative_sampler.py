"""Per-fit negative-sampling executors shared by all our models.

`TimeAwareDataset` (see `timeaware_data.py`) owns the *definition* of causal
sampling — per-item first-seen and the sorted causal index. These two classes
are the *execution*: a model builds one sampler in `fit()` and calls
`.sample(pos_ts, num_neg)`; the sampler also tracks the counterfactual rate
(fraction of drawn negatives that did not yet exist — a Mechanism-1
faithfulness probe, ~0 for the causal sampler by construction).

Two backends, same behavior:
  - `NumpyCausalSampler`  — for the pure-NumPy CPU BPR.
  - `TorchCausalSampler`  — for the GPU models (BPR-GPU, NeuMF, LightGCN),
    keeping the draw on-device for speed.

Both read the arrays computed once on the training `Dataset`. If handed a
plain `Dataset` (not a `TimeAwareDataset`), they compute the index on the fly,
so a model still works under a stock `TimestampSplit`.
"""

import numpy as np

from .timeaware_data import (
    compute_item_first_seen,
    causal_draw,
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
        self._counterfactual_negs = 0
        self._total_negs = 0

    def sample(self, pos_ts, num_neg=1):
        pos_ts = np.asarray(pos_ts, dtype=np.int64)
        m = len(pos_ts)
        ts_rep = pos_ts if num_neg == 1 else np.repeat(pos_ts, num_neg)
        if self.kind == "uniform":
            negs = uniform_draw(self.num_items, len(ts_rep), self.rng)
            self._counterfactual_negs += int((self.item_first_seen[negs] > ts_rep).sum())
        else:
            negs = causal_draw(self.sorted_first_seen, self.sorted_item_order,
                               ts_rep, self.rng)
        self._total_negs += len(ts_rep)
        return negs if num_neg == 1 else negs.reshape(m, num_neg)

    @property
    def counterfactual_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._counterfactual_negs / self._total_negs


class TorchCausalSampler:
    """On-device (torch) twin of `NumpyCausalSampler`.

    `generator` is optional; when `None` the draws use the global torch RNG,
    matching the pre-refactor models (which seed via `torch.manual_seed`).
    """

    def __init__(self, train_set, sampler_kind, device, generator=None):
        import torch

        ifs, sfs, sio = ensure_causal_arrays(train_set)
        self.kind = sampler_kind
        self.device = device
        self.generator = generator
        self.num_items = len(ifs)
        self.item_first_seen = torch.as_tensor(ifs, dtype=torch.int64, device=device)
        self.sorted_first_seen = torch.as_tensor(sfs, dtype=torch.int64, device=device)
        self.sorted_items = torch.as_tensor(sio, dtype=torch.int64, device=device)
        self._counterfactual_negs = 0
        self._total_negs = 0

    def sample(self, pos_ts, num_neg=1):
        import torch

        b = pos_ts.shape[0]
        ts_rep = pos_ts if num_neg == 1 else pos_ts.repeat_interleave(num_neg)
        n = ts_rep.shape[0]
        if self.kind == "uniform":
            negs = torch.randint(0, self.num_items, (n,), device=self.device,
                                 generator=self.generator)
            self._counterfactual_negs += int((self.item_first_seen[negs] > ts_rep).sum().item())
        else:
            prefix_len = torch.searchsorted(
                self.sorted_first_seen, ts_rep, right=True
            ).clamp(min=1)
            rand_u = torch.rand(n, device=self.device, generator=self.generator)
            idx_in_prefix = (rand_u * prefix_len.to(torch.float32)).to(torch.int64)
            negs = self.sorted_items[idx_in_prefix]
        self._total_negs += n
        return negs if num_neg == 1 else negs.view(b, num_neg)

    @property
    def counterfactual_rate(self):
        if self._total_negs == 0:
            return 0.0
        return self._counterfactual_negs / self._total_negs
