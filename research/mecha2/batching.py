import numpy as np

from ..lib.timeaware_data import NEG_SAMPLING_MODES, TimeAwareDataset

BATCH_ORDERS = ("shuffle", "coherent", "temporal")


def _check_order(batch_order):
    order = str(batch_order).lower()
    if order not in BATCH_ORDERS:
        raise ValueError(
            "batch_order must be one of {}, got {!r}".format(
                BATCH_ORDERS, batch_order))
    return order


class TemporalBatchDataset(TimeAwareDataset):
    @classmethod
    def from_dataset(cls, dataset, neg_sampling="causal", batch_order="temporal"):
        obj = super().from_dataset(dataset, neg_sampling=neg_sampling)
        obj.batch_order = _check_order(batch_order)
        return obj

    @property
    def time_order(self):
        cached = getattr(self, "_time_order", None)
        if cached is None:
            cached = np.argsort(self._ts_array, kind="stable")
            self._time_order = cached
        return cached

    def idx_iter(self, idx_range, batch_size=1, shuffle=False):
        order = _check_order(getattr(self, "batch_order", "shuffle"))
        rows = self.time_order
        if order == "shuffle" or idx_range != len(rows):
            yield from super().idx_iter(idx_range, batch_size, shuffle)
            return

        n_batches = int(np.ceil(len(rows) / batch_size))
        batches = np.arange(n_batches)
        if order == "coherent":
            self.rng.shuffle(batches)

        for b in batches:
            yield rows[b * batch_size:(b + 1) * batch_size]


def use_temporal_batching(eval_method, batch_order="temporal",
                          neg_sampling=None):
    train_set = eval_method.train_set
    if neg_sampling is None:
        neg_sampling = getattr(train_set, "neg_sampling", "causal")
    eval_method.train_set = TemporalBatchDataset.from_dataset(
        train_set, neg_sampling=neg_sampling, batch_order=batch_order)
    return eval_method.train_set


M2_MODELS = ("bpr", "neumf", "lightgcn")


def set_arm(train_set, neg_sampling, batch_order):
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
