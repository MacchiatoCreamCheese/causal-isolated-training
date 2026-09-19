from ..lib.bpr_cpu import BPRMiniBatch


class TemporalBPR(BPRMiniBatch):
    def _epoch_batches(self, train_set, n_samples):
        order = getattr(train_set, "batch_order", None)
        if order is None or order == "shuffle":
            yield from super()._epoch_batches(train_set, n_samples)
            return
        yield from train_set.idx_iter(n_samples, self.batch_size, shuffle=True)
