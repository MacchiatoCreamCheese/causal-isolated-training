import numpy as np

from .timeaware_data import (
    build_observed_index,
    causal_draw,
    compute_item_first_seen,
    reject_collisions,
    uniform_draw,
)


def ensure_causal_arrays(train_set):
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
    def __init__(self, train_set, sampler_kind, rng):
        self.kind = sampler_kind
        self.rng = rng
        (self.item_first_seen,
         self.sorted_first_seen,
         self.sorted_item_order) = ensure_causal_arrays(train_set)
        self.num_items = len(self.item_first_seen)
        self.observed_keys = getattr(train_set, "observed_keys", None)
        if self.observed_keys is None:
            self.observed_keys, _ = build_observed_index(
                train_set.uir_tuple[0], train_set.uir_tuple[1],
                train_set.uir_tuple[2], self.num_items,
            )
        self._counterfactual_negs = 0
        self._total_negs = 0
        self._residual_collisions = 0

    def sample(self, pos_ts, users, num_neg=1):
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
            redraw=lambda sel: draw(len(sel), ts_rep[sel]),
        )
        self._residual_collisions += residual
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
