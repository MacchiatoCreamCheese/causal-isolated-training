import numpy as np

from cornac.data import Dataset
from cornac.eval_methods import TimestampSplit


_NEVER_SEEN = np.iinfo(np.int64).max

NEG_SAMPLING_MODES = ("causal", "uniform")


def compute_item_first_seen(item_indices, timestamps, num_items):
    first_seen = np.full(num_items, _NEVER_SEEN, dtype=np.int64)
    np.minimum.at(first_seen, np.asarray(item_indices, dtype=np.int64),
                  np.asarray(timestamps, dtype=np.int64))
    return first_seen


def uniform_draw(num_items, size, rng):
    draw = getattr(rng, "integers", None)
    if draw is None:
        draw = rng.randint
    return draw(0, num_items, size=size)


MAX_REJECT_ROUNDS = 4


def build_observed_index(users, items, ratings, num_items):
    keys = (np.asarray(users, dtype=np.int64) * np.int64(num_items)
            + np.asarray(items, dtype=np.int64))
    ratings = np.asarray(ratings, dtype=np.float64)
    order = np.argsort(keys)
    keys = keys[order]
    if keys.size > 1 and not np.all(np.diff(keys)):
        raise ValueError(
            "duplicate (user, item) pairs in the training split: the observed "
            "index assumes cornac's Dataset.build has already deduplicated them, "
            "and searchsorted would otherwise return an arbitrary one of the "
            "duplicates' ratings."
        )
    return keys, ratings[order]


def find_collisions(users, negs, observed_keys, num_items,
                    observed_ratings=None, pos_ratings=None):
    if observed_keys.size == 0:
        return np.zeros(len(negs), dtype=bool)
    keys = (np.asarray(users, dtype=np.int64) * np.int64(num_items)
            + np.asarray(negs, dtype=np.int64))

    order = np.argsort(keys)
    probes = keys[order]
    idx = np.searchsorted(observed_keys, probes)
    np.clip(idx, 0, observed_keys.size - 1, out=idx)
    matched = observed_keys[idx] == probes

    hit = np.empty(keys.size, dtype=bool)
    hit[order] = matched
    if pos_ratings is None:
        return hit
    gathered = np.empty(keys.size, dtype=observed_ratings.dtype)
    gathered[order] = observed_ratings[idx]
    return hit & (gathered >= np.asarray(pos_ratings))


def reject_collisions(users, negs, observed_keys, num_items, redraw,
                      observed_ratings=None, pos_ratings=None,
                      max_rounds=MAX_REJECT_ROUNDS):
    negs = np.array(negs, dtype=np.int64, copy=True)
    users = np.asarray(users)
    pos_ratings = None if pos_ratings is None else np.asarray(pos_ratings)

    hit = find_collisions(users, negs, observed_keys, num_items,
                          observed_ratings, pos_ratings)
    sel = np.flatnonzero(hit)
    for _ in range(max_rounds):
        if sel.size == 0:
            return negs, 0
        negs[sel] = redraw(sel)
        still = find_collisions(
            users[sel], negs[sel], observed_keys, num_items, observed_ratings,
            None if pos_ratings is None else pos_ratings[sel],
        )
        sel = sel[still]
    return negs, int(sel.size)


def causal_draw(sorted_first_seen, sorted_items, pos_timestamps, rng):
    pos_timestamps = np.asarray(pos_timestamps, dtype=np.int64)
    prefix_len = np.searchsorted(sorted_first_seen, pos_timestamps, side="right")
    prefix_len = np.maximum(prefix_len, 1)
    idx_in_prefix = (rng.random(len(pos_timestamps)) * prefix_len).astype(np.int64)
    return sorted_items[idx_in_prefix]


class TimeAwareDataset(Dataset):
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


    def reset_counterfactual_counters(self):
        self._counterfactual_negs = 0
        self._total_negs = 0
        self._residual_collisions = 0

    @property
    def total_negatives(self):
        return self._total_negs

    @property
    def residual_collisions(self):
        return self._residual_collisions

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

    def _draw_negatives(self, users, batch_ts, pos_ratings=None):
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
            redraw=lambda sel: draw(len(sel), batch_ts[sel]),
            observed_ratings=self.observed_ratings,
            pos_ratings=pos_ratings,
        )
        self._residual_collisions += residual

        self._counterfactual_negs += int(
            (self.item_first_seen[negs] > batch_ts).sum()
        )
        self._total_negs += len(batch_ts)
        return negs


    def uij_iter(self, batch_size=1, shuffle=False, neg_sampling=None):
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
    def __init__(self, *args, neg_sampling="causal", **kwargs):
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
