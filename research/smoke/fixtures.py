import numpy as np

from ..lib.timeaware_data import CausalTimestampSplit

VAL_TS = 1_615_000_000_000
TEST_TS = 1_625_000_000_000


def negatives_only(batch_users, batch_items, num_zeros):
    split = len(batch_users) // (num_zeros + 1)
    return batch_users[split:], batch_items[split:]


def _generate(rng, rating_fn, n_users, per_block):
    rows = []
    for block in range(4):
        base_ts = 1_600_000_000_000 + block * 10_000_000_000
        n_available = (block + 1) * 40
        for _ in range(per_block):
            u = int(rng.integers(0, n_users))
            i = int(rng.integers(0, n_available))
            rows.append((f"u{u}", f"i{i}", rating_fn(rng),
                         base_ts + int(rng.integers(0, 5_000_000_000))))
    return rows


def _split(rows, neg_sampling):
    return CausalTimestampSplit(
        data=rows,
        val_timestamp=VAL_TS,
        test_timestamp=TEST_TS,
        fmt="UIRT", exclude_unknowns=True,
        neg_sampling=neg_sampling, verbose=False,
    )


def build_split(neg_sampling, seed=0):
    rng = np.random.default_rng(seed)
    return _split(_generate(rng, lambda _rng: 1.0, 300, 4000), neg_sampling)


def build_rated_split(neg_sampling, seed=0):
    rng = np.random.default_rng(seed)
    rows = _generate(rng, lambda r: float(r.integers(1, 6)), 150, 5000)
    return _split(rows, neg_sampling)


def uij_batches(train_set, batch_size):
    return ((bu, bj, bi)
            for bu, bi, bj in train_set.uij_iter(batch_size=batch_size,
                                                 shuffle=True))
