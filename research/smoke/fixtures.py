"""Synthetic UIRT splits shared by the smoke tests.

Both fixtures generate a log whose catalog *grows over time*, so a time-blind
sampler demonstrably draws items that did not yet exist. They need no dataset
CSVs and build in a second or two.

`build_split` is the original all-1.0-ratings fixture: implicit feedback, where
cornac's two rejection rules (`>= pos_rating` and `> 0`) coincide.
`build_rated_split` varies the ratings, which is what makes the two rules
diverge — see the rule table in `lib/timeaware_data.py`. Any check on the
per-model filter has to use the latter; on the former it would pass either way.
"""

import numpy as np

from ..lib.timeaware_data import CausalTimestampSplit

VAL_TS = 1_615_000_000_000
TEST_TS = 1_625_000_000_000


def _generate(rng, rating_fn, n_users, per_block):
    """Four blocks of interactions; block `b` may use the first `(b+1)*40`
    items, so the catalog grows from 40 to 160 as time advances."""
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
    """Implicit-feedback fixture: every rating is 1.0."""
    rng = np.random.default_rng(seed)
    return _split(_generate(rng, lambda r: 1.0, 300, 4000), neg_sampling)


def build_rated_split(neg_sampling, seed=0):
    """Explicit-rating fixture: ratings drawn from {1..5}.

    Deliberately dense — 150 users over at most 160 items — so a user owns a
    large slice of the catalog and a uniform draw lands on one of their own
    items often enough for the two rejection rules to visibly differ.
    """
    rng = np.random.default_rng(seed)
    rows = _generate(rng, lambda r: float(r.integers(1, 6)), 150, 5000)
    return _split(rows, neg_sampling)
