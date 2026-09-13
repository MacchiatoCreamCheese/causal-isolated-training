"""Generic dataset loading + eval-method construction.

One place that knows (a) where each dataset lives and its val/test timestamp
cutoffs, and (b) how to turn a dataset key into a ready-to-use eval method.
Every runner calls `build_eval_method(key)` instead of re-writing the same
`TimestampSplit(data=…, val_timestamp=…, test_timestamp=…)` block.

The returned eval method is a `CausalTimestampSplit`, so its training split
supports both `neg_sampling="uniform"` and `neg_sampling="causal"` — the
per-cell choice is made by the model at sampling time, not baked into the
split.
"""

from pathlib import Path

from cornac.data import Reader

from .timeaware_data import CausalTimestampSplit
from ..paths import DATA_DIR


# Each dataset records its CSV path (relative to DATA_DIR) plus the val/test
# cutoff timestamps used by TimestampSplit. All three are Amazon Reviews 2023
# 5-core subsets sharing the same calendar cutoffs (Aug 2021 / Jul 2022).
#
# Those cutoffs are not approximations: splitting the combined 5-core CSV at
# them reproduces the official `benchmark/5core/timestamp` partition exactly
# (verified row-for-row on all three datasets).
# The primary three are musical, baby, cellphone; healthcare is the largest
# (7.18M) and is run last, so it is not in any runner's default list.
DATASETS = {
    "musical": {
        "path": "musical_dataset/Musical_Instruments.csv",
        "val_ts": 1628643414042,
        "test_ts": 1658002729837,
    },
    "baby": {
        "path": "baby_dataset/Baby_Products.csv",
        "val_ts": 1628643414042,
        "test_ts": 1658002729837,
    },
    "cellphone": {
        "path": "cellphone_dataset/Cell_Phones_and_Accessories.csv",
        "val_ts": 1628643414042,
        "test_ts": 1658002729837,
    },
    # Yelp, Philadelphia businesses (from the OSCAR project). Same protocol as
    # the Amazon sets -- already 5-core, split once for the whole corpus at the
    # 80th and 90th percentiles of interaction time -- but on its own dates: its
    # log ends in Jan 2022, before the Amazon test cutoff. Timestamps were
    # seconds in the source and are stored in ms here, like the Amazon files.
    # The test period starts Nov 2019 and so spans the COVID years. Opt-in, like
    # healthcare: pass --datasets philadelphia.
    "philadelphia": {
        "path": "philadelphia_dataset/Philadelphia.csv",
        "val_ts": 1537357870200,    # 2018-09-19
        "test_ts": 1572859360400,   # 2019-11-04
    },
    # MovieLens-32M (GroupLens), same protocol as the Amazon sets. The last 10
    # years of the log (2013-10-12 to 2023-10-13) -- the span holding ~95% of the
    # Amazon sets' interactions, on the same calendar -- then a random 20% of
    # those users with their full histories (numpy default_rng(42)), then
    # iterative 5-core. Ratings as-is (half stars; the 0.5 ratings fall below
    # cornac's rating_threshold of 1.0 at evaluation), timestamps converted from
    # seconds to ms, split once at the 80th and 90th percentiles of interaction
    # time. 2.61M ratings, 15,336 users, 18,807 movies. Opt-in: --datasets movielens.
    "movielens": {
        "path": "movielens_dataset/MovieLens-32M.csv",
        "val_ts": 1614537679800,    # 2021-02-28
        "test_ts": 1652827323000,   # 2022-05-17
    },
    "healthcare": {
        "path": "healthcare_dataset/Health_and_Household.csv",
        "val_ts": 1628643414042,
        "test_ts": 1658002729837,
    },
}


def dataset_path(key: str) -> str:
    """Absolute CSV path for a dataset key, resolved under DATA_DIR."""
    return str(DATA_DIR / DATASETS[key]["path"])


def load_uirt(path: str):
    """Read a UIRT CSV into a list of (user, item, rating, timestamp) tuples
    using cornac's own Reader (header row skipped)."""
    return Reader().read(path, fmt="UIRT", sep=",", skip_lines=1)


def build_eval_method(key: str, neg_sampling: str = "causal",
                      seed: int = None,
                      verbose: bool = False) -> CausalTimestampSplit:
    """Resolve a dataset key to a ready `CausalTimestampSplit`.

    Reads the dataset CSV straight from disk (under DATA_DIR). The training
    split is a `TimeAwareDataset` stamped with `neg_sampling`, which is how a
    runner selects the ablation arm — cornac's models take their negatives
    from the split and expose no sampling option of their own.

    **Pass `seed`.** It is what makes negative sampling reproducible: cornac's
    models draw their negatives from this split, not from themselves, so a
    model-side seed does not cover them. Left as None, `Dataset.rng` falls back
    to numpy's global singleton and every run samples differently.
    """
    if key not in DATASETS:
        raise KeyError(f"unknown dataset key: {key} (known: {list(DATASETS)})")
    info = DATASETS[key]
    path = dataset_path(key)
    if not Path(path).exists():
        raise FileNotFoundError(
            f"dataset CSV not found: {path}\n"
            f"Put the file there, or set RESEARCH_DATA_DIR to the folder that "
            f"contains {DATASETS[key]['path']}."
        )
    rows = load_uirt(path)
    if verbose:
        print(f"[data] loaded {len(rows):,} rows from {path}", flush=True)
    return CausalTimestampSplit(
        data=rows,
        val_timestamp=info["val_ts"],
        test_timestamp=info["test_ts"],
        fmt="UIRT",
        exclude_unknowns=True,
        neg_sampling=neg_sampling,
        seed=seed,
        verbose=verbose,
    )
