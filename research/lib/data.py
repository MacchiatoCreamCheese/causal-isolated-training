from pathlib import Path

from cornac.data import Reader

from .timeaware_data import CausalTimestampSplit
from ..paths import DATA_DIR


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
    "philadelphia": {
        "path": "philadelphia_dataset/Philadelphia.csv",
        "val_ts": 1537357870200,
        "test_ts": 1572859360400,
    },
    "movielens": {
        "path": "movielens_dataset/MovieLens-10M.csv",
        "val_ts": 1145393009000,
        "test_ts": 1188213670000,
    },
    "healthcare": {
        "path": "healthcare_dataset/Health_and_Household.csv",
        "val_ts": 1628643414042,
        "test_ts": 1658002729837,
    },
}


def dataset_path(key: str) -> str:
    return str(DATA_DIR / DATASETS[key]["path"])


def load_uirt(path: str):
    return Reader().read(path, fmt="UIRT", sep=",", skip_lines=1)


def build_eval_method(key: str, neg_sampling: str = "causal",
                      seed: int = None,
                      verbose: bool = False) -> CausalTimestampSplit:
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
