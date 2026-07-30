# Datasets

The loader resolves dataset CSVs under **`RESEARCH_DATA_DIR`** (defaults to the
project root — i.e. the folder that contains `research/`). Point it at wherever
you keep the data:

```bash
export RESEARCH_DATA_DIR=/path/to/data      # Linux / macOS / WSL
# setx RESEARCH_DATA_DIR C:\path\to\data    # Windows (new shells)
```

Expected layout under `RESEARCH_DATA_DIR` (paths from `research/lib/data.py:DATASETS`):

```
baby_dataset/Baby_Products.csv
cellphone_dataset/Cell_Phones_and_Accessories.csv
healthcare_dataset/Health_and_Household.csv
movielens_dataset/MovieLens_1M.csv
```

Each CSV is `user_id,item_id,rating,timestamp` (UIRT, header row), read straight
from disk by cornac's `Reader`. Only the single combined CSV per dataset is
needed — not the `.train/.valid/.test` splits — because the split is done
in-code by `CausalTimestampSplit` at the `val_ts`/`test_ts` cutoffs recorded in
`DATASETS`.

Provide all four CSVs yourself (copy them here, or set `RESEARCH_DATA_DIR` to
their existing location). If a file is missing, `build_eval_method` raises a
`FileNotFoundError` naming the exact path it expected — nothing is downloaded.
