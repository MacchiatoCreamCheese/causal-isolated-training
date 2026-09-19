import glob
import json
import os

import numpy as np

from ..lib.data import build_eval_method
from ..paths import RESULTS_DIR

DATASETS = ("musical", "baby", "cellphone")


def closed_form(dataset):
    train = build_eval_method(dataset, neg_sampling="uniform", seed=42).train_set
    _u, items, _r = train.uir_tuple
    items = np.asarray(items, dtype=np.int64)
    ts = np.asarray(train.timestamps, dtype=np.int64)

    tau = np.full(train.num_items, np.iinfo(np.int64).max)
    np.minimum.at(tau, items, ts)

    not_yet = train.num_items - np.searchsorted(np.sort(tau), ts, side="right")
    return 100.0 * float((not_yet / train.num_items).mean())


def measured(dataset):
    values = []
    pattern = os.path.join(RESULTS_DIR, "ablation", f"*_{dataset}_seed*.json")
    for path in glob.glob(pattern):
        model = os.path.basename(path).rsplit("_", 2)[0]
        if "-" in model:
            continue
        with open(path, encoding="utf-8-sig") as f:
            rate = json.load(f)["recipes"].get("uniform", {}).get("counterfactual_rate")
        if rate is not None:
            values.append(100.0 * rate)
    return (float(np.mean(values)), len(values)) if values else (float("nan"), 0)


def main():
    print(f"{'dataset':<10} {'closed form':>12} {'measured':>10} {'cells':>6}")
    for ds in DATASETS:
        m, n = measured(ds)
        print(f"{ds:<10} {closed_form(ds):>11.2f}% {m:>9.2f}% {n:>6}")


if __name__ == "__main__":
    main()
