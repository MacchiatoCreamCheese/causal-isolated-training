"""Is a low NDCG the dataset or the model? A fast check with cornac's own models.

NDCG@20 on the Amazon sets is far lower than on Philadelphia. This runs two
cornac built-ins -- no tuning, no GPU, same split and evaluator as every other
result -- on each active dataset, next to facts about the split itself:

    MostPop   recommends the most-interacted training items. No personalisation,
              so its score is a measure of how hard the *data* is.
    MF        cornac's matrix factorisation (SGD on observed ratings, with
              biases), at cornac's defaults: the fastest personalised model.

Split facts, computed without training:
    items / random HR   20 / items: what guessing scores; a big catalogue alone
                        pushes every metric down
    test users, pos/user  users evaluated and test positives per user
    top-20 share        share of test positives landing on the 20 most popular
                        training items -- roughly MostPop's ceiling
    gap days            last training interaction to median test interaction:
                        how far into the future the model must predict
    rho                 closed-form future-negative rate (paper, Prop. 3.1)

How to read it:
  - MostPop low on Amazon, high on Philadelphia -> the data/split is harder
    (big sparse catalogue, drift, few positives per user), not the models.
  - MostPop similar everywhere but MF low only on Amazon -> look at the models.
  - MF / MostPop is how much personalisation adds, comparable across datasets.
    cornac's MF regresses ratings rather than ranking items, so it can sit below
    MostPop on top-20 ranking; that is informative, not a bug.

Usage:
    python -m research.analysis.dataset_difficulty
    python -m research.analysis.dataset_difficulty --datasets philadelphia --seed 42
"""

import argparse
import json
import os
import time

import numpy as np

import cornac
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import MF, MostPop

from ..lib.causal_sampling import TOP_K
from ..lib.data import DATASETS, build_eval_method
from ..paths import RESULTS_DIR

ACTIVE = ("musical", "baby", "cellphone", "philadelphia", "movielens")
OUT = RESULTS_DIR / "diagnostics" / "dataset_difficulty.json"
DAY_MS = 86_400_000


def split_facts(em):
    """Facts about the split itself -- no model involved."""
    tr, te = em.train_set, em.test_set
    tr_u, tr_i, _ = (np.asarray(a) for a in tr.uir_tuple)
    te_u, te_i, te_r = (np.asarray(a) for a in te.uir_tuple)
    tr_ts = np.asarray(tr.timestamps, dtype=np.int64)
    te_ts = np.asarray(getattr(te, "timestamps", []), dtype=np.int64)

    # Positives exactly as cornac's ranking evaluation counts them.
    pos = te_r >= 1.0
    test_users = np.unique(te_u[pos])
    top20 = np.argsort(-np.bincount(tr_i, minlength=tr.num_items))[:TOP_K]

    tau = np.full(tr.num_items, np.iinfo(np.int64).max)
    np.minimum.at(tau, tr_i, tr_ts)
    not_yet = tr.num_items - np.searchsorted(np.sort(tau), tr_ts, side="right")

    return {
        "users": int(tr.num_users),
        "items": int(tr.num_items),
        "train_rows": int(len(tr_u)),
        "test_users": int(len(test_users)),
        "test_positives": int(pos.sum()),
        "pos_per_user": float(pos.sum() / max(len(test_users), 1)),
        "random_hr": float(min(1.0, TOP_K / tr.num_items)),
        "top20_share": float(np.isin(te_i[pos], top20).mean()) if pos.any() else 0.0,
        "gap_days": float((np.median(te_ts) - tr_ts.max()) / DAY_MS) if len(te_ts) else float("nan"),
        "rho": float((not_yet / tr.num_items).mean()),
    }


def run_models(em, seed):
    """MostPop and cornac MF at defaults, in one experiment."""
    models = [MostPop(), MF(seed=seed)]
    exp = cornac.Experiment(eval_method=em, models=models,
                            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
                            user_based=True, save_dir=None, verbose=False)
    exp.run()
    return {res.model_name: {k: float(v) for k, v in res.metric_avg_results.items()}
            for res in exp.result}


def load():
    if OUT.exists():
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save(records):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    os.replace(tmp, OUT)


def print_table(records, keys):
    head = (f"{'dataset':<13}{'items':>8}{'test u':>8}{'pos/u':>7}{'rand HR':>9}"
            f"{'top20':>7}{'gap d':>7}{'rho':>6} | {'Pop NDCG':>9}{'Pop HR':>8} | "
            f"{'MF NDCG':>8}{'MF HR':>8} | {'MF/Pop':>7}")
    print("\n" + head + "\n" + "-" * len(head))
    for key in keys:
        r = records.get(key)
        if r is None:
            continue
        s, pop, mf = r["split"], r["models"]["MostPop"], r["models"]["MF"]
        ratio = mf["NDCG@20"] / pop["NDCG@20"] if pop["NDCG@20"] else float("nan")
        print(f"{r['dataset']:<13}{s['items']:>8,}{s['test_users']:>8,}{s['pos_per_user']:>7.2f}"
              f"{100 * s['random_hr']:>8.3f}%{100 * s['top20_share']:>6.1f}%{s['gap_days']:>7.0f}"
              f"{100 * s['rho']:>5.1f}% | {pop['NDCG@20']:>9.4f}{pop['HitRatio@20']:>8.4f} | "
              f"{mf['NDCG@20']:>8.4f}{mf['HitRatio@20']:>8.4f} | {ratio:>7.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=",".join(ACTIVE))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {sorted(DATASETS)})")

    records = load()
    keys = [f"{d}|seed{args.seed}" for d in datasets]
    for ds, key in zip(datasets, keys):
        if key in records:
            print(f"[skip] {ds} (seed {args.seed}) already in {OUT.name}", flush=True)
            continue
        t0 = time.time()
        print(f"[{ds}] building split ...", flush=True)
        em = build_eval_method(ds, neg_sampling="uniform", seed=args.seed)
        facts = split_facts(em)
        print(f"[{ds}] training MostPop + MF ...", flush=True)
        models = run_models(em, args.seed)
        records[key] = {"dataset": ds, "seed": args.seed, "split": facts,
                        "models": models, "wall_seconds": time.time() - t0}
        save(records)
        print(f"[{ds}] done in {time.time() - t0:.0f}s", flush=True)

    print_table(records, keys)


if __name__ == "__main__":
    main()
