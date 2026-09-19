from typing import Dict

import numpy as np

import cornac
from cornac.eval_methods import RatioSplit, StratifiedSplit, TimestampSplit
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.causal_sampling import item_first_seen
from ..lib.data import dataset_path, load_uirt
from ..paths import logs_dir


DATASET_CSV = dataset_path("baby")
VAL_TIMESTAMP = 1628643414042
TEST_TIMESTAMP = 1658002729837
TOP_K = 20
BPR_KWARGS = dict(k=64, max_iter=100, learning_rate=0.001, lambda_reg=0.001, seed=42)


def per_user_test_timestamps(rows, test_set) -> Dict[str, int]:
    test_pairs = set()
    iid2raw = {v: k for k, v in test_set.iid_map.items()}
    uid2raw = {v: k for k, v in test_set.uid_map.items()}
    u_idx, i_idx, _ = test_set.uir_tuple
    for uu, ii in zip(u_idx, i_idx):
        test_pairs.add((uid2raw[int(uu)], iid2raw[int(ii)]))

    per_user: Dict[str, int] = {}
    for u, i, _, t in rows:
        if (u, i) in test_pairs:
            prev = per_user.get(u)
            if prev is None or t < prev:
                per_user[u] = t
    return per_user


def future_items_pct(model, eval_method, item_first, raw_rows, k):
    test_set = eval_method.test_set
    uid2raw = {v: k_ for k_, v in test_set.uid_map.items()}

    if getattr(test_set, "timestamps", None) is not None:
        u_indices, _, _ = test_set.uir_tuple
        ts_by_row = list(test_set.timestamps)
        pairs = list(zip(u_indices, ts_by_row))
    else:
        per_user_ts = per_user_test_timestamps(raw_rows, test_set)
        u_indices, _, _ = test_set.uir_tuple
        pairs = []
        for u_idx in u_indices:
            raw_u = uid2raw[int(u_idx)]
            ts = per_user_ts.get(raw_u)
            if ts is not None:
                pairs.append((u_idx, ts))

    per_instance = []
    n_future = 0
    n_total = 0
    skipped = 0
    for u_idx, ts in pairs:
        raw_user = uid2raw.get(int(u_idx))
        if raw_user is None:
            skipped += 1
            continue
        try:
            recs = model.recommend(user_id=raw_user, k=k, remove_seen=False)
        except Exception:
            skipped += 1
            continue
        future_here = sum(1 for it in recs if item_first.get(it, 0) > ts)
        per_instance.append(future_here / max(len(recs), 1))
        n_future += future_here
        n_total += len(recs)

    return {
        "mean_pct_per_instance": float(np.mean(per_instance)) if per_instance else 0.0,
        "global_pct": n_future / n_total if n_total else 0.0,
        "evaluated": len(per_instance),
        "skipped": skipped,
    }


def evaluate(name, eval_method, raw_rows, item_first):
    print(f"\n========== {name} ==========")
    bpr = cornac.models.BPR(**BPR_KWARGS, verbose=False)
    cornac.Experiment(
        eval_method=eval_method,
        models=[bpr],
        metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
        user_based=True,
        save_dir=logs_dir(),
    ).run()
    stats = future_items_pct(bpr, eval_method, item_first, raw_rows, k=TOP_K)
    print(f"[{name}] future_items: {stats}")
    return stats


def main():
    rows = load_uirt(DATASET_CSV)
    print(f"Loaded {len(rows):,} interactions from {DATASET_CSV}")
    item_first = item_first_seen(rows)

    ratio = RatioSplit(
        data=rows,
        fmt="UIRT",
        test_size=0.1,
        val_size=0.1,
        rating_threshold=1.0,
        exclude_unknowns=True,
        seed=42,
        verbose=False,
    )
    evaluate("RatioSplit (no timeline)", ratio, rows, item_first)

    strat = StratifiedSplit(
        data=rows,
        fmt="UIRT",
        group_by="user",
        chrono=True,
        test_size=0.1,
        val_size=0.1,
        rating_threshold=1.0,
        exclude_unknowns=True,
        seed=42,
        verbose=False,
    )
    evaluate("StratifiedSplit (local timeline)", strat, rows, item_first)

    ts = TimestampSplit(
        data=rows,
        val_timestamp=VAL_TIMESTAMP,
        test_timestamp=TEST_TIMESTAMP,
        fmt="UIRT",
        exclude_unknowns=True,
        verbose=False,
    )
    evaluate("TimestampSplit (global timeline)", ts, rows, item_first)


if __name__ == "__main__":
    main()
