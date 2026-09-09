"""Phase 1 baseline: measure evaluation-time leakage with BPR.

Goal of this script is NOT to fix leakage. It is to establish the
measurement scaffold that later phases (temporal batching, leakage-aware
negatives, time-evolving params) must beat.

We train the SAME BPR model under three split regimes (the three rows
of LeakageStudy Table 2 that matter):

  1. RatioSplit (random)             -- ignores both local and global
     timeline. Most common in published work; the leakiest.
  2. StratifiedSplit (chrono per user) -- respects each USER's local
     timeline (leave-last-N% per user), but not the global one. Items
     released after user A's test point can still appear in user B's
     training rows, leaking through collaborative filtering.
  3. TimestampSplit                  -- respects the global timeline:
     training items strictly predate validation, validation predates
     test. The only setup that closes the future-items channel.

For each regime we report:
  - HR@20, NDCG@20, Recall@20            (accuracy)
  - future_items_pct                     (LeakageStudy Finding 1)

A non-zero future_items_pct under RatioSplit, and ~0 under TimestampSplit,
is the visible signature of evaluation-time leakage. The accuracy delta
between the two is the headline number for the data-side fix.

Phase 2+ will attack the residual TRAINING-time leakage: even inside a
clean TimestampSplit, batch BPR shares parameters across the training
timeline, so a row at t1 is implicitly informed by a row at t2 > t1.

This is an experiment, not a self-test: it trains three real cornac BPR
models on the baby CSV, takes minutes, and prints numbers to be read rather
than a pass/fail. It also asks a different question from the rest of the
package -- evaluation-time leakage as a function of the *split*, where the
ablation studies training-time leakage as a function of the *sampler*. It
lived in `smoke/` until 2026-09-09 for historical reasons.

Usage:  python -m research.experiments.baseline_bpr_leakage
"""

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
    """Earliest test-instance timestamp per user.

    Under RatioSplit, test_set has no timestamps (it's not UIRT-built),
    so we have to derive timestamps from the raw rows by re-identifying
    which (user, item) pairs ended up in the test set.
    """
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
    """Deliberately not `lib.causal_sampling.future_items_pct`.

    That one assumes a UIRT-built test set with one timestamp per row, which is
    all the ablation ever uses. This script also evaluates `RatioSplit`, whose
    test set carries no timestamps at all, so it falls back to re-deriving a
    per-user timestamp from the raw rows (`per_user_test_timestamps` above).
    The two agree on the UIRT path; only the fallback is extra.
    """
    test_set = eval_method.test_set
    uid2raw = {v: k_ for k_, v in test_set.uid_map.items()}

    if getattr(test_set, "timestamps", None) is not None:
        # UIRT path: one timestamp per test row
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
