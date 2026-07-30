"""Direct faithfulness metrics.

These probe whether a recommender's training or inference behavior
uses information that wouldn't be available in production. Higher
values indicate worse faithfulness.

- counterfactual_rate(model): training-time counter that lives on
  BPRMiniBatch. Read it after fit() via `model.counterfactual_rate`.
  Defined here only for completeness — the work happens inside the
  sampler.

- future_items_pct(): inference-time. For each test instance, what
  fraction of top-K recommendations weren't released yet at the test
  timestamp. (Already defined in phase2_causal_negatives.py — we re-
  import it here for one-stop shopping.)

- recommendation_recency_distribution(): inference-time. For each
  test instance, look at the recency of every top-K recommendation
  (test_ts - first_seen). Returns the distribution + summary stats.
  Lets us catch the "model over-recommends recently-released items"
  failure mode that future_items_pct misses under TimestampSplit.
"""

from typing import Dict, List

import numpy as np

from .causal_sampling import future_items_pct  # re-export


def recommendation_recency_distribution(
    model,
    eval_method,
    item_first: Dict[str, int],
    k: int = 20,
    return_raw: bool = False,
):
    """For each test instance at time t, compute `t - first_seen(rec)` for
    each of the top-K recommendations. Return aggregated distribution.

    A faithful model's recency distribution should be broad — recommending
    a mix of recently-released and long-available items, proportional to
    what was actually available before time t. A leaky model concentrates
    on recently-released items (because those items received the most
    parameter updates during training).
    """
    test_set = eval_method.test_set
    uid2raw = {v: k_ for k_, v in test_set.uid_map.items()}
    u_indices, _, _ = test_set.uir_tuple
    timestamps = getattr(test_set, "timestamps", None)
    if timestamps is None:
        raise RuntimeError("test_set lacks timestamps; eval method must be UIRT-aware.")

    recencies: List[int] = []
    skipped = 0
    for u_idx, ts in zip(u_indices, timestamps):
        raw_user = uid2raw.get(int(u_idx))
        if raw_user is None:
            skipped += 1
            continue
        try:
            recs = model.recommend(user_id=raw_user, k=k, remove_seen=False)
        except Exception:
            skipped += 1
            continue
        for item_raw in recs:
            fs = item_first.get(item_raw)
            if fs is not None:
                recencies.append(int(ts) - int(fs))

    if not recencies:
        out = {"evaluated": 0, "skipped": skipped}
        if return_raw:
            out["recency_days_raw"] = np.empty(0, dtype=float)
        return out

    arr = np.asarray(recencies, dtype=np.int64)
    # Convert ms -> days for readability (timestamps are ms-epoch).
    days = arr / (1000 * 60 * 60 * 24)
    out = {
        "evaluated_recs": int(arr.size),
        "skipped_users": skipped,
        "recency_days_mean":   float(np.mean(days)),
        "recency_days_median": float(np.median(days)),
        "recency_days_p10":    float(np.percentile(days, 10)),
        "recency_days_p25":    float(np.percentile(days, 25)),
        "recency_days_p75":    float(np.percentile(days, 75)),
        "recency_days_p90":    float(np.percentile(days, 90)),
    }
    if return_raw:
        out["recency_days_raw"] = days
    return out
