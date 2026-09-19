from typing import Dict, List

import numpy as np

from .causal_sampling import future_items_pct


def recommendation_recency_distribution(
    model,
    eval_method,
    item_first: Dict[str, int],
    k: int = 20,
    return_raw: bool = False,
):
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
