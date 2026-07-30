"""Mechanism 1 faithfulness probes (evaluation-time).

Training-time causal negative sampling now lives at the data-loader layer
(`timeaware_data.TimeAwareDataset` + `causal_negative_sampler`). What remains
here are the *evaluation-time* probes that quantify residual leakage — how many
recommended items did not yet exist at the moment of the test interaction
(`future_items_pct`) — plus `item_first_seen`, shared by the recency analysis.
"""

from typing import Dict

import numpy as np

from .data import load_uirt, dataset_path  # re-export the canonical UIRT reader


# Retained for single-dataset probe scripts that still reference them directly.
DATASET_CSV = dataset_path("baby")
VAL_TIMESTAMP = 1628643414042
TEST_TIMESTAMP = 1658002729837
TOP_K = 20


def item_first_seen(rows) -> Dict[str, int]:
    first: Dict[str, int] = {}
    for _, item, _, ts in rows:
        prev = first.get(item)
        if prev is None or ts < prev:
            first[item] = ts
    return first


def future_items_pct(model, eval_method, item_first, k):
    test_set = eval_method.test_set
    uid2raw = {v: k_ for k_, v in test_set.uid_map.items()}
    u_indices, _, _ = test_set.uir_tuple
    ts_by_row = list(test_set.timestamps)

    per_instance = []
    n_future = 0
    n_total = 0
    for u_idx, ts in zip(u_indices, ts_by_row):
        raw_user = uid2raw.get(int(u_idx))
        if raw_user is None:
            continue
        try:
            recs = model.recommend(user_id=raw_user, k=k, remove_seen=False)
        except Exception:
            continue
        f = sum(1 for it in recs if item_first.get(it, 0) > ts)
        per_instance.append(f / max(len(recs), 1))
        n_future += f
        n_total += len(recs)
    return {
        "mean_pct_per_instance": float(np.mean(per_instance)) if per_instance else 0.0,
        "global_pct": n_future / n_total if n_total else 0.0,
        "evaluated": len(per_instance),
    }


def future_items_pct_batched(model, eval_method, item_first, k, chunk=512):
    """Vectorized version of future_items_pct.

    Calls model.score(u_idx) (length num_items vector) in user chunks, takes
    top-K via argpartition, checks future-item count via item_first array.
    Identical integer counts to the per-user-recommend loop above except for
    the negligible-probability case of exact float-score ties on the K/K+1
    boundary.
    """
    test_set = eval_method.test_set
    iid2raw = {v: k_ for k_, v in test_set.iid_map.items()}
    u_indices, _, _ = test_set.uir_tuple
    ts_by_row = np.asarray(list(test_set.timestamps), dtype=np.int64)
    u_arr = np.asarray(u_indices, dtype=np.int64)

    # Precompute first-seen timestamp per item INDEX (mapped via iid2raw).
    num_items = test_set.num_items
    item_first_arr = np.zeros(num_items, dtype=np.int64)
    for idx in range(num_items):
        raw = iid2raw.get(idx)
        if raw is not None:
            item_first_arr[idx] = item_first.get(raw, 0)

    n_future = 0
    n_total = 0
    per_instance_sum = 0.0
    n_evaluated = 0

    for start in range(0, len(u_arr), chunk):
        end = min(start + chunk, len(u_arr))
        u_chunk = u_arr[start:end]
        ts_chunk = ts_by_row[start:end]

        scores_rows = []
        keep = []
        for i, u_idx in enumerate(u_chunk):
            try:
                s = model.score(int(u_idx))
            except Exception:
                continue
            scores_rows.append(np.asarray(s, dtype=np.float32))
            keep.append(i)
        if not scores_rows:
            continue
        scores = np.stack(scores_rows, axis=0)  # (b, num_items)
        ts_kept = ts_chunk[keep]

        # top-K item indices per row (unsorted; set membership only).
        if k >= scores.shape[1]:
            topk = np.tile(np.arange(scores.shape[1]), (scores.shape[0], 1))
        else:
            topk = np.argpartition(scores, -k, axis=1)[:, -k:]
        first_seen_topk = item_first_arr[topk]                 # (b, k)
        future_mask = first_seen_topk > ts_kept[:, None]       # (b, k)
        per_row_future = future_mask.sum(axis=1)               # (b,)

        n_future += int(per_row_future.sum())
        n_total += int(topk.size)
        per_instance_sum += float((per_row_future / max(k, 1)).sum())
        n_evaluated += len(keep)

    return {
        "mean_pct_per_instance": per_instance_sum / n_evaluated if n_evaluated else 0.0,
        "global_pct": n_future / n_total if n_total else 0.0,
        "evaluated": n_evaluated,
    }
