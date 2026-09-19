import os
from pathlib import Path

import numpy as np

METRICS = ("HitRatio@20", "NDCG@20", "Recall@20", "AUC")


def from_experiment(exp, tol=1e-6):
    res = exp.result[0]
    per_user = res.metric_user_results
    users = sorted(per_user[METRICS[1]])
    idx_to_raw = {idx: raw for raw, idx in exp.eval_method.test_set.uid_map.items()}

    scores = {
        "user_idx": np.asarray(users, dtype=np.int64),
        "raw_user_id": np.asarray([str(idx_to_raw.get(u, "")) for u in users]),
    }
    for m in METRICS:
        if m not in per_user:
            continue
        values = np.asarray([per_user[m][u] for u in users], dtype=np.float64)
        reported = float(res.metric_avg_results[m])
        if abs(float(values.mean()) - reported) > tol:
            raise ValueError(f"per-user mean of {m} is {values.mean():.8f}, but the "
                             f"experiment reported {reported:.8f}")
        scores[m] = values
    return scores


def save(path, scores):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[:-len(".npz")] + ".tmp.npz")
    np.savez_compressed(tmp, **scores)
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
