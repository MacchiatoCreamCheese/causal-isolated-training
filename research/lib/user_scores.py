"""Per-user test scores: what a paired significance test needs.

A table reports one number per cell -- the mean NDCG@20 over test users. A paired
t-test needs the numbers that mean was taken over: every test user's score under
each rung, so each user can be compared with themselves. cornac computes them
anyway (`Result.metric_user_results`) and then only the averages are usually kept.

One `.npz` per trained model::

    user_idx      int64, sorted -- cornac's user index on this split
    raw_user_id   str           -- the dataset's own id, for joining across runs
    HitRatio@20, NDCG@20, Recall@20   float64, aligned with user_idx

**The file cannot disagree with the table.** `from_experiment` checks that each
metric's per-user mean equals the average cornac reported for the same run, and
raises if not, so a paired test can never be run on scores that do not belong to
the number printed next to it.
"""

import os
from pathlib import Path

import numpy as np

METRICS = ("HitRatio@20", "NDCG@20", "Recall@20")


def from_experiment(exp, tol=1e-6):
    """Per-user test scores of a finished single-model `cornac.Experiment`."""
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
    """Atomic compressed write; a kill mid-write leaves no half file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name[:-len(".npz")] + ".tmp.npz")
    np.savez_compressed(tmp, **scores)
    os.replace(tmp, path)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
