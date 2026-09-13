"""Per-user test scores: the input to every paired significance test.

Three things would make a paired t-test silently wrong, and each is checked:

  - scores that do not average to the number in the table (a test on the wrong
    run's scores);
  - two rungs scored on different users (nothing to pair);
  - the tuner not returning them, so a tuning winner could not stand in for a
    ladder rung.

Runs on the synthetic fixture in a few seconds; no dataset needed.
"""

import numpy as np
import pytest

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib import user_scores
from ..lib.ablation_harness import PROBE_METRICS
from ..lib.bpr_cpu import BPRMiniBatch
from ..runners import tuning
from .fixtures import build_split


def _experiment(eval_method, sampler, tmp_path):
    model = BPRMiniBatch(name=f"t/{sampler}", k=8, batch_size=256,
                         learning_rate=0.05, n_epochs=3, sampler=sampler,
                         seed=0, verbose=False)
    exp = cornac.Experiment(eval_method=eval_method, models=[model],
                            metrics=[HitRatio(k=20), NDCG(k=20), Recall(k=20)],
                            user_based=True, save_dir=str(tmp_path),
                            verbose=False)
    exp.run()
    return exp


def test_scores_average_to_the_reported_metric_and_round_trip(tmp_path):
    em = build_split("uniform", seed=0)
    exp = _experiment(em, "uniform", tmp_path)
    scores = user_scores.from_experiment(exp)

    assert len(scores["user_idx"]) > 0
    for m in user_scores.METRICS:
        assert scores[m].shape == scores["user_idx"].shape
        assert np.isclose(scores[m].mean(), exp.result[0].metric_avg_results[m])

    path = tmp_path / "cell.npz"
    user_scores.save(path, scores)
    back = user_scores.load(path)
    assert set(back) == set(scores)
    for k in scores:
        assert np.array_equal(back[k], scores[k])


def test_mismatched_average_is_refused(tmp_path):
    em = build_split("uniform", seed=0)
    exp = _experiment(em, "uniform", tmp_path)
    exp.result[0].metric_avg_results["NDCG@20"] += 0.01
    with pytest.raises(ValueError):
        user_scores.from_experiment(exp)


def test_two_rungs_score_the_same_users(tmp_path):
    em = build_split("uniform", seed=0)
    a = user_scores.from_experiment(_experiment(em, "uniform", tmp_path))
    b = user_scores.from_experiment(_experiment(em, "causal", tmp_path))
    assert np.array_equal(a["user_idx"], b["user_idx"])
    assert np.array_equal(a["raw_user_id"], b["raw_user_id"])


def test_tuner_trial_returns_scores_and_probes(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "logs_dir", lambda: str(tmp_path))
    em = build_split("uniform", seed=0)
    config = dict(k_embed_dim=8, batch_size=256, learning_rate=0.05,
                  lambda_shared=1e-4, n_epochs=3)
    scope = tuning.Scope(dataset="musical", recipe="uniform", seed=0)
    val, test, users, probes, _wall = tuning._train_once("bpr", config, em, scope)

    assert np.isclose(users["NDCG@20"].mean(), test["NDCG@20"])
    assert set(probes) == set(PROBE_METRICS)
    # The uniform arm draws not-yet-existing items on this fixture; a zero here
    # would mean the probe was read off an object that never sampled.
    assert probes["counterfactual_rate"] > 0.0
