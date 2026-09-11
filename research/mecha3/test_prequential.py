"""Prove the prequential loop scores before it trains, and that the arms differ.

Two things could silently invalidate every point on the curve.

If a batch were trained on *before* being scored, the model would be tested on
data it had just fitted -- leakage of the most direct kind, and the resulting
curve would look excellent and mean nothing. `check_scores_before_training`
catches that.

If the two arms drew the same negatives, the comparison would be between a model
and itself, and the curve would show a difference that is pure seed noise.
`check_arms_diverge` catches that. Note the first point or two can legitimately
coincide: HR@20 over a few hundred rows is coarse (whole hits over a fixed
denominator), so a barely-trained model can land on the same integer count in
both arms before the parameters have pulled apart.

Named `test_*.py` so pytest collects it -- see the note in
`research/mecha2/test_batching.py`.

Usage:  python -m research.mecha3.test_prequential
"""

import sys

import numpy as np

from ..lib.bpr_cpu import BPRMiniBatch
from ..smoke.fixtures import build_split
from .adapters import build_adapter
from .prequential import _rank_of_positive, run_prequential

BATCH = 256


class Results:
    def __init__(self):
        self.passed = []
        self.failed = []

    def record(self, label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        (self.passed if ok else self.failed).append(label)


def _run(arm, warmup_frac=0.2, warmup_epochs=3, seed=42, model_name="bpr"):
    """One prequential pass, returning `(records, adapter, train_set)`.

    Defaults to BPR because it is the cheapest of the three on the synthetic
    fixture; `model_name` lets a check exercise the cornac-backed adapters, whose
    training loops are written here rather than imported and so are the ones most
    able to drift.
    """
    train_set = build_split("causal", seed=0).train_set
    if model_name == "bpr":
        model = BPRMiniBatch(name="p", k=16, batch_size=BATCH,
                             learning_rate=0.05, sampler=arm, seed=seed,
                             verbose=False)
    elif model_name == "neumf":
        from ..lib.cornac_compat import NeuMF
        model = NeuMF(name="p", num_factors=8, layers=(64, 32, 16, 8),
                      num_epochs=1, batch_size=BATCH, num_neg=2,
                      backend="pytorch", seed=seed, verbose=False)
    else:
        raise ValueError(model_name)
    adapter = build_adapter(model_name, model, arm)
    records = run_prequential(adapter, train_set, batch_size=BATCH,
                              warmup_frac=warmup_frac,
                              warmup_epochs=warmup_epochs, verbose=False)
    return records, adapter, train_set


def check_scores_before_training(res):
    """With no warm-up the first point must look untrained.

    A randomly-initialised model ranks the true item uniformly at random, so
    HR@20 should sit near `20 / num_items`. If the loop trained a batch before
    scoring it, the first point would be far above that -- the model would have
    just been fitted on the very rows being tested.
    """
    print("\n=== a batch is scored before it is trained on ===")
    records, _adapter, train_set = _run("causal", warmup_frac=0.0, warmup_epochs=0)
    first = records[0]["HitRatio@20"]
    chance = 20.0 / train_set.num_items
    res.record("first point is near chance with no warm-up",
               first < 4 * chance,
               f": HR@20={first:.4f} vs chance {chance:.4f} "
               f"({first / chance:.1f}x)")


def check_positive_is_never_masked(res):
    """The item being scored must not be masked as 'already seen'.

    It becomes part of the user's history only *after* this step. Masking it
    would push its score to -inf, drop it to last place, and make every metric
    read zero -- a failure that looks like a modelling result rather than a bug.
    """
    print("\n=== the scored item is never treated as already-seen ===")
    scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.3]])
    pos = np.array([1, 0])
    mask = np.ones_like(scores, dtype=bool)  # everything marked seen
    ranks = _rank_of_positive(scores.copy(), pos, mask)
    res.record("positive survives a fully-masked row", bool((ranks == 1).all()),
               f": ranks={ranks.tolist()} (want all 1)")

    # And an unmasked run must rank by score, not by index.
    ranks = _rank_of_positive(scores.copy(), np.array([0, 1]), None)
    res.record("rank counts strictly-higher scores", ranks.tolist() == [3, 3],
               f": ranks={ranks.tolist()} (want [3, 3])")


def check_arms_diverge(res):
    """uniform and causal must actually draw different negatives."""
    print("\n=== the two arms are genuinely different models ===")
    _ru, au, _ = _run("uniform")
    _rc, ac, _ = _run("causal")
    res.record("uniform draws counterfactual negatives, causal draws none",
               au.sampler.counterfactual_rate > 0.0
               and ac.sampler.counterfactual_rate == 0.0,
               f": uniform rho={au.sampler.counterfactual_rate * 100:.2f}%, "
               f"causal rho={ac.sampler.counterfactual_rate * 100:.2f}%")
    res.record("both arms drew the same number of negatives",
               au.sampler._total_negs == ac.sampler._total_negs,
               f": {au.sampler._total_negs:,} each")
    res.record("the fitted parameters differ",
               not np.allclose(au.model.i_factors, ac.model.i_factors),
               f": max |diff| = "
               f"{np.abs(au.model.i_factors - ac.model.i_factors).max():.5f}")


def check_arms_are_comparable(res):
    """Same points, same rows, same order -- or the curves are not comparable."""
    print("\n=== the arms are measured on identical batches ===")
    ru, _au, _ = _run("uniform")
    rc, _ac, _ = _run("causal")
    res.record("same number of points", len(ru) == len(rc),
               f": {len(ru)} vs {len(rc)}")
    same_spans = all(a["ts_start"] == b["ts_start"] and a["ts_end"] == b["ts_end"]
                     and a["n"] == b["n"] for a, b in zip(ru, rc))
    res.record("same batch spans and sizes", same_spans, "")
    ordered = all(ru[i]["ts_start"] <= ru[i + 1]["ts_start"]
                  for i in range(len(ru) - 1))
    res.record("points advance through time", ordered, "")
    in_range = all(0.0 <= r["HitRatio@20"] <= 1.0 and 0.0 <= r["NDCG@20"] <= 1.0
                   for r in ru + rc)
    res.record("metrics lie in [0, 1]", in_range, "")


def check_neumf_adapter(res):
    """The NeuMF adapter drives cornac's own module, and still obeys the arm.

    Its training loop is written in `adapters.py` rather than imported, so unlike
    the ablation path there is no cornac code guaranteeing the negatives are ours.
    A rho of zero on the uniform arm would mean the sampler was bypassed.
    """
    print("\n=== NeuMF adapter: cornac's module, our loop ===")
    for arm, expect_zero in (("uniform", False), ("causal", True)):
        recs, ad, _ = _run(arm, warmup_epochs=1, model_name="neumf")
        rho = ad.sampler.counterfactual_rate
        drew = ad.sampler._total_negs
        ok = len(recs) > 0 and drew > 0 and (
            (rho == 0.0) if expect_zero else (rho > 0.0))
        res.record(f"neumf/{arm}: ran and sampled through our sampler", ok,
                   f": {len(recs)} points, rho={rho * 100:.2f}% over {drew:,} draws")


def main():
    res = Results()
    check_scores_before_training(res)
    check_positive_is_never_masked(res)
    check_arms_diverge(res)
    check_arms_are_comparable(res)
    check_neumf_adapter(res)

    print()
    ran = len(res.passed) + len(res.failed)
    if res.failed:
        print(f"FAILED: {len(res.failed)} check(s): {res.failed}")
        sys.exit(1)
    print(f"ALL CHECKS PASSED ({ran}/{ran}) — prequential scores each batch "
          f"before training on it.")


def test_scores_before_training():
    res = Results()
    check_scores_before_training(res)
    assert not res.failed, res.failed


def test_positive_is_never_masked():
    res = Results()
    check_positive_is_never_masked(res)
    assert not res.failed, res.failed


def test_arms_diverge():
    res = Results()
    check_arms_diverge(res)
    assert not res.failed, res.failed


def test_neumf_adapter_runs_and_samples():
    res = Results()
    check_neumf_adapter(res)
    assert not res.failed, res.failed


def test_arms_are_comparable():
    res = Results()
    check_arms_are_comparable(res)
    assert not res.failed, res.failed


if __name__ == "__main__":
    main()
