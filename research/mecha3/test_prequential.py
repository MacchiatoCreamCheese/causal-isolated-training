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
    print("\n=== a batch is scored before it is trained on ===")
    records, _adapter, train_set = _run("causal", warmup_frac=0.0, warmup_epochs=0)
    first = records[0]["HitRatio@20"]
    chance = 20.0 / train_set.num_items
    res.record("first point is near chance with no warm-up",
               first < 4 * chance,
               f": HR@20={first:.4f} vs chance {chance:.4f} "
               f"({first / chance:.1f}x)")


def check_positive_is_never_masked(res):
    print("\n=== the scored item is never treated as already-seen ===")
    scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.3]])
    pos = np.array([1, 0])
    mask = np.ones_like(scores, dtype=bool)
    ranks = _rank_of_positive(scores.copy(), pos, mask)
    res.record("positive survives a fully-masked row", bool((ranks == 1).all()),
               f": ranks={ranks.tolist()} (want all 1)")

    ranks = _rank_of_positive(scores.copy(), np.array([0, 1]), None)
    res.record("rank counts strictly-higher scores", ranks.tolist() == [3, 3],
               f": ranks={ranks.tolist()} (want [3, 3])")


def check_arms_diverge(res):
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
                   and 0.0 <= r["AUC"] <= 1.0 for r in ru + rc)
    res.record("metrics lie in [0, 1]", in_range, "")
    above = all(r["AUC"] >= r["HitRatio@20"] for r in ru + rc)
    res.record("AUC is at least HR@20", above, "")


def check_neumf_adapter(res):
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
