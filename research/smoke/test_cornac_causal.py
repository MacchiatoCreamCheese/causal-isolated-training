import argparse
import os
import sys

from .fixtures import build_rated_split, build_split, negatives_only

TOTAL_CHECKS = 11

ALLOW_ENV = "RESEARCH_ALLOW_MISSING_DGL"


class Results:
    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def record(self, label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        (self.passed if ok else self.failed).append(label)

    def check(self, label, train_set, expect_zero):
        drawn = train_set.total_negatives
        rho = train_set.counterfactual_rate
        want = "== 0%" if expect_zero else "> 0%"
        ok = drawn > 0 and ((rho == 0.0) if expect_zero else (rho > 0.0))
        self.record(label, ok, f": rho={rho * 100:.2f}%  (expected {want}) "
                               f"over {drawn:,} draws"
                               + ("  <-- NO DRAWS: loader was bypassed"
                                  if drawn == 0 else ""))

    def skip(self, label, why):
        print(f"  [SKIP] {label}: {why}")
        self.skipped.append(label)


def have_dgl():
    try:
        import dgl  # noqa: F401
    except ImportError:
        return False
    return True


def check_rejection_rules(res):
    print("\n=== rejection rules (rating-aware vs any-observed) ===")
    train_set = build_rated_split("uniform").train_set
    num_items = train_set.num_items
    observed = set(
        (int(u) * num_items + int(i))
        for u, i in zip(train_set.uir_tuple[0], train_set.uir_tuple[1])
    )

    def count_own(batches):
        return sum(1
                   for users, negs in batches
                   for u, j in zip(users, negs)
                   if int(u) * num_items + int(j) in observed)

    train_set.reset_counterfactual_counters()
    own = count_own((bu, bj) for bu, _bi, bj
                    in train_set.uij_iter(batch_size=1024, shuffle=True))
    residual = train_set.residual_collisions
    res.record("uij_iter keeps lower-rated own items", own > residual,
               f": {own:,} own items emitted vs {residual:,} unavoidable "
               f"residuals — the surplus is what cornac's rating-aware rule admits")

    num_zeros = 4
    train_set.reset_counterfactual_counters()
    own = count_own(negatives_only(bu, bi, num_zeros)
                    for bu, bi, _br in train_set.uir_iter(
                        batch_size=1024, shuffle=True, binary=True,
                        num_zeros=num_zeros))
    residual = train_set.residual_collisions
    res.record("uir_iter rejects every own item", own == residual,
               f": {own:,} own items emitted, all {residual:,} of them "
               f"unavoidable residuals (want equal)")


def check_sampling_mode(res, mode, dgl_ok, allow_missing_dgl):
    expect_zero = mode == "causal"
    print(f"\n=== neg_sampling={mode!r} ===")
    train_set = build_split(mode).train_set

    train_set.reset_counterfactual_counters()
    for _bu, _bi, _bj in train_set.uij_iter(batch_size=512, shuffle=True):
        pass
    res.check("uij_iter (LightGCN path)", train_set, expect_zero)

    train_set.reset_counterfactual_counters()
    n_pos = len(train_set.uir_tuple[0])
    num_neg = 4
    emitted = 0
    for bu, bi, br in train_set.uir_iter(batch_size=512, shuffle=True,
                                         binary=True, num_zeros=num_neg):
        assert len(bu) == len(bi) == len(br), "uir_iter emitted ragged batches"
        emitted += len(bu)
    expected = n_pos * (1 + num_neg)
    assert emitted == expected, (
        f"uir_iter emitted {emitted} rows, expected {expected} "
        f"({n_pos} positives x (1 + {num_neg} negatives))")
    res.check("uir_iter (NCF path)", train_set, expect_zero)

    from cornac.models import NeuMF
    train_set.reset_counterfactual_counters()
    NeuMF(num_factors=8, layers=(64, 32, 16, 8), num_epochs=1,
          batch_size=256, num_neg=num_neg, backend="pytorch",
          seed=42, verbose=False).fit(train_set)
    res.check("cornac NeuMF.fit", train_set, expect_zero)

    label = f"cornac LightGCN.fit [{mode}]"
    if dgl_ok:
        from cornac.models import LightGCN
        train_set.reset_counterfactual_counters()
        LightGCN(num_epochs=1, batch_size=1024, emb_size=64, num_layers=3,
                 seed=42, verbose=False).fit(train_set)
        res.check(label, train_set, expect_zero)
    elif allow_missing_dgl:
        res.skip(label, "dgl not installed (waived)")
    else:
        res.record(label, False,
                   ": dgl not installed -- this leaves the uij_iter path "
                   "unverified. Run under WSL (see README.md), or waive "
                   f"with --allow-missing-dgl / {ALLOW_ENV}=1")


def check_pretrained_neumf(res):
    print("\n=== pre-trained NeuMF (RESEARCH_NEUMF_PRETRAIN path) ===")
    from ..lib.cornac_compat import NeuMF

    train_set = build_split("causal").train_set
    train_set.reset_counterfactual_counters()
    model = NeuMF(name="pretrain-probe", num_factors=8, layers=(64, 32, 16, 8),
                  num_epochs=1, batch_size=256, num_neg=4, backend="pytorch",
                  learner="sgd", pretrain=True, seed=42, verbose=False)
    model.fit(train_set)

    drawn = train_set.total_negatives
    rho = train_set.counterfactual_rate
    ok = model.pretrained and drawn > 0 and rho == 0.0
    res.record("pre-trained NeuMF samples causally", ok,
               f": pretrained={model.pretrained} alpha={getattr(model, 'alpha', None)} "
               f"rho={rho * 100:.2f}% over {drawn:,} draws across 3 fits")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--allow-missing-dgl", action="store_true",
                    default=os.environ.get(ALLOW_ENV, "") == "1",
                    help="downgrade the LightGCN checks to skips when dgl is "
                         f"absent (or set {ALLOW_ENV}=1)")
    args = ap.parse_args()

    res = Results()
    dgl_ok = have_dgl()

    for mode in ("uniform", "causal"):
        check_sampling_mode(res, mode, dgl_ok, args.allow_missing_dgl)

    check_rejection_rules(res)
    check_pretrained_neumf(res)

    print()
    ran = len(res.passed) + len(res.failed)
    if res.failed:
        print(f"FAILED: {len(res.failed)} check(s): {res.failed}")
        sys.exit(1)
    if res.skipped:
        print(f"PASSED {ran}/{TOTAL_CHECKS} - {len(res.skipped)} skipped "
              f"({', '.join(res.skipped)}). Not a full pass.")
        return
    assert ran == TOTAL_CHECKS, (
        f"expected {TOTAL_CHECKS} checks, ran {ran} -- TOTAL_CHECKS is stale")
    print(f"ALL CHECKS PASSED ({ran}/{TOTAL_CHECKS}) "
          f"— cornac's models are sampling through the loader.")


def _waived():
    return os.environ.get(ALLOW_ENV, "") == "1"


def test_uniform_arm_draws_counterfactual_negatives():
    res = Results()
    check_sampling_mode(res, "uniform", have_dgl(), _waived())
    assert not res.failed, res.failed


def test_causal_arm_draws_none():
    res = Results()
    check_sampling_mode(res, "causal", have_dgl(), _waived())
    assert not res.failed, res.failed


def test_pretrained_neumf_samples_causally():
    res = Results()
    check_pretrained_neumf(res)
    assert not res.failed, res.failed


def test_rejection_rules_match_cornac():
    res = Results()
    check_rejection_rules(res)
    assert not res.failed, res.failed


if __name__ == "__main__":
    main()
