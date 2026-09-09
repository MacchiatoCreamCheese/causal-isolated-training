"""Prove that cornac's stock models actually receive causal negatives.

This is the claim the whole design rests on: the causal rule lives in the data
loader, so unmodified cornac models inherit it without knowing. If this test
passes, `TimeAwareDataset` is genuinely serving cornac's training loops; if it
regresses (say cornac changes which iterator a model uses), the ablation would
silently become a no-op and every result would be wrong in a way the accuracy
numbers alone would not reveal.

Runs on synthetic data, so it needs no dataset CSVs and finishes in seconds.

  - `uij_iter`  is LightGCN's entry point (cornac/models/lightgcn/recom_lightgcn.py)
  - `uir_iter`  is the NCF family's entry point (cornac/models/ncf/recom_ncf_base.py)

A run that could not execute every check never reports a plain pass: the
LightGCN checks need dgl, and without it the test *fails* unless the operator
explicitly opts out with `--allow-missing-dgl` (or RESEARCH_ALLOW_MISSING_DGL=1),
in which case the summary says how many checks were skipped. Silence there used
to be the failure mode: on native Windows dgl is never installed, so the routine
local run reported "ALL CHECKS PASSED" having exercised only 8 of 10 checks --
and the missing ones covered `uij_iter`, whose rejection rule differs most from
cornac's.

Usage:  python -m research.smoke.test_cornac_causal [--allow-missing-dgl]
"""

import argparse
import os
import sys

from .fixtures import build_rated_split, build_split, negatives_only

# 2 sampling modes x 4 per-mode checks, plus 2 rejection-rule checks and the
# pre-trained-NeuMF check.
TOTAL_CHECKS = 11

ALLOW_ENV = "RESEARCH_ALLOW_MISSING_DGL"


class Results:
    """Tally of what actually ran, so the summary cannot overstate it."""

    def __init__(self):
        self.passed = []
        self.failed = []
        self.skipped = []

    def record(self, label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        (self.passed if ok else self.failed).append(label)

    def check(self, label, train_set, expect_zero):
        """Record a counterfactual-rate assertion: zero under causal sampling,
        strictly positive under uniform.

        Also requires that negatives were actually drawn through our loader.
        Under causal sampling rho is 0 by construction -- every positive's own
        item is in its causal prefix, so the prefix is never empty and no future
        item can be drawn -- which makes `rho == 0` a weak assertion on its own:
        `counterfactual_rate` returns 0.0 when *no draw happened at all*, so a
        model that bypassed `TimeAwareDataset` entirely would pass it. That
        bypass is the exact regression this file exists to catch, so the draw
        count is asserted alongside the rate.
        """
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
    """The rejection filter must differ per model, matching cornac.

    cornac's `uij_iter` rejects a negative only when the user rated it at least
    as highly as the positive, so their lower-rated items stay eligible;
    `uir_iter` rejects any observed item, because a pointwise loss would label
    it 0. Uses the rated fixture -- on all-1.0 ratings the two rules coincide
    and this check could not tell them apart.
    """
    print("\n=== rejection rules (rating-aware vs any-observed) ===")
    train_set = build_rated_split("uniform").train_set
    num_items = train_set.num_items
    observed = set(
        (int(u) * num_items + int(i))
        for u, i in zip(train_set.uir_tuple[0], train_set.uir_tuple[1])
    )

    def count_own(batches):
        """How many emitted negatives are items the user actually interacted
        with. The caller resets the probe first, so `train_set.residual_collisions`
        afterwards covers exactly this pass."""
        return sum(1
                   for users, negs in batches
                   for u, j in zip(users, negs)
                   if int(u) * num_items + int(j) in observed)

    # The fixture is deliberately dense, so MAX_REJECT_ROUNDS genuinely runs out
    # sometimes and a few of the user's own items survive. Those are counted as
    # residuals, so the assertion is not "no own items" but "none beyond the
    # documented residual" -- otherwise this would test the redraw budget rather
    # than the filter.
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
    """The four per-mode checks: both iterators, then both cornac models.

    Factored out of `main()` so the pytest wrappers at the bottom exercise
    exactly the code path the standalone script does, rather than a parallel
    copy that could drift from it.
    """
    expect_zero = mode == "causal"
    print(f"\n=== neg_sampling={mode!r} ===")
    train_set = build_split(mode).train_set

    # 1. LightGCN's iterator, called the way cornac calls it: no mode named.
    train_set.reset_counterfactual_counters()
    for _bu, _bi, _bj in train_set.uij_iter(batch_size=512, shuffle=True):
        pass
    res.check("uij_iter (LightGCN path)", train_set, expect_zero)

    # 2. NCF's iterator, with cornac's own argument shape.
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

    # 3. The real thing: an unmodified cornac NeuMF, one epoch.
    from cornac.models import NeuMF
    train_set.reset_counterfactual_counters()
    NeuMF(num_factors=8, layers=(64, 32, 16, 8), num_epochs=1,
          batch_size=256, num_neg=num_neg, backend="pytorch",
          seed=42, verbose=False).fit(train_set)
    res.check("cornac NeuMF.fit", train_set, expect_zero)

    # 4. Same for LightGCN, which reaches the loader by the other route.
    #    Needs dgl -- absent on native Windows, see the environment note in
    #    README.md. Missing dgl is a failure unless explicitly waived.
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
    """Pre-training must sample through the loader too.

    `pretrain` ships off (RESEARCH_NEUMF_PRETRAIN), so nothing in a normal run
    exercises this path -- which is exactly why it needs a check: a capability
    that is never run is one that breaks quietly before anyone gets to use it.

    With pre-training on, a cell trains three times (GMF, then MLP, then the
    fused NeuMF). All three subclass cornac's NCFBase and draw through
    `train_set.uir_iter(...)`, so all three must respect the causal rule. A
    non-zero rho here would mean a pre-training phase bypassed the loader --
    the same class of silent regression the rest of this file exists to catch,
    just on a newer code path.
    """
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


# ---------------------------------------------------------------------------
# pytest entry points.
#
# The checks run standalone (`python -m ...`), which is how the README drives
# them and how they print their per-check detail lines. These wrappers expose the
# same functions to `pytest research/smoke`, so CI need not shell out and parse
# stdout. No import-time pytest dependency: a bare `def test_*()` is all pytest
# needs to collect a test, so the module still runs with pytest absent.
#
# Missing dgl stays a *failure* here, exactly as in the CLI -- without it the
# `uij_iter` path goes unverified, which is the regression this file exists to
# catch. Waive it the same way: RESEARCH_ALLOW_MISSING_DGL=1.
# ---------------------------------------------------------------------------

def _waived():
    return os.environ.get(ALLOW_ENV, "") == "1"


def test_uniform_arm_draws_counterfactual_negatives():
    """Vanilla sampling must visibly draw not-yet-existent items — otherwise
    the fixture is not exercising the leakage the causal arm removes."""
    res = Results()
    check_sampling_mode(res, "uniform", have_dgl(), _waived())
    assert not res.failed, res.failed


def test_causal_arm_draws_none():
    """The claim: cornac's stock models, sampling through our loader, never
    see a negative that post-dates its positive."""
    res = Results()
    check_sampling_mode(res, "causal", have_dgl(), _waived())
    assert not res.failed, res.failed


def test_pretrained_neumf_samples_causally():
    """The opt-in pre-training path still draws through TimeAwareDataset."""
    res = Results()
    check_pretrained_neumf(res)
    assert not res.failed, res.failed


def test_rejection_rules_match_cornac():
    """`uij_iter` is rating-aware, `uir_iter` is any-observed — per cornac."""
    res = Results()
    check_rejection_rules(res)
    assert not res.failed, res.failed


if __name__ == "__main__":
    main()
