"""Show that our vectorized sampler is cornac's sampler, measured not assumed.

`TimeAwareDataset` replaces cornac's `uij_iter` / `uir_iter` wholesale, so the
uniform arm of the ablation is our code, not cornac's. That arm is the baseline
every reported effect is measured against, and "we reimplemented it faithfully"
is a claim worth backing with evidence rather than a source-code reading.

What is and is not checkable
----------------------------

Not checkable: an identical random stream. cornac draws one negative at a time
and its redraw `while` consumes a data-dependent number of values, so where the
generator sits when element i+1 starts depends on how many redraws element i
happened to need. A batched draw cannot reproduce that ordering, and matching it
would cost roughly 8x the sampling time (measured) to buy nothing -- the causal
arm has no cornac counterpart to match anyway.

Checkable, and what this test does:

  1. **Legality, exactly.** Every negative we emit must be one cornac's own
     sampler would have accepted, evaluated against `train_set.dok_matrix` --
     cornac's own data structure, using cornac's own predicate. Deterministic:
     no tolerance, no flake.
  2. **Distribution, loosely.** Our per-item draw frequencies against those of
     cornac's real per-element loop, by total-variation distance. Two correct
     samplers still differ run to run, so the bound is deliberately slack; this
     catches a wrong *pool*, not a wrong seed.
  3. **The causal conjunction.** Under `neg_sampling="causal"`, every negative
     must satisfy cornac's legality predicate *and* have existed at the
     positive's timestamp. That conjunction is the whole design claim.

Usage:
    python -m research.smoke.test_sampler_equivalence
    python -m research.smoke.test_sampler_equivalence --dataset musical
"""

import argparse
import sys
from itertools import repeat

import numpy as np
from cornac.data import Dataset

from .fixtures import build_rated_split, negatives_only, uij_batches

# How far the distance-to-cornac may exceed our own sampler's run-to-run
# self-distance. Both are measured on the same data at the same sample size, so
# this only has to absorb the wobble in the floor estimate itself -- a wrong
# item pool (say a causal prefix mistaken for the full catalog) lands far
# outside it.
TV_SLACK = 0.05

BATCH = 1024


def report(label, ok, detail, failures):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


# cornac's two rejection rules are one rule at different thresholds: reject
# when `dok[u,j] >= threshold`. `uij_iter` uses the positive's own rating;
# `uir_iter` uses "any rating at all", which is the smallest number above zero,
# since `dok_matrix` returns exactly 0.0 for an unobserved pair.
ANY_RATING = np.nextafter(0.0, 1.0)


def count_illegal(train_set, batches):
    """Count emitted negatives cornac's own predicate would have rejected.

    `batches` yields `(users, negatives, positives_or_None)`: a positives array
    means "reject at that item's rating" (cornac's `uij_iter` rule), `None`
    means "reject at any rating at all" (its `uir_iter` rule).

    Returns `(illegal, own_but_legal, total, residual)`, where `own_but_legal`
    counts the user's own items that cornac nonetheless admits (lower-rated
    than the positive) -- zero under the any-rating threshold by construction.
    The caller resets the probe, so `residual` covers exactly this pass.
    """
    dok = train_set.dok_matrix
    illegal = own_but_legal = total = 0
    for users, negs, positives in batches:
        thresholds = (repeat(ANY_RATING) if positives is None
                      else (dok[u, i] for u, i in zip(users, positives)))
        for u, j, threshold in zip(users, negs, thresholds):
            observed = dok[u, j]
            if observed >= threshold:
                illegal += 1
            elif observed > 0:
                own_but_legal += 1
        total += len(users)
    return illegal, own_but_legal, total, train_set.residual_collisions


def report_legality(label, illegal, residual, total, failures):
    """Assert that cornac-illegal negatives are *exactly* the counted residuals.

    Zero illegal negatives is not the right bar. cornac redraws in an unbounded
    `while`; we stop after MAX_REJECT_ROUNDS and accept whatever is left,
    because the earliest interaction in the log has a causal prefix of one item
    -- the positive itself -- and an unbounded loop would spin forever on it.
    So we do emit a few negatives cornac would have rejected. What must hold is
    that every one of them is accounted for by the residual-collision probe: if
    `illegal > residual`, the *filter* disagrees with cornac, which is the
    failure this test exists to catch. The residual rate is reported alongside
    so the size of the accepted deviation is visible rather than implied.
    """
    report(label, illegal == residual,
           f"{illegal:,}/{total:,} negatives are cornac-illegal; "
           f"{residual:,} counted as unavoidable residuals "
           f"({residual / max(total, 1) * 100:.2f}% of draws)", failures)


def check_uij_legality(train_set, failures):
    """Every `uij_iter` negative must satisfy cornac's `dok[u,j] < pos_rating`."""
    train_set.reset_counterfactual_counters()
    illegal, own_but_legal, total, residual = count_illegal(
        train_set, uij_batches(train_set, BATCH))
    report_legality("uij_iter legality (cornac's own predicate)", illegal,
                    residual, total, failures)
    # Not pass/fail -- reported because it is the whole reason the two rules
    # differ. A zero here would mean the data is not exercising the
    # rating-aware path at all, and the check above would prove nothing.
    print(f"         {own_but_legal:,} of the negatives are the user's own, "
          f"lower-rated items — admitted by cornac, and now by us")


def check_uir_legality(train_set, failures):
    """Every `uir_iter` negative must satisfy cornac's `dok[u,j] == 0`."""
    num_zeros = 4
    train_set.reset_counterfactual_counters()
    illegal, _own, total, residual = count_illegal(
        train_set,
        ((*negatives_only(bu, bi, num_zeros), None)
         for bu, bi, _br in train_set.uir_iter(
             batch_size=BATCH, shuffle=True, binary=True, num_zeros=num_zeros)))
    report_legality("uir_iter legality (cornac's own predicate)", illegal,
                    residual, total, failures)


def item_histogram(neg_batches, num_items):
    counts = np.zeros(num_items, dtype=np.int64)
    for negs in neg_batches:
        np.add.at(counts, np.asarray(negs, dtype=np.int64), 1)
    return counts


def check_distribution(train_set, failures, max_batches=40):
    """Our draw frequencies vs cornac's real per-element loop.

    The tolerance is *measured, not assumed*. Comparing two finite samples of a
    22k-item catalog, most bins hold a handful of draws, so two runs of the
    identical sampler are already far apart in total variation -- on
    Musical Instruments the floor is around 0.3, which a fixed threshold either
    sits above (proving nothing) or below (failing always). So we first draw
    twice from our own sampler to measure that noise floor, then require the
    distance to cornac to be no worse than it by more than `TV_SLACK`.

    cornac's loop runs ~80 us/draw against its scipy `dok_matrix` (135x ours),
    hence the batch cap: enough to compare shapes, not a full epoch.
    """
    num_items = train_set.num_items

    def collect(iterator):
        out = []
        for k, (_bu, _bi, bj) in enumerate(iterator):
            out.append(bj)
            if k + 1 == max_batches:
                break
        return item_histogram(out, num_items)

    def tv(a, b):
        p = a / max(a.sum(), 1)
        q = b / max(b.sum(), 1)
        return 0.5 * np.abs(p - q).sum()

    # Two independent draws from our own sampler: the RNG carries over between
    # calls, so these are genuinely different samples of the same distribution.
    ours_a = collect(train_set.uij_iter(batch_size=BATCH, shuffle=True))
    ours_b = collect(train_set.uij_iter(batch_size=BATCH, shuffle=True))
    theirs = collect(Dataset.uij_iter(train_set, batch_size=BATCH, shuffle=True,
                                      neg_sampling="uniform"))

    floor = tv(ours_a, ours_b)
    cross = tv(ours_a, theirs)
    report("draw distribution vs cornac's loop", cross <= floor + TV_SLACK,
           f"distance to cornac {cross:.4f} vs self-distance {floor:.4f} "
           f"(slack {TV_SLACK}, {ours_a.sum():,} draws each)", failures)


def check_causal_conjunction(train_set, failures):
    """Causal arm: negatives must be cornac-legal *and* already have existed.

    Legality is checked per row against `dok_matrix`; the "already existed" half
    is read off the dataset's own counterfactual probe, which is incremented
    inside the draw and so needs no re-derivation of per-row timestamps here.

    rho is 0 here by construction -- a positive's own item always sits in its
    causal prefix, so the prefix is never empty and no future item is reachable.
    That makes `rho == 0` worth little alone (it also holds when nothing was
    drawn), so the probe's own draw count is asserted against the rows we
    counted independently: if the loader were bypassed, or silently sampled
    fewer negatives than positives, the two would not agree.
    """
    train_set.reset_counterfactual_counters()
    illegal, _own, total, residual = count_illegal(
        train_set, uij_batches(train_set, BATCH))
    report_legality("causal arm is cornac-legal", illegal, residual, total,
                    failures)
    rho = train_set.counterfactual_rate
    drawn = train_set.total_negatives
    report("causal arm draws no future items", rho == 0.0 and drawn == total,
           f"counterfactual rate {rho * 100:.2f}%; probe saw {drawn:,} draws "
           f"vs {total:,} negatives emitted (want equal, and non-zero)",
           failures)


def load(dataset, neg_sampling):
    if dataset is None:
        return build_rated_split(neg_sampling).train_set
    from ..lib.data import build_eval_method
    return build_eval_method(dataset, neg_sampling=neg_sampling,
                             seed=42).train_set


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default=None,
                    help="dataset key (e.g. musical); default is the synthetic "
                         "rated fixture, which needs no CSV")
    args = ap.parse_args()

    failures = []
    source = args.dataset or "synthetic rated fixture"

    print(f"\n=== uniform arm vs cornac ({source}) ===")
    train_set = load(args.dataset, "uniform")
    print(f"  {len(train_set.uir_tuple[0]):,} interactions, "
          f"{train_set.num_users:,} users, {train_set.num_items:,} items")
    check_uij_legality(train_set, failures)
    check_uir_legality(train_set, failures)
    check_distribution(train_set, failures)

    print(f"\n=== causal arm ({source}) ===")
    train_set.neg_sampling = "causal"
    check_causal_conjunction(train_set, failures)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED — our filter agrees with cornac's on every draw; "
          "the only negatives cornac would have rejected are the counted "
          "bounded-rejection residuals.")


if __name__ == "__main__":
    main()
