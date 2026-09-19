import argparse
import sys
from itertools import repeat

import numpy as np
from cornac.data import Dataset

from .fixtures import build_rated_split, negatives_only, uij_batches

TV_SLACK = 0.05

BATCH = 1024


def report(label, ok, detail, failures):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: {detail}")
    if not ok:
        failures.append(label)


ANY_RATING = np.nextafter(0.0, 1.0)


def count_illegal(train_set, batches):
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
    report(label, illegal == residual,
           f"{illegal:,}/{total:,} negatives are cornac-illegal; "
           f"{residual:,} counted as unavoidable residuals "
           f"({residual / max(total, 1) * 100:.2f}% of draws)", failures)


def check_uij_legality(train_set, failures):
    train_set.reset_counterfactual_counters()
    illegal, own_but_legal, total, residual = count_illegal(
        train_set, uij_batches(train_set, BATCH))
    report_legality("uij_iter legality (cornac's own predicate)", illegal,
                    residual, total, failures)
    print(f"         {own_but_legal:,} of the negatives are the user's own, "
          f"lower-rated items — admitted by cornac, and now by us")


def check_uir_legality(train_set, failures):
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


def test_uniform_arm_matches_cornac():
    failures = []
    train_set = load(None, "uniform")
    check_uij_legality(train_set, failures)
    check_uir_legality(train_set, failures)
    check_distribution(train_set, failures)
    assert not failures, failures


def test_causal_arm_is_legal_and_time_respecting():
    failures = []
    train_set = load(None, "causal")
    check_causal_conjunction(train_set, failures)
    assert not failures, failures


if __name__ == "__main__":
    main()
