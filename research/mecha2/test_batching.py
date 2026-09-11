"""Prove that temporal batching regroups rows without changing which rows run.

The mechanism's whole claim is that `shuffle`, `coherent` and `temporal` differ in
**grouping and visit order only**. If any mode consumed a different set of rows --
dropped a tail batch, double-counted, silently truncated -- then the arms would
also differ in training volume, and every accuracy comparison between them would
be measuring that instead. That is the check this file exists for; the rest are
supporting.

Runs on the synthetic fixture, so it needs no dataset CSVs and finishes in
seconds.

Named `test_batching.py`, not `smoke.py`: pytest collects `test_*.py`, and the
wrappers at the bottom are silently never run under any other name -- which is
how this file first shipped.

Usage:  python -m research.mecha2.test_batching
"""

import sys

import numpy as np

from ..lib.timeaware_data import TimeAwareDataset
from ..smoke.fixtures import build_split
from .batching import BATCH_ORDERS, TemporalBatchDataset

BATCH = 256


class Results:
    """Tally of what actually ran, so the summary cannot overstate it."""

    def __init__(self):
        self.passed = []
        self.failed = []

    def record(self, label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}{detail}")
        (self.passed if ok else self.failed).append(label)


def make(batch_order, neg_sampling="causal", seed=0):
    split = build_split(neg_sampling, seed=seed)
    return TemporalBatchDataset.from_dataset(
        split.train_set, neg_sampling=neg_sampling, batch_order=batch_order)


def batches_of(train_set):
    """One epoch of `uij_iter` batches, as lists of row indices."""
    n = len(train_set.uir_tuple[0])
    return [np.asarray(b) for b in train_set.idx_iter(n, BATCH, shuffle=True)]


def check_same_rows(res):
    """Every mode must consume the same rows -- the load-bearing check."""
    print("\n=== all three orders consume identical rows ===")
    seen = {}
    for order in BATCH_ORDERS:
        rows = np.concatenate(batches_of(make(order)))
        seen[order] = rows
    baseline = np.sort(seen["shuffle"])
    n = len(baseline)
    for order in BATCH_ORDERS:
        rows = seen[order]
        ok = len(rows) == n and np.array_equal(np.sort(rows), baseline)
        res.record(f"{order}: same rows as cornac's shuffle", ok,
                   f": {len(rows):,} rows, {len(np.unique(rows)):,} distinct "
                   f"(want {n:,} / {len(np.unique(baseline)):,})")


def check_contiguous_in_time(res):
    """`temporal` batches must be bands: no batch may start before the last ended."""
    print("\n=== temporal batches are contiguous bands of the timeline ===")
    train_set = make("temporal")
    ts = train_set._ts_array
    prev_max = None
    violations = 0
    spans = []
    for batch in batches_of(train_set):
        lo, hi = ts[batch].min(), ts[batch].max()
        spans.append((lo, hi))
        if prev_max is not None and lo < prev_max:
            violations += 1
        prev_max = hi
    res.record("temporal: batches are non-overlapping and in time order",
               violations == 0,
               f": {len(spans)} batches, {violations} out of order")
    # A band must also be *narrow* relative to the whole span, or "coherent"
    # means nothing. Compare the median batch span to the full training span.
    full = ts.max() - ts.min()
    median_span = float(np.median([hi - lo for lo, hi in spans]))
    res.record("temporal: a batch spans a small slice of the timeline",
               median_span < full / 10,
               f": median batch span {median_span / max(full, 1) * 100:.2f}% of total")


def check_coherent_is_same_bands(res):
    """`coherent` must be `temporal`'s bands in a different visit order."""
    print("\n=== coherent = same bands, shuffled visit order ===")
    temporal = {tuple(sorted(b.tolist())) for b in batches_of(make("temporal"))}
    coh_batches = batches_of(make("coherent"))
    coherent = {tuple(sorted(b.tolist())) for b in coh_batches}
    res.record("coherent: identical batch contents to temporal",
               temporal == coherent,
               f": {len(temporal)} bands, {len(temporal & coherent)} shared")

    # ...and genuinely reordered, or it is just `temporal` under another name.
    order_differs = [tuple(sorted(b.tolist())) for b in coh_batches] != \
        [tuple(sorted(b.tolist())) for b in batches_of(make("temporal"))]
    res.record("coherent: visit order actually differs from temporal",
               order_differs, "")


def check_sampler_untouched(res):
    """Mechanism 2 must not disturb Mechanism 1: rho depends on neg_sampling only."""
    print("\n=== batch_order does not touch the negative sampler ===")
    for neg_sampling, expect_zero in (("uniform", False), ("causal", True)):
        rates = {}
        for order in BATCH_ORDERS:
            train_set = make(order, neg_sampling=neg_sampling)
            train_set.reset_counterfactual_counters()
            for _bu, _bi, _bj in train_set.uij_iter(batch_size=BATCH, shuffle=True):
                pass
            rates[order] = (train_set.counterfactual_rate,
                            train_set.total_negatives)
        drew = all(n > 0 for _r, n in rates.values())
        if expect_zero:
            ok = drew and all(r == 0.0 for r, _n in rates.values())
        else:
            ok = drew and all(r > 0.0 for r, _n in rates.values())
        res.record(f"neg_sampling={neg_sampling!r}: rho unchanged by batch_order", ok,
                   ": " + ", ".join(f"{o}={r * 100:.2f}%" for o, (r, _n) in rates.items()))


def check_shuffle_matches_cornac(res):
    """`shuffle` must delegate, not reimplement."""
    print("\n=== shuffle mode is cornac's own path ===")
    split = build_split("causal", seed=0)
    plain = TimeAwareDataset.from_dataset(split.train_set, neg_sampling="causal")
    ours = make("shuffle")
    n = len(plain.uir_tuple[0])
    a = np.sort(np.concatenate([np.asarray(b) for b in plain.idx_iter(n, BATCH, True)]))
    b = np.sort(np.concatenate(batches_of(ours)))
    res.record("shuffle: same rows as an un-upgraded TimeAwareDataset",
               np.array_equal(a, b), f": {len(a):,} rows each")


def check_bpr_gets_bands(res):
    """Our BPR batches itself, so it needs the `_epoch_batches` seam.

    Two things must hold. With a temporal split, `TemporalBPR` must actually
    receive time bands -- otherwise a cell labelled "temporal" trained on
    shuffled rows, the exact silent no-op this suite exists to catch. And with a
    `shuffle` split it must fall back to the parent's own permutation, so the
    baseline arm stays bit-identical to a plain `BPRMiniBatch` rather than
    quietly switching which RNG shuffles it.
    """
    print("\n=== BPR reaches Mechanism 2 through _epoch_batches ===")
    from ..lib.bpr_cpu import BPRMiniBatch
    from .bpr import TemporalBPR

    temporal = make("temporal")
    n = len(temporal.uir_tuple[0])
    ts = temporal._ts_array

    model = TemporalBPR(name="probe", k=8, batch_size=BATCH, seed=42, verbose=False)
    batches = [np.asarray(b) for b in model._epoch_batches(temporal, n)]
    rows = np.concatenate(batches)
    prev_max, violations = None, 0
    for b in batches:
        lo, hi = ts[b].min(), ts[b].max()
        if prev_max is not None and lo < prev_max:
            violations += 1
        prev_max = hi
    res.record("TemporalBPR: batches are time bands", violations == 0,
               f": {len(batches)} batches, {violations} out of order, "
               f"{len(rows):,} rows")
    res.record("TemporalBPR: consumes every row exactly once",
               len(rows) == n and len(np.unique(rows)) == n,
               f": {len(np.unique(rows)):,}/{n:,} distinct")

    # Fallback must match the parent exactly, same seed, same RNG stream.
    shuffled = make("shuffle")
    a = TemporalBPR(name="a", k=8, batch_size=BATCH, seed=42, verbose=False)
    b = BPRMiniBatch(name="b", k=8, batch_size=BATCH, seed=42, verbose=False)
    same = [np.array_equal(x, y) for x, y in zip(
        a._epoch_batches(shuffled, n), b._epoch_batches(shuffled, n))]
    res.record("TemporalBPR: shuffle mode is bit-identical to BPRMiniBatch",
               len(same) > 0 and all(same),
               f": {sum(same)}/{len(same)} batches identical")


def main():
    res = Results()
    check_same_rows(res)
    check_contiguous_in_time(res)
    check_coherent_is_same_bands(res)
    check_sampler_untouched(res)
    check_shuffle_matches_cornac(res)
    check_bpr_gets_bands(res)

    print()
    ran = len(res.passed) + len(res.failed)
    if res.failed:
        print(f"FAILED: {len(res.failed)} check(s): {res.failed}")
        sys.exit(1)
    print(f"ALL CHECKS PASSED ({ran}/{ran}) — temporal batching regroups rows "
          f"without changing which rows train.")


# ---------------------------------------------------------------------------
# pytest entry points. No import-time pytest dependency: a bare `def test_*()`
# is all pytest needs, so this module still runs with pytest absent.
# ---------------------------------------------------------------------------

def test_all_orders_consume_identical_rows():
    res = Results()
    check_same_rows(res)
    assert not res.failed, res.failed


def test_temporal_batches_are_time_bands():
    res = Results()
    check_contiguous_in_time(res)
    assert not res.failed, res.failed


def test_coherent_is_temporal_reordered():
    res = Results()
    check_coherent_is_same_bands(res)
    assert not res.failed, res.failed


def test_batch_order_does_not_affect_sampler():
    res = Results()
    check_sampler_untouched(res)
    assert not res.failed, res.failed


def test_shuffle_delegates_to_cornac():
    res = Results()
    check_shuffle_matches_cornac(res)
    assert not res.failed, res.failed


def test_bpr_receives_temporal_batches():
    res = Results()
    check_bpr_gets_bands(res)
    assert not res.failed, res.failed


if __name__ == "__main__":
    main()
