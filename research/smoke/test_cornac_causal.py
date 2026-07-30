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

Usage:  python -m research.smoke.test_cornac_causal
"""

import sys

import numpy as np

from ..lib.timeaware_data import CausalTimestampSplit


def build_split(neg_sampling):
    """Synthetic UIRT log where the catalog grows over time, so a
    time-blind sampler demonstrably draws not-yet-existent items."""
    rng = np.random.default_rng(0)
    rows = []
    for block in range(4):
        base_ts = 1_600_000_000_000 + block * 10_000_000_000
        n_available = (block + 1) * 40
        for _ in range(4000):
            u = int(rng.integers(0, 300))
            i = int(rng.integers(0, n_available))
            rows.append((f"u{u}", f"i{i}", 1.0,
                         base_ts + int(rng.integers(0, 5_000_000_000))))
    return CausalTimestampSplit(
        data=rows,
        val_timestamp=1_615_000_000_000,
        test_timestamp=1_625_000_000_000,
        fmt="UIRT", exclude_unknowns=True,
        neg_sampling=neg_sampling, verbose=False,
    )


def check(label, rho, expect_zero, failures):
    ok = (rho == 0.0) if expect_zero else (rho > 0.0)
    verdict = "PASS" if ok else "FAIL"
    want = "== 0%" if expect_zero else "> 0%"
    print(f"  [{verdict}] {label}: rho={rho * 100:.2f}%  (expected {want})")
    if not ok:
        failures.append(label)


def main():
    failures = []

    for mode in ("uniform", "causal"):
        expect_zero = mode == "causal"
        print(f"\n=== neg_sampling={mode!r} ===")
        train_set = build_split(mode).train_set

        # 1. LightGCN's iterator, called the way cornac calls it: no mode named.
        train_set.reset_counterfactual_counters()
        for _bu, _bi, _bj in train_set.uij_iter(batch_size=512, shuffle=True):
            pass
        check("uij_iter (LightGCN path)", train_set.counterfactual_rate,
              expect_zero, failures)

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
        check("uir_iter (NCF path)", train_set.counterfactual_rate,
              expect_zero, failures)

        # 3. The real thing: an unmodified cornac NeuMF, one epoch.
        from cornac.models import NeuMF
        train_set.reset_counterfactual_counters()
        NeuMF(num_factors=8, layers=(64, 32, 16, 8), num_epochs=1,
              batch_size=256, num_neg=num_neg, backend="pytorch",
              seed=42, verbose=False).fit(train_set)
        check("cornac NeuMF.fit", train_set.counterfactual_rate,
              expect_zero, failures)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED — cornac's models are sampling through the loader.")


if __name__ == "__main__":
    main()
