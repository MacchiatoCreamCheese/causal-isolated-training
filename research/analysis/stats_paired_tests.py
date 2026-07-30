"""Paired statistical tests on prequential HR/NDCG/Recall.

For each (dataset, slice) we have two observations: vanilla
(shuffle+uniform) and faithful (win10+causal). With 3 datasets * 5
slices = 15 paired observations per metric. We run:

  - Paired t-test (scipy.stats.ttest_rel)
  - Wilcoxon signed-rank (scipy.stats.wilcoxon) -- non-parametric
    fallback in case HR is not Gaussian, which it usually isn't.
  - Sign test (just count positive deltas, exact binomial p-value).

We report each test (i) pooled across all 15 paired observations and
(ii) per-dataset across the 5 slices.

Multi-seed extension: if multiple seed JSONs exist for the same
dataset, every (slice, seed) pair becomes one paired observation, so
pooled n grows from 15 to (5 * S * D) where S is seeds, D is datasets.
The script handles this automatically.

Input: research/results/preq_*.json (per-cell aggregated JSON,
written by research.runners.prequential).
Output: stdout + research/reports/STATS_paired_tests.md.

Usage:
  python -m research.analysis.stats_paired_tests
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


# research/analysis/stats_paired_tests.py — three .parent steps reach repo root.
ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / "research" / "results"
OUTPUT_MD = ROOT / "research" / "reports" / "STATS_paired_tests.md"

METRICS = ["HitRatio@20", "NDCG@20", "Recall@20"]
VANILLA = "shuffle+uniform"
FAITHFUL = "win10+causal"


def load_payloads() -> List[dict]:
    payloads = []
    for path in sorted(RESULTS_DIR.glob("preq_*.json")):
        with open(path) as f:
            payloads.append(json.load(f))
    return payloads


def collect_pairs(payloads: List[dict]) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Returns: metric -> dataset -> list of (vanilla, faithful) per slice/seed."""
    by_metric: Dict[str, Dict[str, List[Tuple[float, float]]]] = {
        m: defaultdict(list) for m in METRICS
    }
    for payload in payloads:
        ds = Path(payload["dataset_csv"]).stem
        cells_vanilla = sorted(payload["cells"].get(VANILLA, []),
                               key=lambda c: c["slice"])
        cells_faithful = sorted(payload["cells"].get(FAITHFUL, []),
                                key=lambda c: c["slice"])
        if not cells_vanilla or not cells_faithful:
            continue
        for v, f in zip(cells_vanilla, cells_faithful):
            if v["slice"] != f["slice"]:
                continue  # paranoia
            for m in METRICS:
                vv = v["metrics"][m]
                ff = f["metrics"][m]
                by_metric[m][ds].append((vv, ff))
    return by_metric


def sign_test(deltas: np.ndarray) -> Tuple[int, int, float]:
    """Returns (n_positive, n_total, exact two-sided binomial p-value)."""
    nz = deltas[deltas != 0.0]
    n_pos = int((nz > 0).sum())
    n_total = int(nz.size)
    if n_total == 0:
        return 0, 0, 1.0
    # Two-sided exact under H0: p_pos = 0.5
    res = stats.binomtest(n_pos, n_total, p=0.5, alternative="two-sided")
    return n_pos, n_total, float(res.pvalue)


def fmt_p(p: float) -> str:
    if math.isnan(p):
        return "n/a"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def run_tests(pairs: List[Tuple[float, float]]) -> dict:
    if len(pairs) < 2:
        return {"n": len(pairs)}
    arr = np.asarray(pairs, dtype=float)
    vanilla = arr[:, 0]
    faithful = arr[:, 1]
    deltas = faithful - vanilla
    out = {
        "n": len(pairs),
        "mean_vanilla":   float(vanilla.mean()),
        "mean_faithful":  float(faithful.mean()),
        "mean_delta":     float(deltas.mean()),
        "median_delta":   float(np.median(deltas)),
        "n_positive":     int((deltas > 0).sum()),
        "n_negative":     int((deltas < 0).sum()),
    }
    # Paired t-test (parametric)
    t_stat, t_p = stats.ttest_rel(faithful, vanilla)
    out["ttest_stat"] = float(t_stat)
    out["ttest_p"] = float(t_p)
    # Wilcoxon signed-rank (non-parametric)
    try:
        w_stat, w_p = stats.wilcoxon(faithful, vanilla, zero_method="wilcox")
        out["wilcoxon_stat"] = float(w_stat)
        out["wilcoxon_p"] = float(w_p)
    except ValueError:
        out["wilcoxon_stat"] = float("nan")
        out["wilcoxon_p"] = float("nan")
    # Sign test (exact binomial)
    n_pos, n_tot, sign_p = sign_test(deltas)
    out["sign_n_positive"] = n_pos
    out["sign_n_total"] = n_tot
    out["sign_p"] = sign_p
    return out


def build_markdown(by_metric) -> str:
    lines = []
    lines.append("# Paired statistical tests on prequential metrics\n")
    lines.append(
        f"Comparison: `{FAITHFUL}` (faithful recipe) vs `{VANILLA}` (vanilla baseline). "
        "Each paired observation is one (dataset, prequential slice [, seed]) cell.\n"
    )

    for m in METRICS:
        lines.append(f"\n## {m}\n")
        per_ds = by_metric[m]
        all_pairs = [p for ds in per_ds for p in per_ds[ds]]
        if not all_pairs:
            lines.append("*No data.*\n")
            continue

        # Pooled
        pooled = run_tests(all_pairs)
        lines.append("### Pooled (all datasets and slices)\n")
        lines.append(f"- n = {pooled['n']}, n_positive = {pooled['n_positive']}/{pooled['n']} (faithful > vanilla)")
        lines.append(f"- mean delta (faithful - vanilla) = {pooled['mean_delta']:+.5f}, median delta = {pooled['median_delta']:+.5f}")
        lines.append(f"- paired t-test: t = {pooled['ttest_stat']:.3f}, p = {fmt_p(pooled['ttest_p'])}")
        lines.append(f"- Wilcoxon signed-rank: W = {pooled['wilcoxon_stat']:.2f}, p = {fmt_p(pooled['wilcoxon_p'])}")
        lines.append(f"- Sign test: {pooled['sign_n_positive']}/{pooled['sign_n_total']} positive, p = {fmt_p(pooled['sign_p'])}\n")

        # Per-dataset
        lines.append("### Per-dataset\n")
        lines.append("| Dataset | n | mean(van) | mean(fai) | mean delta | t-test p | Wilcoxon p | Sign |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for ds in sorted(per_ds):
            r = run_tests(per_ds[ds])
            if r.get("n", 0) < 2:
                lines.append(f"| {ds} | {r.get('n', 0)} | – | – | – | – | – | – |")
                continue
            lines.append(
                f"| {ds} | {r['n']} | "
                f"{r['mean_vanilla']:.4f} | "
                f"{r['mean_faithful']:.4f} | "
                f"{r['mean_delta']:+.4f} | "
                f"{fmt_p(r['ttest_p'])} | "
                f"{fmt_p(r['wilcoxon_p'])} | "
                f"{r['sign_n_positive']}/{r['sign_n_total']} |"
            )

    lines.append("\n## Reading guide\n")
    lines.append("- The **paired t-test** assumes the paired differences are Gaussian; safer for large n.")
    lines.append("- The **Wilcoxon signed-rank test** is the non-parametric alternative and does not require normality.")
    lines.append("- The **sign test** is the weakest of the three but the most robust to outliers; it asks only whether the faithful recipe wins more often than chance.")
    lines.append("- With our current single-seed sweep, n per cell is small (1 seed × 5 slices = 5 paired observations per dataset). Multi-seed runs would grow n linearly and tighten the p-values without rerunning the existing seed.")
    lines.append("- Per-dataset p-values are uncorrected; if you want family-wise control across the three datasets, multiply by 3 (Bonferroni).\n")
    return "\n".join(lines) + "\n"


def main():
    payloads = load_payloads()
    if not payloads:
        print(f"No JSON files in {RESULTS_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(payloads)} payload(s):", flush=True)
    for p in payloads:
        ds = Path(p["dataset_csv"]).stem
        seed = p.get("seed", "?")
        print(f"  {ds} (seed={seed})", flush=True)

    by_metric = collect_pairs(payloads)
    md = build_markdown(by_metric)
    OUTPUT_MD.write_text(md, encoding="utf-8")

    # Also dump headline to stdout for at-a-glance reading.
    print()
    print(md)
    print(f"\nWrote {OUTPUT_MD}", flush=True)


if __name__ == "__main__":
    main()
