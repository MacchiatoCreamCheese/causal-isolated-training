"""Two-sample KS test on recommendation-recency distributions.

For each (dataset, prequential slice) we regenerate the raw recency-in-
days array for each recipe (vanilla = shuffle+uniform, faithful =
win10+causal), then run scipy.stats.ks_2samp comparing the two.

Recency: for each test instance at time t, the array of
  (t - first_seen(item))  /  days
over the model's top-K recommended items. We get one array per
(dataset, slice, recipe). KS tests whether the two distributions
(vanilla vs faithful, paired by slice) come from the same population.

Reading: a small p-value means the two recipes produce qualitatively
different recommendations, not just slightly different point metrics.
We Bonferroni-correct across the 15 (dataset, slice) cells when
reporting the family-wise summary.

Dependencies: this script loads pickled BPR models written by
research.runners.prequential's per-cell checkpointing. Only cells with
pickles available are tested; missing pickles are reported and
skipped.

Usage:
  python -m research.analysis.stats_ks_recency
"""

import argparse
import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from cornac.eval_methods.base_method import BaseMethod

from ..lib.faithfulness_metrics import recommendation_recency_distribution
from ..lib.causal_sampling import (
    TOP_K, load_uirt, item_first_seen,
)
from ..paths import DATA_DIR, RESULTS_DIR, REPORTS_DIR


RUNS_DIR = RESULTS_DIR / "runs"
OUTPUT_MD = REPORTS_DIR / "STATS_ks_recency.md"

VANILLA_SLUG = "shuffle_uniform"
FAITHFUL_SLUG = "win10_causal"

DATASET_CSV_FOR = {
    "Baby_Products":              "baby_dataset/Baby_Products.csv",
    "Cell_Phones_and_Accessories": "cellphone_dataset/Cell_Phones_and_Accessories.csv",
    "Health_and_Household":        "healthcare_dataset/Health_and_Household.csv",
}


def parse_run_dir(name: str) -> Optional[dict]:
    """Run dirs are named <dataset_stem>_<backend>_seed<n>."""
    parts = name.rsplit("_", 2)
    if len(parts) != 3 or not parts[2].startswith("seed"):
        return None
    return {
        "dataset_stem": parts[0],
        "backend": parts[1],
        "seed": int(parts[2][4:]),
    }


def build_method(train_rows, test_rows):
    bm = BaseMethod(
        fmt="UIRT",
        rating_threshold=1.0,
        exclude_unknowns=True,
        verbose=False,
        seed=42,
    )
    bm._build_datasets(train_data=train_rows, test_data=test_rows, val_data=None)
    bm._build_modalities()
    return bm


def prequential_slices(rows: List[tuple], n_slices: int = 6):
    rows_sorted = sorted(rows, key=lambda r: r[3])
    boundaries = np.linspace(0, len(rows_sorted), n_slices + 1, dtype=int)
    return [rows_sorted[boundaries[i]:boundaries[i+1]] for i in range(n_slices)]


def recency_raw_for_cell(model_pkl: Path, train_rows, test_rows, item_first):
    with open(model_pkl, "rb") as f:
        model = pickle.load(f)
    bm = build_method(train_rows, test_rows)
    stats_dict = recommendation_recency_distribution(
        model, bm, item_first, k=TOP_K, return_raw=True,
    )
    return stats_dict.get("recency_days_raw", np.empty(0))


def ks_for_dataset(dataset_stem: str, run_dir: Path, n_slices: int = 6):
    """Returns list of dicts, one per slice with KS results, or skip reason."""
    csv_rel = DATASET_CSV_FOR.get(dataset_stem)
    if csv_rel is None:
        return None, f"unknown dataset {dataset_stem}"
    csv_path = DATA_DIR / csv_rel
    if not csv_path.exists():
        return None, f"dataset CSV not found: {csv_path}"

    rows = load_uirt(str(csv_path))
    item_first = item_first_seen(rows)
    slices = prequential_slices(rows, n_slices=n_slices)

    out = []
    for i in range(1, n_slices):
        v_pkl = run_dir / VANILLA_SLUG / f"slice{i}.pkl"
        f_pkl = run_dir / FAITHFUL_SLUG / f"slice{i}.pkl"
        if not v_pkl.exists() or not f_pkl.exists():
            out.append({"slice": i, "status": "missing_pkl",
                        "missing": [str(p) for p in (v_pkl, f_pkl) if not p.exists()]})
            continue

        train_rows = [r for s in slices[:i] for r in s]
        test_rows = slices[i]
        try:
            v_arr = recency_raw_for_cell(v_pkl, train_rows, test_rows, item_first)
            f_arr = recency_raw_for_cell(f_pkl, train_rows, test_rows, item_first)
        except Exception as e:
            out.append({"slice": i, "status": "error", "error": str(e)})
            continue

        if v_arr.size == 0 or f_arr.size == 0:
            out.append({"slice": i, "status": "empty"})
            continue

        ks_stat, ks_p = stats.ks_2samp(v_arr, f_arr, alternative="two-sided")
        out.append({
            "slice": i,
            "status": "ok",
            "n_vanilla": int(v_arr.size),
            "n_faithful": int(f_arr.size),
            "median_vanilla":  float(np.median(v_arr)),
            "median_faithful": float(np.median(f_arr)),
            "ks_stat": float(ks_stat),
            "ks_p":    float(ks_p),
        })

    return out, None


def fmt_p(p: float) -> str:
    if p < 1e-12:
        return "< 1e-12"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


def build_markdown(per_dataset_results: Dict[str, list]) -> str:
    lines = []
    lines.append("# Two-sample KS test on recommendation recency\n")
    lines.append(
        "Comparison: per (dataset, prequential slice), the raw recency-in-days "
        "distribution of top-20 recommendations from `win10+causal` vs `shuffle+uniform`. "
        "A small p-value means the two recipes recommend qualitatively different items.\n"
    )

    # Per-dataset per-slice tables.
    all_p = []
    for ds in sorted(per_dataset_results):
        rows = per_dataset_results[ds]
        lines.append(f"\n## {ds}\n")
        if not rows:
            lines.append("*No slices found.*\n")
            continue
        lines.append("| Slice | n (van/fai) | median (van) | median (fai) | KS stat | KS p |")
        lines.append("|---|---|---|---|---|---|")
        for r in rows:
            if r["status"] != "ok":
                detail = r.get("error") or r.get("missing") or r["status"]
                lines.append(f"| {r['slice']} | – | – | – | – | *{r['status']}*: `{detail}` |")
                continue
            all_p.append(r["ks_p"])
            lines.append(
                f"| {r['slice']} | {r['n_vanilla']}/{r['n_faithful']} | "
                f"{r['median_vanilla']:.1f} | {r['median_faithful']:.1f} | "
                f"{r['ks_stat']:.3f} | {fmt_p(r['ks_p'])} |"
            )

    # Family-wise summary.
    lines.append("\n## Family-wise summary\n")
    if not all_p:
        lines.append("*No successful KS tests.*\n")
    else:
        all_p_arr = np.asarray(all_p)
        n = all_p_arr.size
        n_sig = int((all_p_arr < 0.05).sum())
        n_sig_bonf = int((all_p_arr < 0.05 / n).sum())
        lines.append(f"- {n} KS tests run.")
        lines.append(f"- {n_sig}/{n} significant at uncorrected $\\alpha = 0.05$.")
        lines.append(f"- {n_sig_bonf}/{n} significant at Bonferroni-corrected $\\alpha = 0.05 / {n} = {0.05 / n:.4f}$.")
        lines.append(f"- Smallest p observed: {fmt_p(float(all_p_arr.min()))}; largest: {fmt_p(float(all_p_arr.max()))}.")

    lines.append("\n## Reading guide\n")
    lines.append(
        "- KS measures how different the *distributions* are, not just the means. "
        "Two recipes can have similar median recency but very different shapes (e.g.\\ "
        "different tail behavior); KS detects that."
    )
    lines.append(
        "- Large dataset means the test is very sensitive: tiny shifts in the empirical "
        "CDF translate to tiny p-values. Treat the **magnitude of the KS statistic** "
        "(between 0 and 1) as the effect-size proxy, with $p$ as confirmation that the "
        "difference is not from sampling noise."
    )
    lines.append(
        "- KS is one-shot per cell; we don't pool across slices because the recency "
        "scale changes meaningfully along the timeline."
    )
    return "\n".join(lines) + "\n"


def main(only_datasets: Optional[set] = None):
    if not RUNS_DIR.exists():
        print(f"No runs directory at {RUNS_DIR}", file=sys.stderr)
        sys.exit(1)

    per_dataset_results: Dict[str, list] = {}
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        meta = parse_run_dir(run_dir.name)
        if meta is None:
            print(f"  skipping unrecognized dir: {run_dir.name}")
            continue
        ds = meta["dataset_stem"]
        if only_datasets is not None and ds not in only_datasets:
            print(f"  skipping {run_dir.name} (not in --datasets filter)")
            continue
        print(f"Processing {run_dir.name} (dataset={ds}, backend={meta['backend']}, seed={meta['seed']})", flush=True)
        result, err = ks_for_dataset(ds, run_dir)
        if err:
            print(f"  error: {err}", flush=True)
            continue
        # Key results by dataset (multi-seed support: append seed suffix if collision).
        key = ds if ds not in per_dataset_results else f"{ds} (seed {meta['seed']})"
        per_dataset_results[key] = result

    md = build_markdown(per_dataset_results)
    OUTPUT_MD.write_text(md, encoding="utf-8")
    print("\n" + md)
    print(f"\nWrote {OUTPUT_MD}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=(
            "Comma-separated dataset stems to include (e.g., "
            "'Cell_Phones_and_Accessories,Health_and_Household'). "
            "If omitted, runs on every dataset stem found under research/results/runs/."
        ),
    )
    args = ap.parse_args()
    filt = None
    if args.datasets:
        filt = {s.strip() for s in args.datasets.split(",") if s.strip()}
        print(f"Filter: only processing datasets {sorted(filt)}", flush=True)
    main(only_datasets=filt)
