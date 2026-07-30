"""Phase 4b: prequential evaluation.

Split the global timeline into N+1 equal-count slices. For i in 1..N,
train on cumulative data through slice i-1, evaluate on slice i.
Report HR@20 / NDCG@20 / Recall@20 per slice plus the macro average.

This is the LeakageStudy timeline scheme applied as an evaluation
methodology: the model is tested at multiple snapshots along the
timeline rather than only at the very end. A "good" causal-training
recipe should hold up across slices, not just at the latest one.

We compare two training recipes per slice:
  - shuffle+uniform (baseline, leaky training under each slice)
  - win10+causal    (Phase 3's best causal-training recipe)

Each cell trains a fresh model on the slice's cumulative data.
"""

import csv
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import numpy as np

from cornac.data import Dataset
from cornac.eval_methods.base_method import BaseMethod, rating_eval, ranking_eval
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.bpr_cpu import BPRMiniBatch
from ..lib.temporal_batching import BPRWindowed
from ..lib.causal_sampling import (
    DATASET_CSV, TOP_K, load_uirt, item_first_seen,
)
from ..lib.timeaware_data import TimeAwareDataset
from ..lib.faithfulness_metrics import recommendation_recency_distribution
from ..paths import RESULTS_DIR, DATA_DIR

USE_GPU = os.environ.get("USE_GPU", "0") == "1"
if USE_GPU:
    from ..lib.bpr_gpu import BPRMiniBatchGPU as _BPR
    from ..lib.bpr_gpu import BPRWindowedGPU as _BPRWin
else:
    _BPR = BPRMiniBatch
    _BPRWin = BPRWindowed


N_SLICES = 6  # 1 warmup + 5 prequential tests
# Prequential carves train_rows by slice but has no val_set, so early stopping
# (BPR_EARLY_STOP requires val_set) won't trigger. We keep the fixed 20-epoch
# budget the prequential analysis was tuned with.
TOTAL_PASSES = 20


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
    # Upgrade the training split so causal negative sampling reads its per-item
    # first-seen index from the same TimeAwareDataset the 2x2 runners use.
    if bm.train_set is not None:
        bm.train_set = TimeAwareDataset.from_dataset(bm.train_set)
    return bm


def evaluate(model, bm):
    metrics = [HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)]
    test_results, _ = ranking_eval(
        model=model,
        metrics=metrics,
        train_set=bm.train_set,
        test_set=bm.test_set,
        val_set=None,
        exclude_unknowns=True,
    )
    return {m.name: val for m, val in zip(metrics, test_results)}


def train_variant(name, train_set, seed):
    bs = 16384 if USE_GPU else 4096
    # No early stopping here: prequential has no val_set.
    if name == "shuffle+uniform":
        model = _BPR(
            name=name, k=64, n_epochs=TOTAL_PASSES, batch_size=bs,
            learning_rate=0.05,
            lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
            sampler="uniform", seed=seed, verbose=False,
        )
    elif name == "win10+causal":
        model = _BPRWin(
            name=name, k=64, n_windows=10, epochs_per_window=TOTAL_PASSES,
            batch_size=bs, learning_rate=0.05,
            lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
            sampler="causal", seed=seed, verbose=False,
        )
    else:
        raise ValueError(name)
    model.fit(train_set)
    return model


def _git_rev():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _checkpoint_dir(results_dir: str, dataset_csv: str, seed: int) -> Path:
    """Per-run directory holding partial cell results + pickled models.

    Identical (dataset, backend, seed) re-uses the same directory, so
    resuming after a crash picks up where it left off without redoing
    finished cells. Git rev is recorded inside config.json for
    forensics — it intentionally does NOT appear in the path, because
    that would invalidate checkpoints across normal commits even when
    the experiment code is unchanged.
    """
    ds_tag = Path(dataset_csv).stem.replace("/", "_")
    backend_tag = "gpu" if USE_GPU else "cpu"
    return Path(results_dir) / "runs" / f"{ds_tag}_{backend_tag}_seed{seed}"


def _cell_paths(run_dir: Path, variant: str, slice_idx: int):
    safe_variant = variant.replace("+", "_").replace("/", "_")
    base = run_dir / safe_variant / f"slice{slice_idx}"
    return {
        "json": base.with_suffix(".json"),
        "pkl":  base.with_suffix(".pkl"),
    }


def _save_cell(paths: dict, cell: dict, model) -> None:
    """Atomic-ish write: write to .tmp then rename. Guarantees that a
    cell file is either fully present or absent — never half-written.
    """
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    tmp_json = paths["json"].with_suffix(".json.tmp")
    tmp_pkl  = paths["pkl"].with_suffix(".pkl.tmp")
    with open(tmp_json, "w") as f:
        json.dump(cell, f, indent=2)
    try:
        with open(tmp_pkl, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        # Model pickling can fail on GPU tensors / Cython objects. Don't kill
        # the run — the JSON is what the aggregator needs.
        print(f"  warning: failed to pickle model ({e}); keeping JSON only", flush=True)
        if tmp_pkl.exists():
            tmp_pkl.unlink()
    os.replace(tmp_json, paths["json"])
    if tmp_pkl.exists():
        os.replace(tmp_pkl, paths["pkl"])


def _load_cell(paths: dict):
    if not paths["json"].exists():
        return None
    with open(paths["json"]) as f:
        return json.load(f)


def main(dataset_csv: str = DATASET_CSV, seed: int = 42, results_dir: str = None):
    if results_dir is None:
        results_dir = str(RESULTS_DIR)
    # Resolve a relative dataset path under DATA_DIR so it works after relocation.
    if not Path(dataset_csv).is_absolute() and not Path(dataset_csv).exists():
        dataset_csv = str(DATA_DIR / dataset_csv)
    rows = load_uirt(dataset_csv)
    print(f"Loaded {len(rows):,} from {dataset_csv}", flush=True)
    item_first = item_first_seen(rows)

    rows_sorted = sorted(rows, key=lambda r: r[3])
    boundaries = np.linspace(0, len(rows_sorted), N_SLICES + 1, dtype=int)
    slices = [rows_sorted[boundaries[i]:boundaries[i+1]] for i in range(N_SLICES)]
    print(f"Slice sizes: {[len(s) for s in slices]}", flush=True)
    print(f"Slice timestamp ranges:", flush=True)
    for i, s in enumerate(slices):
        if s:
            print(f"  slice {i}: ts [{s[0][3]}, {s[-1][3]}]  rows={len(s)}", flush=True)

    run_dir = _checkpoint_dir(results_dir, dataset_csv, seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    if not config_path.exists():
        config = {
            "dataset_csv": dataset_csv,
            "seed": seed,
            "backend": "gpu" if USE_GPU else "cpu",
            "n_slices": N_SLICES,
            "total_passes": TOTAL_PASSES,
            "top_k": TOP_K,
            "code_revision": _git_rev(),
            "first_started_utc": datetime.now(timezone.utc).isoformat(),
            "slice_sizes": [len(s) for s in slices],
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    print(f"Checkpoint dir: {run_dir}", flush=True)

    cells = {name: [] for name in ["shuffle+uniform", "win10+causal"]}

    for i in range(1, N_SLICES):
        train_rows = None
        test_rows = None
        print(f"\n--- Prequential step {i}/{N_SLICES-1} ---", flush=True)

        for name in cells:
            paths = _cell_paths(run_dir, name, i)
            cached = _load_cell(paths)
            if cached is not None:
                cells[name].append(cached)
                m = cached["metrics"]
                print(f"[step {i}] {name}: RESUMED from {paths['json'].name}  "
                      f"HR={m['HitRatio@20']:.4f} cf={cached['counterfactual_rate']*100:.2f}%",
                      flush=True)
                continue

            if train_rows is None:
                train_rows = [r for slice_data in slices[:i] for r in slice_data]
                test_rows = slices[i]
                print(f"  train n={len(train_rows)}, test n={len(test_rows)}", flush=True)

            bm = build_method(train_rows, test_rows)
            model = train_variant(name, bm.train_set, seed)
            scores = evaluate(model, bm)
            cf = float(getattr(model, "counterfactual_rate", 0.0))
            rec_stats = recommendation_recency_distribution(model, bm, item_first, k=TOP_K)
            cell = {
                "slice": i,
                "train_n": len(train_rows),
                "test_n": len(test_rows),
                "metrics": {k: float(v) for k, v in scores.items()},
                "counterfactual_rate": cf,
                "recency": rec_stats,
            }
            _save_cell(paths, cell, model)
            cells[name].append(cell)
            print(f"[step {i}] {name}: {scores}", flush=True)
            print(f"[step {i}] {name}: counterfactual_rate={cf*100:.2f}%", flush=True)
            print(f"[step {i}] {name}: recency_days mean={rec_stats.get('recency_days_mean', float('nan')):.1f} "
                  f"median={rec_stats.get('recency_days_median', float('nan')):.1f} "
                  f"p10-p90=[{rec_stats.get('recency_days_p10', float('nan')):.0f},"
                  f"{rec_stats.get('recency_days_p90', float('nan')):.0f}]", flush=True)
            print(f"[step {i}] {name}: saved {paths['json'].name}"
                  + (f" + {paths['pkl'].name}" if paths['pkl'].exists() else ""),
                  flush=True)

    print("\n===== PREQUENTIAL SUMMARY =====", flush=True)
    for name, slice_results in cells.items():
        if not slice_results:
            continue
        hrs = [r["metrics"]["HitRatio@20"] for r in slice_results]
        ndcgs = [r["metrics"]["NDCG@20"] for r in slice_results]
        recalls = [r["metrics"]["Recall@20"] for r in slice_results]
        cfs = [r["counterfactual_rate"] for r in slice_results]
        print(f"\n{name}", flush=True)
        for i, r in enumerate(slice_results, start=1):
            m = r["metrics"]
            print(f"  slice {i}: HR={m['HitRatio@20']:.4f}  NDCG={m['NDCG@20']:.4f}  Recall={m['Recall@20']:.4f}  cf={r['counterfactual_rate']*100:.2f}%", flush=True)
        print(f"  MACRO AVG: HR={np.mean(hrs):.4f}  NDCG={np.mean(ndcgs):.4f}  Recall={np.mean(recalls):.4f}", flush=True)
        print(f"  CF MEAN:   {np.mean(cfs)*100:.2f}%", flush=True)
        med_recs = [r["recency"].get("recency_days_median", float('nan')) for r in slice_results]
        print(f"  RECENCY MEDIAN (days, per slice): {[f'{m:.0f}' for m in med_recs]}", flush=True)

    # Persist for later cross-seed analysis.
    payload = {
        "dataset_csv": dataset_csv,
        "seed": seed,
        "backend": "gpu" if USE_GPU else "cpu",
        "n_slices": N_SLICES,
        "total_passes": TOTAL_PASSES,
        "top_k": TOP_K,
        "code_revision": _git_rev(),
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "slice_sizes": [len(s) for s in slices],
        "cells": cells,
    }
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_tag = Path(dataset_csv).stem.replace("/", "_")
    backend_tag = "gpu" if USE_GPU else "cpu"
    out_path = out_dir / f"preq_{ds_tag}_{backend_tag}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote results to {out_path}", flush=True)


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else DATASET_CSV
    sd = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    main(ds, sd)
