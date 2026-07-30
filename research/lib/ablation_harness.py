"""Shared multi-seed harness for phase5/6/7 2x2 ablations.

Each phase script wraps its 4 recipe cells in a per-(dataset, seed) loop,
captures HR/NDCG/Recall via cornac.Experiment.result, and writes a JSON
per (model, dataset, seed) to research/results/2x2/.

The JSON layout is flat so the aggregator can scan a directory:

    {
      "model": "BPR",
      "dataset": "baby",
      "seed": 42,
      "recipes": {
        "shuffle+uniform": {"HR@20": ..., "NDCG@20": ..., "Recall@20": ...,
                             "counterfactual_rate": ...},
        ...
      }
    }
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

# Dataset registry lives in data.py now (the generic loader); re-exported here so
# existing `from ..lib.ablation_harness import DATASETS` call sites keep working.
from .data import DATASETS  # noqa: F401
from ..paths import RESULTS_DIR

OUT_DIR = RESULTS_DIR / "2x2"


def parse_args(default_seeds=(42,), default_datasets=("baby",)):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=",".join(str(s) for s in default_seeds),
                   help="Comma-separated seeds, e.g. 42,123,2026")
    p.add_argument("--datasets", default=",".join(default_datasets),
                   help="Comma-separated dataset keys: baby,cellphone,healthcare")
    args = p.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {list(DATASETS)})")
    return seeds, datasets


def seed_json_path(model: str, dataset: str, seed: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{model.lower()}_{dataset}_seed{seed}.json"


def already_done(model: str, dataset: str, seed: int) -> bool:
    return seed_json_path(model, dataset, seed).exists()


def load_partial(model: str, dataset: str, seed: int) -> Dict[str, Dict[str, float]]:
    """Return the recipes dict from a partial/full JSON, or empty if none."""
    path = seed_json_path(model, dataset, seed)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("recipes", {})
    except Exception:
        return {}


def write_partial(model: str, dataset: str, seed: int,
                  recipes: Dict[str, Dict[str, float]]) -> None:
    """Atomic write so a kill mid-write does not corrupt the JSON."""
    path = seed_json_path(model, dataset, seed)
    tmp = path.with_suffix(".json.tmp")
    payload = {"model": model, "dataset": dataset, "seed": seed, "recipes": recipes}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def extract_metrics(experiment) -> Dict[str, float]:
    """Pull metric averages from a freshly-run cornac.Experiment."""
    if experiment.result is None or len(experiment.result) == 0:
        return {}
    r = experiment.result[0]
    return dict(r.metric_avg_results)


def write_seed_json(model: str, dataset: str, seed: int,
                    recipes: Dict[str, Dict[str, float]]) -> None:
    path = seed_json_path(model, dataset, seed)
    tmp = path.with_suffix(".json.tmp")
    payload = {"model": model, "dataset": dataset, "seed": seed, "recipes": recipes}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    print(f"[multiseed] wrote {path}", flush=True)
