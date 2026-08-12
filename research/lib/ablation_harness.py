"""Shared multi-seed harness for the causal-sampling ablation.

Each runner wraps its recipe cells in a per-(dataset, seed) loop, captures
HR/NDCG/Recall via cornac.Experiment.result, and writes a JSON per
(model, dataset, seed) to research/results/ablation/.

The ablation is a single-variable comparison: `uniform` (vanilla, negatives
from the whole catalog) vs `causal` (negatives restricted to items that
existed at the positive's timestamp). Nothing else differs between cells.

The JSON layout is flat so the aggregator can scan a directory:

    {
      "model": "BPR",
      "dataset": "baby",
      "seed": 42,
      "recipes": {
        "uniform": {"HR@20": ..., "NDCG@20": ..., "Recall@20": ...,
                    "counterfactual_rate": ..., "collision_rate": ...},
        "causal":  {...}
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

OUT_DIR = RESULTS_DIR / "ablation"

# The two arms of the ablation, in report order.
RECIPES = ("uniform", "causal")


def set_recipe(eval_method, recipe: str):
    """Point an eval method's training split at a sampling arm.

    Switching in place rather than rebuilding the split matters: re-reading
    and re-splitting the CSV per cell costs minutes on the larger datasets,
    and it would also give each cell a differently-seeded `Dataset.rng`.

    Returns the training split so callers can read `counterfactual_rate` off
    it after the fit. Resetting the probe here is what keeps counts from
    accumulating across cells.

    `reset()` restores the split's RNG to its seed, so each cell starts from the
    same random stream and the arms form a paired comparison. Without it the
    second cell would inherit whatever state the first left behind, and results
    would depend on the order the recipes ran in. It only works if the eval
    method was built with a seed — see `data.build_eval_method`.
    """
    train_set = eval_method.train_set
    train_set.neg_sampling = recipe
    train_set.reset()
    train_set.reset_counterfactual_counters()
    return train_set


# The primary three, in ascending size. `healthcare` (7.18M) is deliberately
# excluded: it is run last, only if there is time, via an explicit --datasets.
DEFAULT_DATASETS = ("musical", "baby", "cellphone")


def parse_args(default_seeds=(42,), default_datasets=DEFAULT_DATASETS):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=",".join(str(s) for s in default_seeds),
                   help="Comma-separated seeds, e.g. 42,123,2026")
    p.add_argument("--datasets", default=",".join(default_datasets),
                   help="Comma-separated dataset keys: musical,baby,cellphone,healthcare "
                        "(default: musical,baby,cellphone — healthcare is opt-in)")
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
        # utf-8-sig, not utf-8: a BOM (which any Windows editor may add) would
        # otherwise raise here, and this except-clause would silently discard a
        # finished cell — turning a resume into a silent re-run.
        with open(path, encoding="utf-8-sig") as f:
            payload = json.load(f)
        return payload.get("recipes", {})
    except Exception as e:
        print(f"[warn] could not read checkpoint {path}: {e}", flush=True)
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
