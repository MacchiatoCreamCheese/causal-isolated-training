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
from typing import Dict

# Dataset registry lives in data.py now (the generic loader); re-exported here so
# existing `from ..lib.ablation_harness import DATASETS` call sites keep working.
from .data import DATASETS  # noqa: F401
from ..paths import PROJECT_ROOT, RESULTS_DIR

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


_ENV_CACHE = None


def _environment() -> Dict[str, str]:
    """Library versions, device, and commit that produced a cell.

    **Recorded, never compared.** A torch patch bump must not invalidate a sweep,
    so this plays no part in the resume decision -- but it is what answers "was
    this produced under cornac 2.3.5 or 2.6.0, CPU or GPU?" after the fact, which
    once cost a session of git archaeology because nothing wrote it down.

    Cached: it shells out to git, and `write_partial` runs once per finished
    recipe.
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE

    from .tuning_config import NEUMF_PRETRAIN, TUNED_ARM

    env = {}
    for mod in ("cornac", "numpy", "scipy", "torch"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            env[mod] = "absent"
    try:
        import torch
        env["device"] = (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "cpu")
    except Exception:
        env["device"] = "unknown"
    try:
        import subprocess
        env["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        env["git_commit"] = "unknown"
    # The switches that change results without changing any single kwarg.
    env["tuned_arm"] = TUNED_ARM
    env["neumf_pretrain"] = NEUMF_PRETRAIN
    _ENV_CACHE = env
    return env


def _jsonable(config):
    """Round-trip through JSON so comparisons see what disk sees.

    Without this a live `layers=(64, 32, 16, 8)` never equals the
    `[64, 32, 16, 8]` that comes back from a stored cell, and every resume would
    re-run everything.
    """
    return json.loads(json.dumps(config, default=str, sort_keys=True))


def _config_diff(stored, current):
    """`[(key, old, new)]` for every key that changed. Empty means a match."""
    stored, current = _jsonable(stored), _jsonable(current)
    return [(k, stored.get(k), current.get(k))
            for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)]


def load_partial(model: str, dataset: str, seed: int,
                 configs: Dict[str, dict] = None) -> Dict[str, Dict[str, float]]:
    """Recipes already computed for this cell, or empty if none.

    `configs` maps recipe -> the settings this run *would* use. Supplied, each
    stored recipe is kept only if the config that produced it still matches, so a
    cell tuned under different hyperparameters is re-run instead of silently
    resumed. Omitted, this is the old existence-only behaviour.

    Per-recipe rather than per-cell because under `RESEARCH_TUNED_ARM=per_arm`
    the two arms genuinely differ, and because it lets one arm be re-run without
    discarding the other.
    """
    path = seed_json_path(model, dataset, seed)
    if not path.exists():
        return {}
    try:
        # utf-8-sig, not utf-8: a BOM (which any Windows editor may add) would
        # otherwise raise here, and this except-clause would silently discard a
        # finished cell — turning a resume into a silent re-run.
        with open(path, encoding="utf-8-sig") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"[warn] could not read checkpoint {path}: {e}", flush=True)
        return {}

    recipes = payload.get("recipes", {})
    if configs is None or not recipes:
        return recipes

    stored_cfgs = payload.get("configs") or {}
    kept = {}
    for recipe, metrics in recipes.items():
        current = configs.get(recipe)
        if current is None:
            kept[recipe] = metrics
            continue
        stored = stored_cfgs.get(recipe)
        if stored is None:
            # Written before provenance stamping. Kept rather than discarded:
            # silently overwriting real results because they predate a feature is
            # worse than saying so. Move the file aside to force a re-run.
            print(f"[warn] {path.name} [{recipe}]: no config recorded, cannot "
                  f"verify it matches current settings — keeping it", flush=True)
            kept[recipe] = metrics
            continue
        diff = _config_diff(stored, current)
        if diff:
            print(f"[stale] {path.name} [{recipe}]: config changed, re-running",
                  flush=True)
            for key, old, cur in diff:
                print(f"         {key}: {old!r} -> {cur!r}", flush=True)
            continue
        kept[recipe] = metrics
    return kept


def write_partial(model: str, dataset: str, seed: int,
                  recipes: Dict[str, Dict[str, float]],
                  configs: Dict[str, dict] = None) -> None:
    """Atomic write so a kill mid-write does not corrupt the JSON."""
    path = seed_json_path(model, dataset, seed)
    tmp = path.with_suffix(".json.tmp")
    payload = {"model": model, "dataset": dataset, "seed": seed,
               "recipes": recipes}
    if configs is not None:
        # Only for recipes actually present, so a partial cell never claims to
        # have run something it has not.
        payload["configs"] = {r: _jsonable(c) for r, c in configs.items()
                              if r in recipes}
        payload["env"] = _environment()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


#: Faithfulness probes recorded alongside the accuracy metrics in every cell.
#:
#: The counters are reset once per cell by `set_recipe`, so these cover every
#: negative drawn during that cell -- which is normally one `fit()`. With NeuMF
#: pre-training enabled (RESEARCH_NEUMF_PRETRAIN=1) a cell runs three fits (GMF,
#: MLP, then the fused model) through the same split, so rho and collision_rate
#: span all three and the denominator is ~3x larger. Both are rates, so they stay
#: comparable -- but do not read a raw draw count across runs with and without it.
#: Named once here because the aggregator reads these keys back
#: (`analysis/aggregate_ablation.py`) and because forwarding them by hand in
#: each runner is how `collision_rate` came to be recorded for two models and
#: crash on the third.
PROBE_METRICS = ("counterfactual_rate", "collision_rate")


def extract_metrics(experiment, probe=None) -> Dict[str, float]:
    """Pull metric averages from a freshly-run cornac.Experiment.

    `probe` is whatever object carries the sampling probes for this model: the
    training split for cornac's own models (they have no idea negatives are
    being sampled for them), or the model itself for our BPR, which owns its
    sampler. Both expose the same `PROBE_METRICS` property names, so the runners
    differ only in which object they hand over.
    """
    if experiment.result is None or len(experiment.result) == 0:
        return {}
    metrics = dict(experiment.result[0].metric_avg_results)
    if probe is not None:
        metrics.update((name, float(getattr(probe, name))) for name in PROBE_METRICS)
    return metrics
