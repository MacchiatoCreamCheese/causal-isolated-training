import argparse
import json
import os
from pathlib import Path
from typing import Dict

from .data import DATASETS  # noqa: F401
from ..paths import PROJECT_ROOT, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "ablation"

RECIPES = ("uniform", "causal")


def set_recipe(eval_method, recipe: str):
    train_set = eval_method.train_set
    train_set.neg_sampling = recipe
    train_set.reset()
    train_set.reset_counterfactual_counters()
    return train_set


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
    env["tuned_arm"] = TUNED_ARM
    env["neumf_pretrain"] = NEUMF_PRETRAIN
    _ENV_CACHE = env
    return env


def _jsonable(config):
    return json.loads(json.dumps(config, default=str, sort_keys=True))


def _config_diff(stored, current):
    stored, current = _jsonable(stored), _jsonable(current)
    return [(k, stored.get(k), current.get(k))
            for k in sorted(set(stored) | set(current))
            if stored.get(k) != current.get(k)]


def load_partial(model: str, dataset: str, seed: int,
                 configs: Dict[str, dict] = None) -> Dict[str, Dict[str, float]]:
    path = seed_json_path(model, dataset, seed)
    if not path.exists():
        return {}
    try:
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
    path = seed_json_path(model, dataset, seed)
    tmp = path.with_suffix(".json.tmp")
    payload = {"model": model, "dataset": dataset, "seed": seed,
               "recipes": recipes}
    if configs is not None:
        payload["configs"] = {r: _jsonable(c) for r, c in configs.items()
                              if r in recipes}
        payload["env"] = _environment()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


PROBE_METRICS = ("counterfactual_rate", "collision_rate")


def extract_metrics(experiment, probe=None) -> Dict[str, float]:
    if experiment.result is None or len(experiment.result) == 0:
        return {}
    metrics = dict(experiment.result[0].metric_avg_results)
    if probe is not None:
        metrics.update((name, float(getattr(probe, name))) for name in PROBE_METRICS)
    return metrics
