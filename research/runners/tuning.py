"""Coordinate-descent hyperparameter tuning for BPR / NeuMF / LightGCN.

Scope (per plan):
  - Dataset:  baby
  - Recipe:   uniform (vanilla baseline cell)
  - Seed:     42
  - Metric:   NDCG@20 (TOP_K from research.lib.causal_sampling)
  - Method:   coordinate descent over knobs declared in
              research/lib/tuning_config.py:TUNE_ORDER, knob order = most-impactful first.

For each model:
  1. Run baseline at defaults (research/lib/tuning_config.py:BASELINE_CONFIG).
  2. For each (knob, grid):
       - Train every grid value (grid excludes the default).
       - Compare best grid result vs the running winner config.
       - Fix knob at the winner value; carry forward.
  3. Write winner.json.

Per-trial JSON layout:  research/results/tuning/<model>/<knob>/<value>.json
Baseline:              research/results/tuning/<model>/baseline.json
Final winner:          research/results/tuning/<model>/winner.json

Re-running is idempotent: existing trial JSONs are skipped.

Usage:
    python -m research.runners.tuning --model lightgcn
    python -m research.runners.tuning --model neumf
    python -m research.runners.tuning --model bpr
    python -m research.runners.tuning --model lightgcn --restart-from lambda_reg
"""

import argparse
import json
import os
import time
from pathlib import Path

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.tuning_config import TUNE_ORDER, BASELINE_CONFIG, BPR_EARLY_STOP
from ..paths import logs_dir, RESULTS_DIR


TUNE_DIR = RESULTS_DIR / "tuning"
TUNE_DATASET = "baby"
TUNE_SEED = 42
TUNE_RECIPE = "uniform"
SELECT_METRIC = f"NDCG@{TOP_K}"


# ---------------------------------------------------------------------------
# Model construction — one builder per model, taking the running config dict.
# ---------------------------------------------------------------------------

def _build_model(model_name: str, config: dict, seed: int):
    if model_name == "lightgcn":
        from cornac.models import LightGCN
        from ..lib.tuning_config import (
            LIGHTGCN_BATCH, LIGHTGCN_EARLY_STOP, LIGHTGCN_EPOCHS,
        )
        return LightGCN(
            name=f"LightGCN/tune/{TUNE_DATASET}/s{seed}",
            emb_size=64,
            num_layers=int(config["num_layers"]),
            learning_rate=float(config["learning_rate"]),
            lambda_reg=float(config["lambda_reg"]),
            batch_size=LIGHTGCN_BATCH[TUNE_DATASET],
            num_epochs=LIGHTGCN_EPOCHS,
            early_stopping=LIGHTGCN_EARLY_STOP,
            seed=seed,
            verbose=False,
        )

    if model_name == "neumf":
        # Persistence shim: see lib/cornac_compat.py:NeuMF.
        from ..lib.cornac_compat import NeuMF
        from ..lib.tuning_config import NEUMF_KWARGS
        num_factors = int(config["num_factors"])
        hidden = int(config["mlp_hidden_count"])
        # Tower-halving with layers[-1] == num_factors (He 2017 §3.3).
        # hidden=3 ⇒ (num_factors*8, *4, *2, *1) = (64,32,16,8) when num_factors=8.
        layers = tuple(num_factors * (2 ** i) for i in range(hidden, -1, -1))
        kwargs = dict(NEUMF_KWARGS)
        kwargs.update(
            num_factors=num_factors,
            layers=layers,
            batch_size=int(config["batch_size"]),
            lr=float(config["learning_rate"]),
            num_neg=int(config["num_neg"]),
        )
        return NeuMF(
            name=f"NeuMF/tune/{TUNE_DATASET}/s{seed}",
            seed=seed,
            verbose=False,
            **kwargs,
        )

    if model_name == "bpr":
        from ..lib.bpr_cpu import BPRMiniBatch
        return BPRMiniBatch(
            name=f"BPR/tune/{TUNE_DATASET}/s{seed}",
            k=int(config["k_embed_dim"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            lambda_u=float(config["lambda_u"]),
            lambda_i=float(config["lambda_i"]),
            lambda_j=float(config["lambda_j"]),
            n_epochs=int(config["n_epochs"]),
            sampler=TUNE_RECIPE,
            **BPR_EARLY_STOP,
            seed=seed,
            verbose=False,
        )

    raise ValueError(f"unknown model: {model_name}")


# ---------------------------------------------------------------------------
# I/O helpers.
# ---------------------------------------------------------------------------

def _value_slug(value) -> str:
    s = repr(value) if isinstance(value, bool) else str(value)
    return (s.replace(".", "p").replace("-", "m").replace("+", "")
             .replace("(", "").replace(")", "").replace(",", "_").replace(" ", ""))


def _trial_path(model_name: str, knob: str, value) -> Path:
    d = TUNE_DIR / model_name / knob
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_value_slug(value)}.json"


def _baseline_path(model_name: str) -> Path:
    d = TUNE_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d / "baseline.json"


def _winner_path(model_name: str) -> Path:
    return TUNE_DIR / model_name / "winner.json"


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)


def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Trial runner.
# ---------------------------------------------------------------------------

def _train_once(model_name: str, config: dict, eval_method) -> tuple:
    """Build the model, run cornac.Experiment, return (metrics_dict, wall_seconds)."""
    model = _build_model(model_name, config, TUNE_SEED)
    exp = cornac.Experiment(
        eval_method=eval_method,
        models=[model],
        metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
        user_based=True,
        save_dir=logs_dir(),
    )
    t0 = time.time()
    exp.run()
    wall = time.time() - t0
    if exp.result is None or len(exp.result) == 0:
        raise RuntimeError("cornac.Experiment returned no result")
    metrics = dict(exp.result[0].metric_avg_results)
    return metrics, wall


def _build_eval_method():
    # Tuning happens on the vanilla arm so the chosen hyperparameters are not
    # picked under the mechanism being evaluated.
    return build_eval_method(TUNE_DATASET, neg_sampling=TUNE_RECIPE,
                             seed=TUNE_SEED)


# ---------------------------------------------------------------------------
# Coordinate-descent loop.
# ---------------------------------------------------------------------------

def run_tuning(model_name: str, restart_from: str = None) -> None:
    print(f"\n=== Tuning {model_name} on {TUNE_DATASET} × {TUNE_RECIPE} "
          f"× seed={TUNE_SEED}, selecting on {SELECT_METRIC} ===", flush=True)

    eval_method = _build_eval_method()
    config = dict(BASELINE_CONFIG[model_name])

    # --- Baseline ---
    bpath = _baseline_path(model_name)
    if bpath.exists():
        baseline = _load_json(bpath)
        print(f"[baseline] cached: {SELECT_METRIC}="
              f"{baseline['metrics'][SELECT_METRIC]:.4f}", flush=True)
    else:
        print(f"[baseline] running with defaults: {config}", flush=True)
        metrics, wall = _train_once(model_name, config, eval_method)
        baseline = {"config": dict(config), "metrics": metrics, "wall_seconds": wall}
        _write_json(bpath, baseline)
        print(f"[baseline] {SELECT_METRIC}={metrics[SELECT_METRIC]:.4f}  "
              f"({wall:.1f}s)", flush=True)

    # Running winner (config dict + its metric). The "default value" for the
    # current knob lives at config[knob]; its metric is prev_winner_metric.
    prev_winner_config = dict(baseline["config"])
    prev_winner_metric = float(baseline["metrics"][SELECT_METRIC])
    config = dict(prev_winner_config)

    # If --restart-from was given, fast-forward through earlier knobs by
    # reading their cached winners off disk; raise if any are missing.
    if restart_from is not None:
        for knob, _ in TUNE_ORDER[model_name]:
            if knob == restart_from:
                break
            # Reuse cached trials to pick the winner without retraining.
            winner_v, winner_m = _resolve_cached_winner(
                model_name, knob, prev_winner_config[knob], prev_winner_metric)
            config[knob] = winner_v
            prev_winner_config = dict(config)
            prev_winner_metric = winner_m

    # --- Per-knob grid sweep ---
    started = restart_from is None
    for knob, grid in TUNE_ORDER[model_name]:
        if not started:
            started = (knob == restart_from)
            if not started:
                continue
        print(f"\n--- knob: {knob}  grid={grid}  "
              f"(default carries metric={prev_winner_metric:.4f}) ---", flush=True)

        results = []  # list of (value, metric)
        for value in grid:
            tpath = _trial_path(model_name, knob, value)
            if tpath.exists():
                trial = _load_json(tpath)
                m = float(trial["metrics"][SELECT_METRIC])
                print(f"[skip] {knob}={value}  {SELECT_METRIC}={m:.4f}  [cached]",
                      flush=True)
            else:
                trial_config = dict(config)
                trial_config[knob] = value
                metrics, wall = _train_once(model_name, trial_config, eval_method)
                trial = {
                    "knob": knob,
                    "value": value,
                    "config": trial_config,
                    "metrics": metrics,
                    "wall_seconds": wall,
                }
                _write_json(tpath, trial)
                m = float(metrics[SELECT_METRIC])
                print(f"[run]  {knob}={value}  {SELECT_METRIC}={m:.4f}  "
                      f"({wall:.1f}s)", flush=True)
            results.append((value, m))

        # Winner = best of (cached default carry-forward) vs grid trials.
        default_value = prev_winner_config[knob]
        candidates = [(default_value, prev_winner_metric)] + results
        best_value, best_metric = max(candidates, key=lambda x: x[1])
        config[knob] = best_value
        prev_winner_config = dict(config)
        prev_winner_metric = best_metric
        marker = " (kept default)" if best_value == default_value else ""
        print(f"[winner@{knob}] {knob}={best_value}  "
              f"{SELECT_METRIC}={best_metric:.4f}{marker}", flush=True)

    # --- Persist final winner ---
    winner_payload = {
        "model": model_name,
        "dataset": TUNE_DATASET,
        "seed": TUNE_SEED,
        "recipe": TUNE_RECIPE,
        "select_metric": SELECT_METRIC,
        "config": prev_winner_config,
        "metric": prev_winner_metric,
    }
    _write_json(_winner_path(model_name), winner_payload)
    print(f"\n=== {model_name} winner: {prev_winner_config}  "
          f"{SELECT_METRIC}={prev_winner_metric:.4f} ===\n", flush=True)


def _resolve_cached_winner(model_name: str, knob: str, default_value,
                           default_metric: float):
    """For --restart-from: replay a knob's winner from cached trial JSONs."""
    grid = dict(TUNE_ORDER[model_name])[knob]
    candidates = [(default_value, default_metric)]
    for value in grid:
        tpath = _trial_path(model_name, knob, value)
        if not tpath.exists():
            raise SystemExit(
                f"cannot --restart-from past {knob!r}: "
                f"missing cached trial {tpath}")
        trial = _load_json(tpath)
        candidates.append((value, float(trial["metrics"][SELECT_METRIC])))
    return max(candidates, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=sorted(TUNE_ORDER.keys()),
                   help="Which model to tune.")
    p.add_argument("--restart-from", default=None,
                   help="Knob name to resume from. Earlier knobs' winners are "
                        "resolved from cached trial JSONs without retraining.")
    args = p.parse_args()
    run_tuning(args.model, restart_from=args.restart_from)


if __name__ == "__main__":
    main()
