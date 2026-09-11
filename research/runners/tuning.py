"""Coordinate-descent hyperparameter tuning for BPR / NeuMF / LightGCN.

Scope is chosen per invocation via --dataset / --recipe / --seed, defaulting to
the historical `baby / uniform / 42` cell:
  - Dataset:  any key in lib/data.py:DATASETS
  - Recipe:   uniform (the vanilla arm) or causal
  - Seed:     42 by default
  - Metric:   **validation** NDCG@20 (TOP_K from research.lib.causal_sampling).
              Test metrics are recorded next to it for reference and never used
              to select.
  - Method:   coordinate descent over knobs declared in
              research/lib/tuning_config.py:TUNE_ORDER, knob order = most-impactful first.

**Validation, not test.** cornac keeps the two splits in different lists --
`exp.val_result` is validation, `exp.result` is *test* -- and reading the wrong
one selects on the test set with no error at all. Until 2026-09-11 this tuner did
exactly that. Files written before the fix have no `val_metrics` and are treated
as stale (re-run, never reused); `runners/tuning_backfill.py` recovers their
validation scores from the stdout logs so cells where the two splits pick the
same values need no re-run.

**A cache hit needs the same config.** A trial file is keyed by knob and value
only, but it was trained around whatever the earlier knobs' winners were. If an
earlier pick changes, every later trial on disk was trained at a different config
and must not be reused; `_usable` checks the stored config, not just the path.

**Which arm to tune on.** Tuning under `uniform` keeps hyperparameters from being
chosen under the mechanism being evaluated, which is what makes the ablation a
single-variable comparison. Sweeping `causal` as well is supported so the two
arms can also be compared each at its own best -- but that is a separate
deployment question, not the ablation, and the two must not be mixed in one
table. See the plan notes in the repo for how the winners are meant to be applied.

For each model:
  1. Run baseline at defaults (research/lib/tuning_config.py:BASELINE_CONFIG).
  2. For each (knob, grid):
       - Train every grid value (grid excludes the default).
       - Compare best grid result vs the running winner config.
       - Fix knob at the winner value; carry forward.
  3. Write winner.json.

Output layout, one tree per (model, dataset, recipe, seed) cell:

    research/results/tuning/<model>/<dataset>/<recipe>/seed<n>/baseline.json
    research/results/tuning/<model>/<dataset>/<recipe>/seed<n>/<knob>/<value>.json
    research/results/tuning/<model>/<dataset>/<recipe>/seed<n>/winner.json

The scope is in the *path*, not just inside the files, and that is load-bearing:
re-running is idempotent because a cached trial JSON is skipped, so a layout that
omitted the dataset would let a second dataset silently adopt the first one's
trials and report them as its own -- printing `[skip] ... [cached]` the whole way
and looking perfectly healthy. Anything written under the older flat
`<model>/<knob>/` layout carries no record of which dataset produced it and must
be deleted rather than migrated.

Usage:
    python -m research.runners.tuning --model lightgcn
    python -m research.runners.tuning --model neumf --dataset musical
    python -m research.runners.tuning --model bpr --dataset cellphone --recipe causal
    python -m research.runners.tuning --model neumf --restart-from learning_rate
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.causal_sampling import TOP_K
from ..lib.data import DATASETS, build_eval_method
from ..lib.timeaware_data import NEG_SAMPLING_MODES
from ..lib.tuning_config import (
    BASELINE_CONFIG, BPR_EARLY_STOP, TUNE_ORDER, tuning_model_dir,
)
from ..paths import logs_dir, RESULTS_DIR


TUNE_DIR = RESULTS_DIR / "tuning"
SELECT_METRIC = f"NDCG@{TOP_K}"

# CLI defaults -- the cell this tuner used to be hardcoded to.
DEFAULT_DATASET = "baby"
DEFAULT_SEED = 42
DEFAULT_RECIPE = "uniform"


@dataclass(frozen=True)
class Scope:
    """Which (dataset, arm, seed) cell a tuning run belongs to.

    Passed around rather than read from module globals so one process can tune
    several cells, and -- more importantly -- so `dir()` below can put all three
    into the output path. Frozen because it keys the on-disk layout.
    """

    dataset: str
    recipe: str
    seed: int

    def dir(self, model_name: str) -> Path:
        # tuning_model_dir keeps a pre-training sweep off the from-scratch
        # sweep's files; it is a no-op when nothing non-default is enabled.
        return (TUNE_DIR / tuning_model_dir(model_name) / self.dataset
                / self.recipe / f"seed{self.seed}")

    def __str__(self) -> str:
        return f"{self.dataset} x {self.recipe} x seed={self.seed}"


# ---------------------------------------------------------------------------
# Model construction — one builder per model, taking the running config dict.
# ---------------------------------------------------------------------------

def _build_model(model_name: str, config: dict, scope: Scope):
    seed = scope.seed
    if model_name == "lightgcn":
        from cornac.models import LightGCN
        from ..lib.tuning_config import (
            LIGHTGCN_BATCH, LIGHTGCN_EARLY_STOP, LIGHTGCN_EPOCHS,
        )
        return LightGCN(
            name=f"LightGCN/tune/{scope.dataset}/{scope.recipe}/s{seed}",
            emb_size=64,
            num_layers=int(config["num_layers"]),
            learning_rate=float(config["learning_rate"]),
            lambda_reg=float(config["lambda_reg"]),
            batch_size=LIGHTGCN_BATCH[scope.dataset],
            num_epochs=LIGHTGCN_EPOCHS,
            early_stopping=LIGHTGCN_EARLY_STOP,
            seed=seed,
            verbose=False,
        )

    if model_name == "neumf":
        # Persistence shim: see lib/cornac_compat.py:NeuMF.
        from ..lib.cornac_compat import NeuMF
        from ..lib.tuning_config import NEUMF_KWARGS, neumf_layers
        num_factors = int(config["num_factors"])
        layers = neumf_layers(num_factors, int(config["mlp_hidden_count"]))
        kwargs = dict(NEUMF_KWARGS)
        kwargs.update(
            num_factors=num_factors,
            layers=layers,
            batch_size=int(config["batch_size"]),
            lr=float(config["learning_rate"]),
            num_neg=int(config["num_neg"]),
        )
        return NeuMF(
            name=f"NeuMF/tune/{scope.dataset}/{scope.recipe}/s{seed}",
            seed=seed,
            verbose=False,
            **kwargs,
        )

    if model_name == "bpr":
        from ..lib.bpr_cpu import BPRMiniBatch
        # One tied lambda fanned out to all three. NewBPR §5.2's claim is that
        # *separated* lambdas matter, which is a claim about their interaction --
        # and coordinate descent tunes each with the others pinned, so it can
        # never visit that joint optimum. Sweeping one tied value is the honest
        # reduction; see BPR["lambda_shared"] in lib/tuning_config.py.
        lam = float(config["lambda_shared"])
        return BPRMiniBatch(
            name=f"BPR/tune/{scope.dataset}/{scope.recipe}/s{seed}",
            k=int(config["k_embed_dim"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            lambda_u=lam,
            lambda_i=lam,
            lambda_j=lam,
            n_epochs=int(config["n_epochs"]),
            sampler=scope.recipe,
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


def _trial_path(model_name: str, scope: Scope, knob: str, value) -> Path:
    d = scope.dir(model_name) / knob
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{_value_slug(value)}.json"


def _baseline_path(model_name: str, scope: Scope) -> Path:
    d = scope.dir(model_name)
    d.mkdir(parents=True, exist_ok=True)
    return d / "baseline.json"


def _winner_path(model_name: str, scope: Scope) -> Path:
    return scope.dir(model_name) / "winner.json"


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

def _train_once(model_name: str, config: dict, eval_method, scope: Scope) -> tuple:
    """Build the model, run cornac.Experiment, return (val, test, wall_seconds).

    Selection reads `val`. `test` is kept only so the file records what the
    configuration scored on the held-out period.
    """
    model = _build_model(model_name, config, scope)
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
    if exp.val_result is None or len(exp.val_result) == 0:
        # Falling back to exp.result would silently select on test.
        raise RuntimeError("cornac.Experiment returned no validation result; "
                           "the split must have a val_set to tune on")
    if exp.result is None or len(exp.result) == 0:
        raise RuntimeError("cornac.Experiment returned no test result")
    val = dict(exp.val_result[0].metric_avg_results)
    test = dict(exp.result[0].metric_avg_results)
    return val, test, wall


def _canon(config) -> str:
    """Config as comparable JSON: tuples become lists, key order is fixed."""
    return json.dumps(config, default=str, sort_keys=True)


def _usable(cached: dict, config: dict) -> bool:
    """Reuse a cached baseline/trial only if it was selected on validation and
    trained at exactly `config`. See "A cache hit needs the same config"."""
    return ("val_metrics" in cached
            and _canon(json.loads(_canon(cached.get("config")))) == _canon(
                json.loads(_canon(config))))


def _record(config: dict, val: dict, test: dict, wall: float, **extra) -> dict:
    # `metrics` stays as an alias of the selection metrics so any reader that
    # only wants "the score this config was picked on" keeps working.
    return {**extra, "config": dict(config), "select_split": "validation",
            "val_metrics": val, "test_metrics": test, "metrics": val,
            "wall_seconds": wall}


def _build_eval_method(scope: Scope):
    # The default arm is `uniform` so hyperparameters are not chosen under the
    # mechanism being evaluated -- see the note in the module docstring.
    return build_eval_method(scope.dataset, neg_sampling=scope.recipe,
                             seed=scope.seed)


# ---------------------------------------------------------------------------
# Coordinate-descent loop.
# ---------------------------------------------------------------------------

def run_tuning(model_name: str, scope: Scope, restart_from: str = None) -> None:
    print(f"\n=== Tuning {model_name} on {scope}, "
          f"selecting on validation {SELECT_METRIC} ===", flush=True)
    print(f"    -> {scope.dir(model_name)}", flush=True)

    eval_method = _build_eval_method(scope)
    config = dict(BASELINE_CONFIG[model_name])

    # --- Baseline ---
    bpath = _baseline_path(model_name, scope)
    baseline = _load_json(bpath) if bpath.exists() else None
    if baseline is not None and _usable(baseline, config):
        print(f"[baseline] cached: val {SELECT_METRIC}="
              f"{baseline['val_metrics'][SELECT_METRIC]:.4f}", flush=True)
    else:
        if baseline is not None:
            print("[stale] baseline: selected on test or trained at another "
                  "config; re-running", flush=True)
        print(f"[baseline] running with defaults: {config}", flush=True)
        val, test, wall = _train_once(model_name, config, eval_method, scope)
        baseline = _record(config, val, test, wall)
        _write_json(bpath, baseline)
        print(f"[baseline] val {SELECT_METRIC}={val[SELECT_METRIC]:.4f}  "
              f"test={test[SELECT_METRIC]:.4f}  ({wall:.1f}s)", flush=True)

    # Running winner: config dict, its validation metric (selects) and its test
    # metric (recorded). The "default value" for the current knob lives at
    # config[knob].
    prev_winner_config = dict(baseline["config"])
    prev_winner_metric = float(baseline["val_metrics"][SELECT_METRIC])
    prev_winner_test = float(baseline["test_metrics"][SELECT_METRIC])
    config = dict(prev_winner_config)

    # If --restart-from was given, fast-forward through earlier knobs by
    # reading their cached winners off disk; raise if any are missing.
    if restart_from is not None:
        for knob, _ in TUNE_ORDER[model_name]:
            if knob == restart_from:
                break
            # Reuse cached trials to pick the winner without retraining.
            winner_v, winner_m, winner_t = _resolve_cached_winner(
                model_name, scope, knob, config, prev_winner_metric,
                prev_winner_test)
            config[knob] = winner_v
            prev_winner_config = dict(config)
            prev_winner_metric, prev_winner_test = winner_m, winner_t

    # --- Per-knob grid sweep ---
    started = restart_from is None
    for knob, grid in TUNE_ORDER[model_name]:
        if not started:
            started = (knob == restart_from)
            if not started:
                continue
        print(f"\n--- knob: {knob}  grid={grid}  "
              f"(default carries metric={prev_winner_metric:.4f}) ---", flush=True)

        results = []  # list of (value, val metric, test metric)
        for value in grid:
            tpath = _trial_path(model_name, scope, knob, value)
            trial_config = dict(config)
            trial_config[knob] = value
            trial = _load_json(tpath) if tpath.exists() else None
            if trial is not None and _usable(trial, trial_config):
                m = float(trial["val_metrics"][SELECT_METRIC])
                t = float(trial["test_metrics"][SELECT_METRIC])
                print(f"[skip] {knob}={value}  val {SELECT_METRIC}={m:.4f}  "
                      f"[cached]", flush=True)
            else:
                if trial is not None:
                    print(f"[stale] {knob}={value}: selected on test or trained "
                          f"at another config; re-running", flush=True)
                val, test, wall = _train_once(model_name, trial_config,
                                              eval_method, scope)
                trial = _record(trial_config, val, test, wall,
                                knob=knob, value=value)
                _write_json(tpath, trial)
                m, t = float(val[SELECT_METRIC]), float(test[SELECT_METRIC])
                print(f"[run]  {knob}={value}  val {SELECT_METRIC}={m:.4f}  "
                      f"test={t:.4f}  ({wall:.1f}s)", flush=True)
            results.append((value, m, t))

        # Winner = best of (cached default carry-forward) vs grid trials, on
        # validation. max() keeps the first of equal scores, so a tie keeps the
        # default.
        default_value = prev_winner_config[knob]
        candidates = [(default_value, prev_winner_metric, prev_winner_test)] + results
        best_value, best_metric, best_test = max(candidates, key=lambda x: x[1])
        config[knob] = best_value
        prev_winner_config = dict(config)
        prev_winner_metric, prev_winner_test = best_metric, best_test
        marker = " (kept default)" if best_value == default_value else ""
        print(f"[winner@{knob}] {knob}={best_value}  "
              f"{SELECT_METRIC}={best_metric:.4f}{marker}", flush=True)

    # --- Persist final winner ---
    winner_payload = {
        "model": model_name,
        # The variant is otherwise only in the path, and these files are synced
        # between machines -- a stray winner.json must still say what produced it.
        "variant": tuning_model_dir(model_name),
        "dataset": scope.dataset,
        "seed": scope.seed,
        "recipe": scope.recipe,
        "select_metric": SELECT_METRIC,
        # winner_config() refuses a winner without this -- one written before
        # the fix was selected on test.
        "select_split": "validation",
        "config": prev_winner_config,
        "metric": prev_winner_metric,
        "test_metric": prev_winner_test,
    }
    _write_json(_winner_path(model_name, scope), winner_payload)
    print(f"\n=== {model_name} [{scope}] winner: {prev_winner_config}  "
          f"val {SELECT_METRIC}={prev_winner_metric:.4f}  "
          f"test={prev_winner_test:.4f} ===\n", flush=True)


def _resolve_cached_winner(model_name: str, scope: Scope, knob: str,
                           config: dict, default_metric: float,
                           default_test: float):
    """For --restart-from: replay a knob's winner from cached trial JSONs.

    Returns (value, val metric, test metric). `config` is the running winner
    config with this knob still at its default.
    """
    grid = dict(TUNE_ORDER[model_name])[knob]
    candidates = [(config[knob], default_metric, default_test)]
    for value in grid:
        tpath = _trial_path(model_name, scope, knob, value)
        trial_config = dict(config)
        trial_config[knob] = value
        trial = _load_json(tpath) if tpath.exists() else None
        if trial is None or not _usable(trial, trial_config):
            raise SystemExit(
                f"cannot --restart-from past {knob!r}: no valid cached trial "
                f"{tpath} (missing, selected on test, or trained at another "
                f"config)")
        candidates.append((value, float(trial["val_metrics"][SELECT_METRIC]),
                           float(trial["test_metrics"][SELECT_METRIC])))
    return max(candidates, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=sorted(TUNE_ORDER.keys()),
                   help="Which model to tune.")
    p.add_argument("--dataset", default=DEFAULT_DATASET, choices=sorted(DATASETS),
                   help=f"Dataset key (default: {DEFAULT_DATASET}).")
    p.add_argument("--recipe", default=DEFAULT_RECIPE, choices=list(NEG_SAMPLING_MODES),
                   help=f"Sampling arm to tune under (default: {DEFAULT_RECIPE}). "
                        "Tuning on 'uniform' keeps hyperparameters from being "
                        "chosen under the mechanism being evaluated.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help=f"Seed for the split and the model (default: {DEFAULT_SEED}).")
    p.add_argument("--restart-from", default=None,
                   help="Knob name to resume from. Earlier knobs' winners are "
                        "resolved from cached trial JSONs without retraining.")
    args = p.parse_args()
    scope = Scope(dataset=args.dataset, recipe=args.recipe, seed=args.seed)
    run_tuning(args.model, scope, restart_from=args.restart_from)


if __name__ == "__main__":
    main()
