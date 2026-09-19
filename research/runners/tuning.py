import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cornac
from cornac.metrics import AUC, NDCG, HitRatio, Recall

from ..lib import user_scores
from ..lib.ablation_harness import PROBE_METRICS
from ..lib.causal_sampling import TOP_K
from ..lib.data import DATASETS, build_eval_method
from ..lib.timeaware_data import NEG_SAMPLING_MODES
from ..lib.tuning_config import (
    BASELINE_CONFIG, BPR_EARLY_STOP, TUNE_ORDER, tuning_model_dir,
)
from ..paths import logs_dir, RESULTS_DIR


TUNE_DIR = RESULTS_DIR / "tuning"
SELECT_METRIC = f"NDCG@{TOP_K}"

DEFAULT_DATASET = "baby"
DEFAULT_SEED = 42
DEFAULT_RECIPE = "uniform"


@dataclass(frozen=True)
class Scope:
    dataset: str
    recipe: str
    seed: int

    def dir(self, model_name: str) -> Path:
        return (TUNE_DIR / tuning_model_dir(model_name) / self.dataset
                / self.recipe / f"seed{self.seed}")

    def __str__(self) -> str:
        return f"{self.dataset} x {self.recipe} x seed={self.seed}"


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
            batch_size=LIGHTGCN_BATCH.get(scope.dataset, 1024),
            num_epochs=LIGHTGCN_EPOCHS,
            early_stopping=LIGHTGCN_EARLY_STOP,
            seed=seed,
            verbose=False,
        )

    if model_name == "neumf":
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


def _train_once(model_name: str, config: dict, eval_method, scope: Scope) -> tuple:
    model = _build_model(model_name, config, scope)
    eval_method.train_set.reset_counterfactual_counters()
    exp = cornac.Experiment(
        eval_method=eval_method,
        models=[model],
        metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K), AUC()],
        user_based=True,
        save_dir=logs_dir(),
    )
    t0 = time.time()
    exp.run()
    wall = time.time() - t0
    if exp.val_result is None or len(exp.val_result) == 0:
        raise RuntimeError("cornac.Experiment returned no validation result; "
                           "the split must have a val_set to tune on")
    if exp.result is None or len(exp.result) == 0:
        raise RuntimeError("cornac.Experiment returned no test result")
    val = dict(exp.val_result[0].metric_avg_results)
    test = dict(exp.result[0].metric_avg_results)
    users = user_scores.from_experiment(exp)
    probe = model if model_name == "bpr" else eval_method.train_set
    probes = {name: float(getattr(probe, name)) for name in PROBE_METRICS}
    return val, test, users, probes, wall


def _users_path(json_path: Path) -> Path:
    return json_path.with_name(json_path.stem + ".users.npz")


def _winner_users(model_name: str, scope: Scope, eval_method, config: dict,
                  users_path: Path) -> dict:
    d = scope.dir(model_name)
    wpath = _winner_path(model_name, scope)
    if users_path.exists() and wpath.exists():
        old = _load_json(wpath)
        if old.get("users_run") and _usable({**old, "val_metrics": True}, config):
            print("[users] winner per-user scores already saved", flush=True)
            return old["users_run"]

    for jpath in [d / "baseline.json"] + sorted(d.glob("*/*.json")):
        npz = _users_path(jpath)
        if not (jpath.exists() and npz.exists()):
            continue
        rec = _load_json(jpath)
        if _usable(rec, config):
            shutil.copyfile(npz, users_path)
            rel = jpath.relative_to(d).as_posix()
            print(f"[users] winner per-user scores from {rel}", flush=True)
            return {"source": "trial", "trial": rel,
                    "val_metrics": rec["val_metrics"],
                    "test_metrics": rec["test_metrics"],
                    "probes": rec.get("probes")}

    print("[users] winner was trained before per-user scores were saved; "
          "retraining it once", flush=True)
    val, test, users, probes, wall = _train_once(model_name, config,
                                                 eval_method, scope)
    user_scores.save(users_path, users)
    print(f"[users] retrained winner: val {SELECT_METRIC}={val[SELECT_METRIC]:.4f}  "
          f"test={test[SELECT_METRIC]:.4f}  ({wall:.1f}s)", flush=True)
    return {"source": "retrain", "val_metrics": val, "test_metrics": test,
            "probes": probes, "wall_seconds": wall}


def _canon(config) -> str:
    return json.dumps(config, default=str, sort_keys=True)


def _usable(cached: dict, config: dict) -> bool:
    return ("val_metrics" in cached
            and _canon(json.loads(_canon(cached.get("config")))) == _canon(
                json.loads(_canon(config))))


def _record(config: dict, val: dict, test: dict, wall: float, **extra) -> dict:
    return {**extra, "config": dict(config), "select_split": "validation",
            "val_metrics": val, "test_metrics": test, "metrics": val,
            "wall_seconds": wall}


def _build_eval_method(scope: Scope):
    return build_eval_method(scope.dataset, neg_sampling=scope.recipe,
                             seed=scope.seed)


def run_tuning(model_name: str, scope: Scope, restart_from: str = None) -> None:
    print(f"\n=== Tuning {model_name} on {scope}, "
          f"selecting on validation {SELECT_METRIC} ===", flush=True)
    print(f"    -> {scope.dir(model_name)}", flush=True)

    eval_method = _build_eval_method(scope)
    config = dict(BASELINE_CONFIG[model_name])

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
        val, test, users, probes, wall = _train_once(model_name, config,
                                                     eval_method, scope)
        baseline = _record(config, val, test, wall, probes=probes)
        user_scores.save(_users_path(bpath), users)
        _write_json(bpath, baseline)
        print(f"[baseline] val {SELECT_METRIC}={val[SELECT_METRIC]:.4f}  "
              f"test={test[SELECT_METRIC]:.4f}  ({wall:.1f}s)", flush=True)

    prev_winner_config = dict(baseline["config"])
    prev_winner_metric = float(baseline["val_metrics"][SELECT_METRIC])
    prev_winner_test = float(baseline["test_metrics"][SELECT_METRIC])
    config = dict(prev_winner_config)

    if restart_from is not None:
        for knob, _ in TUNE_ORDER[model_name]:
            if knob == restart_from:
                break
            winner_v, winner_m, winner_t = _resolve_cached_winner(
                model_name, scope, knob, config, prev_winner_metric,
                prev_winner_test)
            config[knob] = winner_v
            prev_winner_config = dict(config)
            prev_winner_metric, prev_winner_test = winner_m, winner_t

    started = restart_from is None
    for knob, grid in TUNE_ORDER[model_name]:
        if not started:
            started = (knob == restart_from)
            if not started:
                continue
        print(f"\n--- knob: {knob}  grid={grid}  "
              f"(default carries metric={prev_winner_metric:.4f}) ---", flush=True)

        results = []
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
                val, test, users, probes, wall = _train_once(
                    model_name, trial_config, eval_method, scope)
                trial = _record(trial_config, val, test, wall,
                                knob=knob, value=value, probes=probes)
                user_scores.save(_users_path(tpath), users)
                _write_json(tpath, trial)
                m, t = float(val[SELECT_METRIC]), float(test[SELECT_METRIC])
                print(f"[run]  {knob}={value}  val {SELECT_METRIC}={m:.4f}  "
                      f"test={t:.4f}  ({wall:.1f}s)", flush=True)
            results.append((value, m, t))

        default_value = prev_winner_config[knob]
        candidates = [(default_value, prev_winner_metric, prev_winner_test)] + results
        best_value, best_metric, best_test = max(candidates, key=lambda x: x[1])
        config[knob] = best_value
        prev_winner_config = dict(config)
        prev_winner_metric, prev_winner_test = best_metric, best_test
        marker = " (kept default)" if best_value == default_value else ""
        print(f"[winner@{knob}] {knob}={best_value}  "
              f"{SELECT_METRIC}={best_metric:.4f}{marker}", flush=True)

    users_path = scope.dir(model_name) / "winner.users.npz"
    users_run = _winner_users(model_name, scope, eval_method,
                              prev_winner_config, users_path)
    winner_payload = {
        "model": model_name,
        "variant": tuning_model_dir(model_name),
        "dataset": scope.dataset,
        "seed": scope.seed,
        "recipe": scope.recipe,
        "select_metric": SELECT_METRIC,
        "select_split": "validation",
        "config": prev_winner_config,
        "metric": prev_winner_metric,
        "test_metric": prev_winner_test,
        "users_run": users_run,
    }
    _write_json(_winner_path(model_name, scope), winner_payload)
    for npz in scope.dir(model_name).rglob("*.users.npz"):
        if npz != users_path:
            npz.unlink()
    print(f"\n=== {model_name} [{scope}] winner: {prev_winner_config}  "
          f"val {SELECT_METRIC}={prev_winner_metric:.4f}  "
          f"test={prev_winner_test:.4f} ===\n", flush=True)


def _resolve_cached_winner(model_name: str, scope: Scope, knob: str,
                           config: dict, default_metric: float,
                           default_test: float):
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
