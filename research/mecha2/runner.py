import argparse
import json
import shutil
import time

import cornac
from cornac.metrics import AUC, NDCG, HitRatio, Recall

from ..lib import user_scores
from ..lib.ablation_harness import (
    DATASETS, OUT_DIR, extract_metrics, load_partial, write_partial,
)
from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.tuning_config import (
    BPR_EARLY_STOP, TUNED_ARM, ablation_label, bpr_kwargs, lightgcn_kwargs,
    neumf_kwargs, winner_dir,
)
from ..paths import logs_dir
from .batching import M2_MODELS, set_arm, use_temporal_batching

ARMS = (
    ("uniform",         "uniform", "shuffle",  "uniform"),
    ("causal",          "causal",  "shuffle",  "causal"),
    ("causal+coherent", "causal",  "coherent", "causal"),
    ("causal+temporal", "causal",  "temporal", "causal"),
)

FROM_TUNING = ("uniform", "causal")

KWARGS_FOR = {"neumf": neumf_kwargs, "lightgcn": lightgcn_kwargs,
              "bpr": bpr_kwargs}

DEFAULT_DATASETS = ("musical", "baby", "cellphone")

PER_USER_DIR = OUT_DIR / "per_user"


def per_user_path(label, ds_name, seed, rung):
    return PER_USER_DIR / f"{label.lower()}_{ds_name}_seed{seed}_{rung}.npz"


def _winner_epochs(model_name, ds_name, arm, seed):
    wpath = winner_dir(model_name, ds_name, arm, seed) / "winner.json"
    if not wpath.exists():
        return None
    with open(wpath, encoding="utf-8-sig") as f:
        winner = json.load(f)
    epochs = (winner.get("users_run") or {}).get("epochs")
    return int(epochs) if epochs else None


def _build(model_name, ds_name, rung, seed, tuned_as, neg_sampling):
    kwargs = KWARGS_FOR[model_name](ds_name, tuned_as, seed)
    if model_name == "neumf":
        from ..lib.cornac_compat import NeuMF
        model = NeuMF(name=f"NeuMF-m2/{ds_name}/{rung}/s{seed}",
                      seed=seed, verbose=True, **kwargs)
    elif model_name == "lightgcn":
        from cornac.models import LightGCN
        model = LightGCN(name=f"LightGCN-m2/{ds_name}/{rung}/s{seed}",
                         seed=seed, verbose=True, **kwargs)
    elif model_name == "bpr":
        from .bpr import TemporalBPR
        epochs = _winner_epochs(model_name, ds_name, tuned_as, seed)
        if epochs and rung not in FROM_TUNING:
            kwargs = {**kwargs, "n_epochs": epochs}
            stopping = {}
        else:
            stopping = BPR_EARLY_STOP
        model = TemporalBPR(name=f"BPR-m2/{ds_name}/{rung}/s{seed}",
                            **kwargs, **stopping,
                            sampler=neg_sampling, seed=seed, verbose=False)
    else:
        raise SystemExit(
            f"--model {model_name} cannot receive temporal batching. "
            f"Supported: {', '.join(M2_MODELS)}.")
    return model, kwargs


def _from_tuning(model_name, ds_name, seed, arm):
    d = winner_dir(model_name, ds_name, arm, seed)
    wpath, upath = d / "winner.json", d / "winner.users.npz"
    if not (wpath.exists() and upath.exists()):
        return None
    with open(wpath, encoding="utf-8-sig") as f:
        winner = json.load(f)
    run = winner.get("users_run")
    if winner.get("select_split") != "validation" or not run:
        return None
    metrics = {k: float(v) for k, v in run["test_metrics"].items()}
    metrics.update({k: float(v) for k, v in (run.get("probes") or {}).items()})
    metrics["from_tuning"] = 1.0
    return metrics, upath


def run_one(model_name: str, label: str, ds_name: str, seed: int) -> None:
    configs = {
        rung: {**KWARGS_FOR[model_name](ds_name, tuned_as, seed),
               "batch_order": order, "neg_sampling": neg}
        for rung, neg, order, tuned_as in ARMS
    }
    if model_name == "bpr":
        for rung, neg, order, tuned_as in ARMS:
            epochs = _winner_epochs(model_name, ds_name, tuned_as, seed)
            if epochs and rung not in FROM_TUNING:
                configs[rung]["n_epochs"] = epochs

    recipes_out = load_partial(label, ds_name, seed, configs)
    for rung in list(recipes_out):
        if not per_user_path(label, ds_name, seed, rung).exists():
            print(f"[stale] {rung}: no per-user scores, so it cannot be tested; "
                  f"re-running", flush=True)
            del recipes_out[rung]
    pending = [r for r, *_ in ARMS if r not in recipes_out]
    if not pending:
        print(f"[skip] {label} {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## {label} / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(ARMS)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = train_set = None

    for rung, neg, order, tuned_as in ARMS:
        if rung not in pending:
            continue
        target = per_user_path(label, ds_name, seed, rung)

        if rung in FROM_TUNING:
            got = _from_tuning(model_name, ds_name, seed, tuned_as)
            if got is not None:
                metrics, upath = got
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(upath, target)
                recipes_out[rung] = metrics
                write_partial(label, ds_name, seed, recipes_out, configs)
                print(f"[{rung}] from tuning winner {upath.parent}  {metrics}  "
                      f"[checkpointed]", flush=True)
                continue
            print(f"[{rung}] no tuning winner with per-user scores for seed "
                  f"{seed}; training it here", flush=True)

        if eval_method is None:
            eval_method = build_eval_method(ds_name, seed=seed, verbose=True)
            train_set = use_temporal_batching(eval_method, batch_order="shuffle")

        print(f"\n--- {rung} (neg={neg}, batches={order}) ---", flush=True)
        t0 = time.time()
        set_arm(train_set, neg, order)
        model, _kwargs = _build(model_name, ds_name, rung, seed, tuned_as, neg)
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K), AUC()],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(
            exp, probe=model if model_name == "bpr" else train_set)
        metrics["from_tuning"] = 0.0
        user_scores.save(target, user_scores.from_experiment(exp))
        recipes_out[rung] = metrics
        write_partial(label, ds_name, seed, recipes_out, configs)
        print(f"[{rung}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### {label}/{ds_name}/seed={seed} total: "
          f"{(time.time()-t_start)/60:.1f} min ####", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="neumf", choices=sorted(M2_MODELS),
                   help="neumf/lightgcn take batch order through the data "
                        "loader; bpr through the TemporalBPR subclass.")
    p.add_argument("--seeds", default="42")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    args = p.parse_args()

    if TUNED_ARM != "per_arm":
        raise SystemExit(
            "This ladder runs rung 1 at the uniform winners and rungs 2-4 at the "
            "causal winners, reusing both tuning winners for rungs 1-2. Run it "
            "with RESEARCH_TUNED_ARM=per_arm (run_ablation.sh sets it).")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {list(DATASETS)})")

    base = {"neumf": "NeuMF", "lightgcn": "LightGCN", "bpr": "BPR"}[args.model]
    label = ablation_label(f"{base}-m2")
    for ds_name in datasets:
        for seed in seeds:
            run_one(args.model, label, ds_name, seed)


if __name__ == "__main__":
    main()
