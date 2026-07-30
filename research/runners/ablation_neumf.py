"""NeuMF causal-sampling ablation: dataset × recipe × seed.

Uses **cornac's own** `cornac.models.NeuMF` unmodified. Its training loop
draws negatives via `train_set.uir_iter(..., num_zeros=num_neg)`
(`cornac/models/ncf/recom_ncf_base.py`), which our `TimeAwareDataset`
overrides — so switching the arm is a property on the training split, not a
model argument. The model never learns that anything changed.

Two cells per (dataset, seed) — `uniform` vs `causal` — checkpointed to
research/results/ablation/neumf_<dataset>_seed<n>.json.

Usage:
    python -m research.runners.ablation_neumf --seeds 42,123,2026 --datasets baby
"""

import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import NeuMF

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    RECIPES, parse_args, extract_metrics, load_partial, write_partial, set_recipe,
)
from ..paths import logs_dir


# He 2017 defaults, as inventoried in lib/tuning_config.py:NEUMF.
# backend="pytorch" because cornac's default TF backend has no GPU support on
# native Windows; both backends take the same uir_iter path.
NEUMF_KWARGS = dict(
    num_factors=8,
    layers=(64, 32, 16, 8),
    act_fn="relu",
    num_epochs=20,
    batch_size=256,
    lr=1e-3,
    num_neg=4,
    backend="pytorch",
)


def run_one(ds_name: str, seed: int) -> None:
    recipes_out = load_partial("NeuMF", ds_name, seed)
    pending = [r for r in RECIPES if r not in recipes_out]
    if not pending:
        print(f"[skip] NeuMF {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## NeuMF / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(RECIPES)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, verbose=True)

    for recipe in pending:
        print(f"\n--- {recipe} ---", flush=True)
        t0 = time.time()
        train_set = set_recipe(eval_method, recipe)
        model = NeuMF(name=f"NeuMF/{ds_name}/{recipe}/s{seed}", seed=seed,
                      verbose=True, **NEUMF_KWARGS)
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(exp)
        # Probe comes off the split: cornac's model has no idea negatives are
        # being sampled for it.
        metrics["counterfactual_rate"] = float(train_set.counterfactual_rate)
        recipes_out[recipe] = metrics
        write_partial("NeuMF", ds_name, seed, recipes_out)
        print(f"[{recipe}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### NeuMF/{ds_name}/seed={seed} total: {(time.time()-t_start)/60:.1f} min ####",
          flush=True)


def main():
    seeds, datasets = parse_args(default_seeds=(42,))
    for ds_name in datasets:
        for seed in seeds:
            run_one(ds_name, seed)


if __name__ == "__main__":
    main()
