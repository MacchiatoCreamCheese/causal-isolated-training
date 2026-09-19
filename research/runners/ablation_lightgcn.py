import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import LightGCN

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    RECIPES, parse_args, extract_metrics, load_partial, write_partial, set_recipe,
)
from ..lib.tuning_config import lightgcn_kwargs
from ..lib.tuning_config import ablation_label
from ..paths import logs_dir


MODEL = ablation_label("LightGCN")


def run_one(ds_name: str, seed: int) -> None:
    configs = {r: lightgcn_kwargs(ds_name, r, seed) for r in RECIPES}
    recipes_out = load_partial(MODEL, ds_name, seed, configs)
    pending = [r for r in RECIPES if r not in recipes_out]
    if not pending:
        print(f"[skip] {MODEL} {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## LightGCN / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(RECIPES)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)

    for recipe in pending:
        print(f"\n--- {recipe} ---", flush=True)
        t0 = time.time()
        train_set = set_recipe(eval_method, recipe)
        model = LightGCN(
            name=f"LightGCN/{ds_name}/{recipe}/s{seed}",
            **configs[recipe],
            seed=seed, verbose=True,
        )
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(exp, probe=train_set)
        recipes_out[recipe] = metrics
        write_partial(MODEL, ds_name, seed, recipes_out, configs)
        print(f"[{recipe}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### {MODEL}/{ds_name}/seed={seed} total: {(time.time()-t_start)/60:.1f} min ####",
          flush=True)


def main():
    seeds, datasets = parse_args(default_seeds=(42,))
    for ds_name in datasets:
        for seed in seeds:
            run_one(ds_name, seed)


if __name__ == "__main__":
    main()
