import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.bpr_cpu import BPRMiniBatch
from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    RECIPES, parse_args, extract_metrics, load_partial, write_partial,
)
from ..lib.tuning_config import BPR_EARLY_STOP, bpr_kwargs
from ..lib.tuning_config import ablation_label
from ..paths import logs_dir


MODEL = ablation_label("BPR")


def cell_configs(ds_name: str, seed: int):
    return {r: {**bpr_kwargs(ds_name, r, seed), **BPR_EARLY_STOP} for r in RECIPES}


def build_model(ds_name: str, seed: int, recipe: str, kwargs: dict):
    return BPRMiniBatch(
        name=f"{ds_name}/{recipe}/s{seed}",
        **kwargs, sampler=recipe, seed=seed, verbose=False,
    )


def run_one(ds_name: str, seed: int) -> None:
    configs = cell_configs(ds_name, seed)
    recipes_out = load_partial(MODEL, ds_name, seed, configs)
    pending = [r for r in RECIPES if r not in recipes_out]
    if not pending:
        print(f"[skip] {MODEL} {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## BPR / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(RECIPES)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)

    for recipe in pending:
        print(f"\n--- {recipe} ---", flush=True)
        t0 = time.time()
        model = build_model(ds_name, seed, recipe, configs[recipe])
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(exp, probe=model)
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
