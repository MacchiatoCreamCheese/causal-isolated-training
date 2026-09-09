"""LightGCN causal-sampling ablation: dataset × recipe × seed.

Uses **cornac's own** `cornac.models.LightGCN` unmodified. Its training loop
iterates `train_set.uij_iter(batch_size=..., shuffle=True)`
(`cornac/models/lightgcn/recom_lightgcn.py`) without naming a sampling mode,
so our `TimeAwareDataset` decides — the arm is a property on the training
split, not a model argument.

Two cells per (dataset, seed) — `uniform` vs `causal` — checkpointed to
research/results/ablation/lightgcn_<dataset>_seed<n>.json.

Requires `dgl` (cornac imports it lazily for LightGCN).

Usage:
    python -m research.runners.ablation_lightgcn --seeds 42,123,2026 --datasets baby
"""

import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import LightGCN

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    RECIPES, parse_args, extract_metrics, load_partial, write_partial, set_recipe,
)
# He 2020 §4.1.2, with each value's provenance recorded alongside it. Imported
# rather than restated: `runners/tuning.py` needs the same three constants, and
# it used to reach them by importing this runner.
from ..lib.tuning_config import lightgcn_kwargs
from ..lib.tuning_config import ablation_label
from ..paths import logs_dir


#: Encodes any non-default switches (see tuning_config.ablation_label),
#: so a variant run cannot overwrite the baseline's seed JSONs.
MODEL = ablation_label("LightGCN")


def run_one(ds_name: str, seed: int) -> None:
    # One source for both the constructor and the provenance stamp.
    configs = {r: lightgcn_kwargs(ds_name, r) for r in RECIPES}
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
