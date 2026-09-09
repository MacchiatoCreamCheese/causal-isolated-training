"""BPR causal-sampling ablation across datasets, multi-seed.

Two cells per (dataset, seed) — `uniform` vs `causal` — dumping
HR@20 / NDCG@20 / Recall@20 plus the counterfactual-negative rate to
research/results/ablation/bpr_<dataset>_seed<n>.json. Aggregation across
seeds is done by research.analysis.aggregate_ablation.

BPR is the one model that does not take its negatives from the data loader:
cornac's BPR trains in compiled Cython over `train_set.matrix`, a CSR that
carries no timestamps, so the causal rule cannot reach it. We therefore use
our own NumPy BPR, which reads the same first-seen index via
`lib.causal_negative_sampler.NumpyCausalSampler` and reports the probe itself.

Usage:
  python -m research.runners.ablation_bpr --seeds 42,123,2026 --datasets baby
"""

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


#: Encodes any non-default switches (see tuning_config.ablation_label),
#: so a variant run cannot overwrite the baseline's seed JSONs.
MODEL = ablation_label("BPR")


def cell_configs(ds_name: str):
    """`{recipe: kwargs}` for one dataset -- the single source for both the model
    and its provenance stamp, so the config recorded in a result file cannot
    drift from the one that produced it.

    Every hyperparameter comes off `lib/tuning_config.py`, which is where each
    one's provenance is recorded -- including the NewBPR §5.2 protocol (up to
    1000 epochs, early stopping on NDCG@20 with patience=13; the actual stop
    epoch is data-dependent). BPR_EARLY_STOP is merged in rather than passed
    separately at the call site, because patience changes the result and so
    belongs in what gets compared on resume.
    """
    return {r: {**bpr_kwargs(ds_name, r), **BPR_EARLY_STOP} for r in RECIPES}


def build_model(ds_name: str, seed: int, recipe: str, kwargs: dict):
    return BPRMiniBatch(
        name=f"{ds_name}/{recipe}/s{seed}",
        **kwargs, sampler=recipe, seed=seed, verbose=False,
    )


def run_one(ds_name: str, seed: int) -> None:
    configs = cell_configs(ds_name)
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
        # BPR owns its sampler, so the probe comes off the model here rather
        # than off the training split.
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
