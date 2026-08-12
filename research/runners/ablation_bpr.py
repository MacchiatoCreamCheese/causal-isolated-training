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
from ..lib.tuning_config import BPR_EARLY_STOP
from ..paths import logs_dir


# NewBPR §5.2 training protocol: train up to 1000 epochs with early stopping
# on NDCG@20 (patience=13). The actual stop epoch is data-dependent.
BPR_EPOCHS = 1000


def build_model(ds_name: str, seed: int, recipe: str):
    return BPRMiniBatch(
        name=f"{ds_name}/{recipe}/s{seed}",
        k=64, batch_size=4096, learning_rate=0.05,
        lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
        n_epochs=BPR_EPOCHS, sampler=recipe,
        **BPR_EARLY_STOP,
        seed=seed, verbose=False,
    )


def run_one(ds_name: str, seed: int) -> None:
    recipes_out = load_partial("BPR", ds_name, seed)
    pending = [r for r in RECIPES if r not in recipes_out]
    if not pending:
        print(f"[skip] BPR {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## BPR / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(RECIPES)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)

    for recipe in pending:
        print(f"\n--- {recipe} ---", flush=True)
        t0 = time.time()
        model = build_model(ds_name, seed, recipe)
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(exp)
        # BPR owns its sampler, so the probe comes off the model here rather
        # than off the training split.
        metrics["counterfactual_rate"] = float(model.counterfactual_rate)
        metrics["collision_rate"] = float(model.collision_rate)
        recipes_out[recipe] = metrics
        write_partial("BPR", ds_name, seed, recipes_out)
        print(f"[{recipe}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### BPR/{ds_name}/seed={seed} total: {(time.time()-t_start)/60:.1f} min ####",
          flush=True)


def main():
    seeds, datasets = parse_args(default_seeds=(42,))
    for ds_name in datasets:
        for seed in seeds:
            run_one(ds_name, seed)


if __name__ == "__main__":
    main()
