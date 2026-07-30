"""Phase 5: BPR 2x2 ablation across baby/cellphone/healthcare, multi-seed.

For each (dataset, seed) combo we run the same 4 cells used in phase 2/3:

  shuffle+uniform / shuffle+causal / win10+uniform / win10+causal

and dump HR@20 / NDCG@20 / Recall@20 (plus counterfactual_rate) to
research/results/2x2/bpr_<dataset>_seed<n>.json. Aggregation across
seeds is done by research/aggregate_2x2.py.

Usage:
  python -m research.runners.ablation_bpr --seeds 42,123,2026 --datasets baby
"""

import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.bpr_gpu import BPRMiniBatchGPU as BPRMiniBatch
from ..lib.bpr_gpu import BPRWindowedGPU as BPRWindowed
from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    parse_args, extract_metrics, load_partial, write_partial,
)
from ..lib.tuning_config import BPR_EARLY_STOP
from ..paths import logs_dir


# NewBPR §5.2 training protocol: train up to 1000 epochs with early stopping
# on NDCG@20 (patience=13). The actual stop epoch is data-dependent.
BPR_EPOCHS = 1000


def build_cells(ds_name: str, seed: int):
    common = dict(k=64, batch_size=16384, learning_rate=0.05,
                  lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
                  **BPR_EARLY_STOP,
                  seed=seed, verbose=False)
    return [
        ("shuffle+uniform", lambda: BPRMiniBatch(name=f"{ds_name}/shuffle+uniform/s{seed}",
                                                  n_epochs=BPR_EPOCHS, sampler="uniform", **common)),
        ("shuffle+causal",  lambda: BPRMiniBatch(name=f"{ds_name}/shuffle+causal/s{seed}",
                                                  n_epochs=BPR_EPOCHS, sampler="causal",  **common)),
        ("win10+uniform",   lambda: BPRWindowed(name=f"{ds_name}/win10+uniform/s{seed}",
                                                  n_windows=10, epochs_per_window=BPR_EPOCHS,
                                                  sampler="uniform", **common)),
        ("win10+causal",    lambda: BPRWindowed(name=f"{ds_name}/win10+causal/s{seed}",
                                                  n_windows=10, epochs_per_window=BPR_EPOCHS,
                                                  sampler="causal",  **common)),
    ]


def run_one(ds_name: str, seed: int) -> None:
    recipes_out = load_partial("BPR", ds_name, seed)
    cells = build_cells(ds_name, seed)
    pending = [(c, ctor) for c, ctor in cells if c not in recipes_out]
    if not pending:
        print(f"[skip] BPR {ds_name} seed={seed} all 4 cells cached", flush=True)
        return

    print(f"\n############## BPR / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(cells)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, verbose=True)

    for cname, ctor in pending:
        print(f"\n--- {cname} ---", flush=True)
        t0 = time.time()
        model = ctor()
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        metrics = extract_metrics(exp)
        metrics["counterfactual_rate"] = float(getattr(model, "counterfactual_rate", 0.0))
        recipes_out[cname] = metrics
        write_partial("BPR", ds_name, seed, recipes_out)
        print(f"[{cname}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### BPR/{ds_name}/seed={seed} total: {(time.time()-t_start)/60:.1f} min ####",
          flush=True)


def main():
    seeds, datasets = parse_args(default_seeds=(42,),
                                 default_datasets=("baby", "cellphone", "healthcare"))
    for ds_name in datasets:
        for seed in seeds:
            run_one(ds_name, seed)


if __name__ == "__main__":
    main()
