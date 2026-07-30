"""2x2 ablation runner for NeuMF: dataset × recipe × seed.

For each (dataset, seed) combination, runs all four recipe cells —
shuffle+uniform, shuffle+causal, win10+uniform, win10+causal — and
checkpoints per-recipe to research/results/2x2/neumf_<dataset>_seed<n>.json.

Usage:
    python -m research.runners.ablation_neumf --seeds 42,123,2026 --datasets baby
"""

import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    parse_args, extract_metrics, load_partial, write_partial,
)
from ..lib.neumf import NeuMFRecommender, DEVICE
from ..paths import logs_dir


print(f"Using device: {DEVICE}", flush=True)


def build_cells(ds_name: str, seed: int):
    # Paper defaults (NeuMF §3.4 / §4.1): factors=8, layers=[64,32,16,8],
    # ReLU, Adam for pre-train, SGD for fine-tune, batch=256, lr=1e-3, num_neg=4.
    common = dict(
        num_factors=8,
        layers=(64, 32, 16, 8),
        act_fn="relu",
        batch_size=256,
        learning_rate=1e-3,
        num_neg=4,
        alpha=0.5,
        seed=seed,
        verbose=True,
    )
    return [
        ("shuffle+uniform", lambda: NeuMFRecommender(
            name=f"NeuMF/{ds_name}/shuffle+uniform/s{seed}", n_epochs=20,
            order="shuffle", sampler="uniform", **common)),
        ("shuffle+causal", lambda: NeuMFRecommender(
            name=f"NeuMF/{ds_name}/shuffle+causal/s{seed}", n_epochs=20,
            order="shuffle", sampler="causal", **common)),
        ("win10+uniform", lambda: NeuMFRecommender(
            name=f"NeuMF/{ds_name}/win10+uniform/s{seed}", n_epochs=20,
            order="windowed", sampler="uniform", n_windows=10, **common)),
        ("win10+causal", lambda: NeuMFRecommender(
            name=f"NeuMF/{ds_name}/win10+causal/s{seed}", n_epochs=20,
            order="windowed", sampler="causal", n_windows=10, **common)),
    ]


def run_one(ds_name: str, seed: int) -> None:
    recipes_out = load_partial("NeuMF", ds_name, seed)
    cells = build_cells(ds_name, seed)
    pending = [(c, ctor) for c, ctor in cells if c not in recipes_out]
    if not pending:
        print(f"[skip] NeuMF {ds_name} seed={seed} all 4 cells cached", flush=True)
        return

    print(f"\n############## NeuMF / {ds_name} / seed={seed} "
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
        write_partial("NeuMF", ds_name, seed, recipes_out)
        print(f"[{cname}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### NeuMF/{ds_name}/seed={seed} total: {(time.time()-t_start)/60:.1f} min ####",
          flush=True)


def main():
    seeds, datasets = parse_args(default_seeds=(42,),
                                 default_datasets=("baby",))
    for ds_name in datasets:
        for seed in seeds:
            run_one(ds_name, seed)


if __name__ == "__main__":
    main()
