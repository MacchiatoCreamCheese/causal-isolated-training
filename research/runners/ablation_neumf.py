"""NeuMF causal-sampling ablation: dataset × recipe × seed.

Uses **cornac's own** `cornac.models.NeuMF`, unmodified everywhere the ablation
measures. Its training loop draws negatives via
`train_set.uir_iter(..., num_zeros=num_neg)`
(`cornac/models/ncf/recom_ncf_base.py`), which our `TimeAwareDataset`
overrides — so switching the arm is a property on the training split, not a
model argument. The model never learns that anything changed.

The one thing subclassed is `save()`, whose pytorch branch cornac left raising
and which would otherwise abort each cell after training; see `SavableNeuMF`
below. It still writes the same `.pkl` cornac writes for every other model.

Two cells per (dataset, seed) — `uniform` vs `causal` — checkpointed to
research/results/ablation/neumf_<dataset>_seed<n>.json.

Usage:
    python -m research.runners.ablation_neumf --seeds 42,123,2026 --datasets baby
"""

import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall
from cornac.models import NeuMF
from cornac.models.recommender import Recommender

from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.ablation_harness import (
    RECIPES, parse_args, extract_metrics, load_partial, write_partial, set_recipe,
)
from ..paths import logs_dir


class SavableNeuMF(NeuMF):
    """cornac's NeuMF, saved the way every other cornac model is.

    `NCFBase.save()` writes the pickle first and only then trips over its own
    unfinished business::

        model_file = Recommender.save(self, save_dir)   # the .pkl, written fine
        if self.backend == "tensorflow":
            self.model.save_weights(model_file.replace(".pkl", ".h5"))
        elif self.backend == "pytorch":
            raise NotImplementedError()                 # TODO, still open in 2.6.0

    The unimplemented part is only the *separate backend weight export* -- the
    `.h5` sidecar the TensorFlow path writes. cornac raises rather than skipping
    it, and since `Experiment.run()` calls `save()` whenever `save_dir` is set,
    a plain NeuMF trains, evaluates, writes its pickle, and *then* aborts,
    taking the metrics down with it before `write_partial` can checkpoint them.

    So this skips exactly the raise and nothing else, going straight to
    `Recommender.save` for the same `.pkl` + `.meta` any other cornac model
    produces. The pickle holds the torch module and round-trips, so nothing is
    lost by not writing the sidecar. Training, evaluation, and negative sampling
    are untouched.
    """

    def save(self, save_dir=None, *args, **kwargs):
        if save_dir is None:
            return None
        return Recommender.save(self, save_dir, *args, **kwargs)


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
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)

    for recipe in pending:
        print(f"\n--- {recipe} ---", flush=True)
        t0 = time.time()
        train_set = set_recipe(eval_method, recipe)
        model = SavableNeuMF(name=f"NeuMF/{ds_name}/{recipe}/s{seed}",
                             seed=seed, verbose=True, **NEUMF_KWARGS)
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
        metrics["collision_rate"] = float(train_set.collision_rate)
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
