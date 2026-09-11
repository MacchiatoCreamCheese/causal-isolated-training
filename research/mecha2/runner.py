"""Mechanism 2 ablation: the cumulative ladder.

Four cells per (dataset, seed), each adding one change to the last::

    uniform            baseline -- cornac's own sampling and batching
    causal             + Mechanism 1, causal negative sampling
    causal+coherent    + time-coherent batches, visited in random order
    causal+temporal    + the same batches, visited in time order

**Why `coherent` sits between them.** `temporal` changes two things at once:
each batch covers one band of time (*coherence*), and the bands are visited in
time order, so every epoch ends on the newest rows (*recency* -- SGD weights its
latest updates most). `coherent` has the first without the second, so the step
`causal -> causal+coherent` measures coherence alone and
`causal+coherent -> causal+temporal` measures recency alone. Without it, a gain
from `temporal` could not be attributed to either.

**Cumulative, not factorial.** `uniform+temporal` is deliberately not run, so the
gain attributed to Mechanism 2 is measured *given* Mechanism 1: it answers "does
temporal batching improve our method", not "does temporal batching help in
general". That is the claim being made -- but the missing cell cannot be
recovered later without running it, so the omission is recorded here rather than
inferred from an absence.

Hyperparameters: `causal+temporal` reuses the **causal** arm's tuned winners
rather than being tuned itself, so the two differ in batching alone. Under the
default `RESEARCH_TUNED_ARM=uniform` every arm resolves to the uniform arm's
winners and this is moot; set `RESEARCH_TUNED_ARM=per_arm` to have the causal
arms use their own.

All three models are supported, by two different routes. NeuMF and LightGCN
batch through `uij_iter` / `uir_iter` and so inherit Mechanism 2 from the data
loader with no model-side code. Our NumPy BPR batches itself, so it arrives via
`mecha2/bpr.py:TemporalBPR`, which overrides the one `_epoch_batches` seam --
see `batching.M2_MODELS`.

Usage:
    python -m research.mecha2.runner --datasets musical --seeds 42
    python -m research.mecha2.runner --model lightgcn --seeds 42,123,2026
"""

import argparse
import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.ablation_harness import (
    DATASETS, extract_metrics, load_partial, write_partial,
)
from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.tuning_config import (
    BPR_EARLY_STOP, ablation_label, bpr_kwargs, lightgcn_kwargs, neumf_kwargs,
)
from ..paths import logs_dir
from .batching import M2_MODELS, set_arm, use_temporal_batching

#: (arm name, negative sampler, batch order, which arm's winners it uses).
#: Order is the ladder order, and `load_partial` keys cells by these names.
ARMS = (
    ("uniform",         "uniform", "shuffle",  "uniform"),
    ("causal",          "causal",  "shuffle",  "causal"),
    ("causal+coherent", "causal",  "coherent", "causal"),
    ("causal+temporal", "causal",  "temporal", "causal"),
)

DEFAULT_DATASETS = ("musical", "baby", "cellphone")


def _build(model_name, ds_name, recipe, seed, tuned_as, neg_sampling):
    """Model plus the exact kwargs it was built from, for the provenance stamp."""
    if model_name == "neumf":
        from ..lib.cornac_compat import NeuMF
        kwargs = neumf_kwargs(ds_name, tuned_as)
        model = NeuMF(name=f"NeuMF-m2/{ds_name}/{recipe}/s{seed}",
                      seed=seed, verbose=True, **kwargs)
    elif model_name == "lightgcn":
        from cornac.models import LightGCN
        kwargs = lightgcn_kwargs(ds_name, tuned_as)
        model = LightGCN(name=f"LightGCN-m2/{ds_name}/{recipe}/s{seed}",
                         seed=seed, verbose=True, **kwargs)
    elif model_name == "bpr":
        # TemporalBPR, not BPRMiniBatch: our BPR batches itself, so it only picks
        # up Mechanism 2 through the `_epoch_batches` seam. See mecha2/bpr.py.
        from .bpr import TemporalBPR
        kwargs = bpr_kwargs(ds_name, tuned_as)
        model = TemporalBPR(name=f"BPR-m2/{ds_name}/{recipe}/s{seed}",
                            **kwargs, **BPR_EARLY_STOP,
                            sampler=neg_sampling, seed=seed, verbose=False)
    else:
        raise SystemExit(
            f"--model {model_name} cannot receive Mechanism 2. "
            f"Supported: {', '.join(M2_MODELS)}.")
    return model, kwargs


def run_one(model_name: str, label: str, ds_name: str, seed: int) -> None:
    # batch_order is part of the recorded config, so re-running after changing an
    # arm's ordering re-runs that cell instead of silently resuming a stale one.
    kwargs_for = {"neumf": neumf_kwargs, "lightgcn": lightgcn_kwargs,
                  "bpr": bpr_kwargs}[model_name]
    configs = {
        arm: {**kwargs_for(ds_name, tuned_as),
              "batch_order": order, "neg_sampling": neg}
        for arm, neg, order, tuned_as in ARMS
    }

    recipes_out = load_partial(label, ds_name, seed, configs)
    pending = [a for a, *_ in ARMS if a not in recipes_out]
    if not pending:
        print(f"[skip] {label} {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## {label} / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(ARMS)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)
    # Upgrade once; the arms then differ only by two attributes on this split.
    train_set = use_temporal_batching(eval_method, batch_order="shuffle")

    for arm, neg, order, tuned_as in ARMS:
        if arm not in pending:
            continue
        print(f"\n--- {arm} (neg={neg}, batches={order}) ---", flush=True)
        t0 = time.time()
        set_arm(train_set, neg, order)
        model, _kwargs = _build(model_name, ds_name, arm, seed, tuned_as, neg)
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        # Probe off the split: cornac's models have no idea negatives are being
        # sampled, nor that their batches were reordered.
        metrics = extract_metrics(exp, probe=train_set)
        recipes_out[arm] = metrics
        write_partial(label, ds_name, seed, recipes_out, configs)
        print(f"[{arm}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### {label}/{ds_name}/seed={seed} total: "
          f"{(time.time()-t_start)/60:.1f} min ####", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="neumf", choices=sorted(M2_MODELS),
                   help="neumf/lightgcn receive Mechanism 2 through the data "
                        "loader; bpr through the TemporalBPR subclass.")
    p.add_argument("--seeds", default="42")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {list(DATASETS)})")

    # `-m2` keeps these cells from colliding with the Mechanism-1 ablation's,
    # and `ablation_label` layers any active variant switches on top.
    base = {"neumf": "NeuMF", "lightgcn": "LightGCN", "bpr": "BPR"}[args.model]
    label = ablation_label(f"{base}-m2")
    for ds_name in datasets:
        for seed in seeds:
            run_one(args.model, label, ds_name, seed)


if __name__ == "__main__":
    main()
