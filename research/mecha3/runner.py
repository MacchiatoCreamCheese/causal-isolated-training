"""Mechanism 3 runner: one prequential pass per arm, checkpointed.

Two arms, `uniform` and `causal`, each a single chronological pass over the
training rows with evaluation interleaved (see `prequential.py`). Both arms walk
the *same* rows in the *same* order and get the same budget, so the only thing
that differs between their curves is the negative pool.

Output: `results/prequential/<model>_<dataset>_seed<n>.json`, one record per
batch per arm. `analysis/make_figures.py` turns that into the timeline figure.

Usage:
    python -m research.mecha3.runner --datasets musical --seeds 42
"""

import argparse
import json
import os
import time

from ..lib.ablation_harness import DATASETS
from ..lib.data import build_eval_method
from ..lib.tuning_config import (
    BPR_EARLY_STOP, TUNED_ARM, ablation_label, bpr_kwargs, lightgcn_kwargs,
    neumf_kwargs,
)
from ..paths import RESULTS_DIR
from .adapters import build_adapter
from .prequential import run_prequential

OUT_DIR = RESULTS_DIR / "prequential"

#: Temporal batching is not an arm here: prequential *requires* chronological
#: batches, so Mechanism 2's ordering is part of the protocol rather than a
#: variable within it. The only axis left is the negative pool.
ARMS = ("uniform", "causal")

DEFAULT_DATASETS = ("musical", "baby", "cellphone")

KWARGS_FOR = {"bpr": bpr_kwargs, "neumf": neumf_kwargs,
              "lightgcn": lightgcn_kwargs}


def out_path(label, dataset, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR / f"{label.lower()}_{dataset}_seed{seed}.json"


def _make_model(model_name, kwargs, ds_name, arm, seed):
    """A model at `kwargs`, built the same way the ablation runners build it."""
    if model_name == "bpr":
        from ..lib.bpr_cpu import BPRMiniBatch
        # BPR_EARLY_STOP is dropped: early stopping needs a validation set and
        # repeated epochs, and a prequential pass has neither -- each row is seen
        # once, in order.
        return BPRMiniBatch(name=f"BPR-m3/{ds_name}/{arm}/s{seed}",
                            **kwargs, sampler=arm, seed=seed, verbose=False)
    if model_name == "neumf":
        from ..lib.cornac_compat import NeuMF
        return NeuMF(name=f"NeuMF-m3/{ds_name}/{arm}/s{seed}",
                     seed=seed, verbose=False, **kwargs)
    if model_name == "lightgcn":
        from cornac.models import LightGCN
        return LightGCN(name=f"LightGCN-m3/{ds_name}/{arm}/s{seed}",
                        seed=seed, verbose=False, **kwargs)
    raise ValueError(model_name)


def run_one(model_name, label, ds_name, seed, batch_size, warmup_frac,
            warmup_epochs, graph_every):
    path = out_path(label, ds_name, seed)
    if path.exists():
        print(f"[skip] {label} {ds_name} seed={seed} already done", flush=True)
        return

    print(f"\n##### {label} / {ds_name} / seed={seed} #####", flush=True)
    eval_method = build_eval_method(ds_name, seed=seed, verbose=True)
    train_set = eval_method.train_set

    arms, configs = {}, {}
    for arm in ARMS:
        # Each arm's own tuned winner when RESEARCH_TUNED_ARM=per_arm, so the
        # curves compare each method at its best; under the default both resolve
        # to the uniform arm's winners and the comparison stays single-variable.
        kwargs = KWARGS_FOR[model_name](ds_name, arm)
        if model_name == "bpr":
            kwargs = {**kwargs, "batch_size": batch_size}
        configs[arm] = {**kwargs, "stream_batch_size": batch_size,
                        "warmup_frac": warmup_frac,
                        "warmup_epochs": warmup_epochs,
                        "protocol": "prequential", "tuned_arm": TUNED_ARM}

        print(f"\n--- {arm} ---  {kwargs}", flush=True)
        t0 = time.time()
        model = _make_model(model_name, kwargs, ds_name, arm, seed)
        adapter = build_adapter(model_name, model, arm, graph_every=graph_every)
        records = run_prequential(adapter, train_set, batch_size=batch_size,
                                  warmup_frac=warmup_frac,
                                  warmup_epochs=warmup_epochs)
        arms[arm] = records
        s = adapter.sampler
        print(f"[{arm}] {len(records)} points  rho={s.counterfactual_rate * 100:.2f}%"
              f"  ({time.time() - t0:.1f}s)", flush=True)

    payload = {"model": label, "dataset": ds_name, "seed": seed,
               "configs": configs, "arms": arms}
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[prequential] wrote {path}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="bpr", choices=sorted(KWARGS_FOR))
    p.add_argument("--seeds", default="42")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Rows per point. Smaller = a denser, noisier curve; "
                        "this is the resolution knob.")
    p.add_argument("--warmup-frac", type=float, default=0.2,
                   help="Fraction of the timeline trained on normally before "
                        "streaming begins. Points before it are not reported.")
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--graph-every", type=int, default=1,
                   help="LightGCN only: rebuild the graph every N batches. >1 "
                        "lags the graph (safe -- fewer past edges, never future "
                        "ones) in exchange for speed.")
    args = p.parse_args()

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {list(DATASETS)})")

    base = {"bpr": "BPR", "neumf": "NeuMF", "lightgcn": "LightGCN"}[args.model]
    label = ablation_label(f"{base}-m3")
    for ds_name in datasets:
        for seed in seeds:
            run_one(args.model, label, ds_name, seed, args.batch_size,
                    args.warmup_frac, args.warmup_epochs, args.graph_every)


if __name__ == "__main__":
    main()
