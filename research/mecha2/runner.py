"""The ablation ladder: four rungs per (model, dataset, seed).

Each rung adds one change to the last::

    uniform            baseline -- cornac's own sampling and batching
    causal             + past-only negative sampling
    causal+coherent    + time-coherent batches, visited in random order
    causal+temporal    + the same batches, visited in time order

**Which hyperparameters.** Rung 1 runs at the *uniform* arm's tuned winners, and
rungs 2-4 at the *causal* arm's winners, each at the cell's own seed. So:

  - uniform -> causal changes the sampler *and* the hyperparameters: it is the
    "each method at its best" comparison, reported as such, not as an ablation;
  - causal -> causal+coherent -> causal+temporal share one recipe, so each step
    is a clean single-variable ablation of batch order.

That is exactly `RESEARCH_TUNED_ARM=per_arm`, which this runner requires (and
which labels its files `-tunedper-arm`, keeping them apart from any other run).

**Rungs 1-2 are not retrained.** Tuning already trained both models -- they are
the uniform and causal tuning winners -- and kept their per-user test scores
(`runners/tuning.py`, `winner.users.npz`). Those are reused, with the test
metrics and probes of the run the scores came from. A rung is trained here only
when its tuning winner (or its scores) does not exist yet.

**Why `coherent` sits between them.** `temporal` changes two things at once:
each batch covers one band of time (*coherence*), and the bands are visited in
time order, so every epoch ends on the newest rows (*recency* -- SGD weights its
latest updates most). `coherent` has the first without the second, so
`causal -> causal+coherent` measures coherence alone and
`causal+coherent -> causal+temporal` recency alone.

**Cumulative, not factorial.** `uniform+temporal` is deliberately not run, so the
effect of batch order is measured *given* past-only sampling.

**Per-user scores.** Every rung leaves
`results/ablation/per_user/<label>_<dataset>_seed<n>_<rung>.npz`
(`lib/user_scores.py`) for the paired t-tests. A stored rung without that file is
re-run, since it cannot be tested.

All three models are supported, by two routes. NeuMF and LightGCN batch through
`uij_iter` / `uir_iter` and so take batch order from the data loader with no
model-side code. Our NumPy BPR batches itself, so it arrives via
`mecha2/bpr.py:TemporalBPR`, which overrides the one `_epoch_batches` seam.

Usage (normally through `run_ablation.sh`, which sets the environment):
    RESEARCH_TUNED_ARM=per_arm python -m research.mecha2.runner --model neumf --datasets musical --seeds 42
"""

import argparse
import json
import shutil
import time

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib import user_scores
from ..lib.ablation_harness import (
    DATASETS, OUT_DIR, extract_metrics, load_partial, write_partial,
)
from ..lib.causal_sampling import TOP_K
from ..lib.data import build_eval_method
from ..lib.tuning_config import (
    BPR_EARLY_STOP, TUNED_ARM, ablation_label, bpr_kwargs, lightgcn_kwargs,
    neumf_kwargs, winner_dir,
)
from ..paths import logs_dir
from .batching import M2_MODELS, set_arm, use_temporal_batching

#: (rung name, negative sampler, batch order, which arm's winners it uses).
#: Order is the ladder order, and `load_partial` keys cells by these names.
ARMS = (
    ("uniform",         "uniform", "shuffle",  "uniform"),
    ("causal",          "causal",  "shuffle",  "causal"),
    ("causal+coherent", "causal",  "coherent", "causal"),
    ("causal+temporal", "causal",  "temporal", "causal"),
)

#: Rungs identical to a tuning winner: its sampler, shuffled batches, its winners.
FROM_TUNING = ("uniform", "causal")

KWARGS_FOR = {"neumf": neumf_kwargs, "lightgcn": lightgcn_kwargs,
              "bpr": bpr_kwargs}

DEFAULT_DATASETS = ("musical", "baby", "cellphone")

PER_USER_DIR = OUT_DIR / "per_user"


def per_user_path(label, ds_name, seed, rung):
    return PER_USER_DIR / f"{label.lower()}_{ds_name}_seed{seed}_{rung}.npz"


def _winner_epochs(model_name, ds_name, arm, seed):
    """How many epochs that arm's tuning winner trained for, if it is recorded.

    `runners/winner_epochs.py` stamps this from the tuning logs. Absent for
    winners whose log this machine does not have, in which case the caller falls
    back to early stopping.
    """
    wpath = winner_dir(model_name, ds_name, arm, seed) / "winner.json"
    if not wpath.exists():
        return None
    with open(wpath, encoding="utf-8-sig") as f:
        winner = json.load(f)
    epochs = (winner.get("users_run") or {}).get("epochs")
    return int(epochs) if epochs else None


def _build(model_name, ds_name, rung, seed, tuned_as, neg_sampling):
    """Model plus the exact kwargs it was built from, for the provenance stamp."""
    kwargs = KWARGS_FOR[model_name](ds_name, tuned_as, seed)
    if model_name == "neumf":
        from ..lib.cornac_compat import NeuMF
        model = NeuMF(name=f"NeuMF-m2/{ds_name}/{rung}/s{seed}",
                      seed=seed, verbose=True, **kwargs)
    elif model_name == "lightgcn":
        from cornac.models import LightGCN
        model = LightGCN(name=f"LightGCN-m2/{ds_name}/{rung}/s{seed}",
                         seed=seed, verbose=True, **kwargs)
    elif model_name == "bpr":
        # TemporalBPR, not BPRMiniBatch: our BPR batches itself, so it only picks
        # up batch order through the `_epoch_batches` seam. See mecha2/bpr.py.
        from .bpr import TemporalBPR
        # Equal budget, not equal stopping rule. Early stopping quits when the
        # validation curve flattens, and reordering the batches flattens it
        # sooner: on musical/causal/s123 the winner trained 198 epochs and both
        # reordered steps stopped at 41, which loses on training length alone
        # and confounds the one variable this ablation is supposed to isolate.
        # So when the winner's epoch count is known, the trained steps run that
        # many epochs with early stopping off. Without it we cannot equalize, and
        # the old behaviour (each step stops on its own rule) stands.
        epochs = _winner_epochs(model_name, ds_name, tuned_as, seed)
        if epochs and rung not in FROM_TUNING:
            kwargs = {**kwargs, "n_epochs": epochs}
            stopping = {}
        else:
            stopping = BPR_EARLY_STOP
        model = TemporalBPR(name=f"BPR-m2/{ds_name}/{rung}/s{seed}",
                            **kwargs, **stopping,
                            sampler=neg_sampling, seed=seed, verbose=False)
    else:
        raise SystemExit(
            f"--model {model_name} cannot receive temporal batching. "
            f"Supported: {', '.join(M2_MODELS)}.")
    return model, kwargs


def _from_tuning(model_name, ds_name, seed, arm):
    """`(metrics, users_npz)` for a rung tuning already trained, else None.

    Needs a validation-selected winner that carries `users_run` and its
    `winner.users.npz`. The metrics are those of the run the scores came from
    (the winning trial, or its one retrain), so the number in the table and the
    per-user scores behind it always belong together.
    """
    d = winner_dir(model_name, ds_name, arm, seed)
    wpath, upath = d / "winner.json", d / "winner.users.npz"
    if not (wpath.exists() and upath.exists()):
        return None
    with open(wpath, encoding="utf-8-sig") as f:
        winner = json.load(f)
    run = winner.get("users_run")
    if winner.get("select_split") != "validation" or not run:
        return None
    metrics = {k: float(v) for k, v in run["test_metrics"].items()}
    metrics.update({k: float(v) for k, v in (run.get("probes") or {}).items()})
    metrics["from_tuning"] = 1.0
    return metrics, upath


def run_one(model_name: str, label: str, ds_name: str, seed: int) -> None:
    # batch_order is part of the recorded config, so re-running after changing a
    # rung's ordering re-runs that cell instead of silently resuming a stale one.
    configs = {
        rung: {**KWARGS_FOR[model_name](ds_name, tuned_as, seed),
               "batch_order": order, "neg_sampling": neg}
        for rung, neg, order, tuned_as in ARMS
    }
    # The equal-budget epoch count belongs in the recorded config too, or a cell
    # trained under early stopping would look cached to `load_partial` and the
    # two protocols would mix inside one table row.
    if model_name == "bpr":
        for rung, neg, order, tuned_as in ARMS:
            epochs = _winner_epochs(model_name, ds_name, tuned_as, seed)
            if epochs and rung not in FROM_TUNING:
                configs[rung]["n_epochs"] = epochs

    recipes_out = load_partial(label, ds_name, seed, configs)
    for rung in list(recipes_out):
        if not per_user_path(label, ds_name, seed, rung).exists():
            print(f"[stale] {rung}: no per-user scores, so it cannot be tested; "
                  f"re-running", flush=True)
            del recipes_out[rung]
    pending = [r for r, *_ in ARMS if r not in recipes_out]
    if not pending:
        print(f"[skip] {label} {ds_name} seed={seed} all cells cached", flush=True)
        return

    print(f"\n############## {label} / {ds_name} / seed={seed} "
          f"({len(pending)}/{len(ARMS)} cells pending) ##############", flush=True)
    t_start = time.time()
    eval_method = train_set = None   # built only if something needs training

    for rung, neg, order, tuned_as in ARMS:
        if rung not in pending:
            continue
        target = per_user_path(label, ds_name, seed, rung)

        if rung in FROM_TUNING:
            got = _from_tuning(model_name, ds_name, seed, tuned_as)
            if got is not None:
                metrics, upath = got
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(upath, target)
                recipes_out[rung] = metrics
                write_partial(label, ds_name, seed, recipes_out, configs)
                print(f"[{rung}] from tuning winner {upath.parent}  {metrics}  "
                      f"[checkpointed]", flush=True)
                continue
            print(f"[{rung}] no tuning winner with per-user scores for seed "
                  f"{seed}; training it here", flush=True)

        if eval_method is None:
            eval_method = build_eval_method(ds_name, seed=seed, verbose=True)
            # Upgrade once; the rungs then differ only by attributes on this split.
            train_set = use_temporal_batching(eval_method, batch_order="shuffle")

        print(f"\n--- {rung} (neg={neg}, batches={order}) ---", flush=True)
        t0 = time.time()
        set_arm(train_set, neg, order)
        model, _kwargs = _build(model_name, ds_name, rung, seed, tuned_as, neg)
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        )
        exp.run()
        # Our BPR owns its sampler, so its probes live on the model. cornac's
        # models draw through the split and have no idea anything changed.
        metrics = extract_metrics(
            exp, probe=model if model_name == "bpr" else train_set)
        metrics["from_tuning"] = 0.0
        user_scores.save(target, user_scores.from_experiment(exp))
        recipes_out[rung] = metrics
        write_partial(label, ds_name, seed, recipes_out, configs)
        print(f"[{rung}] {metrics}  ({time.time()-t0:.1f}s)  [checkpointed]", flush=True)

    print(f"#### {label}/{ds_name}/seed={seed} total: "
          f"{(time.time()-t_start)/60:.1f} min ####", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="neumf", choices=sorted(M2_MODELS),
                   help="neumf/lightgcn take batch order through the data "
                        "loader; bpr through the TemporalBPR subclass.")
    p.add_argument("--seeds", default="42")
    p.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    args = p.parse_args()

    if TUNED_ARM != "per_arm":
        raise SystemExit(
            "This ladder runs rung 1 at the uniform winners and rungs 2-4 at the "
            "causal winners, reusing both tuning winners for rungs 1-2. Run it "
            "with RESEARCH_TUNED_ARM=per_arm (run_ablation.sh sets it).")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            raise SystemExit(f"unknown dataset key: {d} (known: {list(DATASETS)})")

    # `-m2` keeps these cells from colliding with the older two-arm ablation's,
    # and `ablation_label` layers the variant switches on top.
    base = {"neumf": "NeuMF", "lightgcn": "LightGCN", "bpr": "BPR"}[args.model]
    label = ablation_label(f"{base}-m2")
    for ds_name in datasets:
        for seed in seeds:
            run_one(args.model, label, ds_name, seed)


if __name__ == "__main__":
    main()
