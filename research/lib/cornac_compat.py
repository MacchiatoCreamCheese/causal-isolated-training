"""Shims for cornac behaviour the runners cannot use as-is.

Two things live here, and they are different in kind:

  - `NeuMF.save`, a pure compatibility fix for cornac code that raises under the
    configuration this project needs. Changes nothing a model learns.
  - `NeuMF.fit`'s optional pre-training, which *does* change what the model
    learns -- but only when explicitly enabled, and it is cornac's own
    `from_pretrained` protocol rather than anything invented here.

Anything that would alter *negative sampling* still belongs in
`timeaware_data.py`, where the ablation can see it.
"""

from cornac.models import GMF, MLP
from cornac.models import NeuMF as _CornacNeuMF
from cornac.models.recommender import Recommender


class NeuMF(_CornacNeuMF):
    """cornac's NeuMF, saved like every other cornac model, and optionally
    pre-trained the way He 2017 §3.4.1 describes.

    Same name as the class it subclasses, deliberately: callers swap
    `from cornac.models import NeuMF` for `from ..lib.cornac_compat import NeuMF`
    and nothing else changes. With `pretrain=False` (the default) this *is*
    cornac's NeuMF for every purpose the ablation touches, and the import path is
    where the difference belongs.

    Persistence
    -----------

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
    taking the metrics down with it before the runner can checkpoint them.

    So `save` skips exactly the raise and nothing else, going straight to
    `Recommender.save` for the same `.pkl` + `.meta` any other cornac model
    produces. The pickle holds the torch module and round-trips, so nothing is
    lost by not writing the sidecar.

    Pre-training (`pretrain=True`)
    ------------------------------

    He 2017 §4.2.1 + Table 2 report NeuMF-with-pretrain beating NeuMF-from-
    scratch on both datasets, and cornac supports it -- but only as an opt-in
    method, `from_pretrained(gmf, mlp, alpha)`, which needs two already-fitted
    models handed to it. `NeuMF.__init__` has no `pretrain` argument at all and
    a fresh model reports `pretrained == False`, so a plain `NeuMF(...)` is the
    from-scratch variant no matter what a config file claims.

    This override closes that gap: with `pretrain=True` it fits a GMF and an MLP
    on the same training split, fuses them, and only then runs cornac's own fit.
    It lives in `fit()` rather than in the runners because `Experiment.run()`
    calls `model.fit(train_set, val_set)` and both call sites
    (`runners/ablation_neumf.py`, `runners/tuning.py:_build_model`) construct the
    model long before a training split exists -- so doing it here means neither
    call site changes, and the tuner and the ablation stay on one code path.

    **Default is off.** Pre-training triples the training cost of a cell, and
    turning it on invalidates any hyperparameters tuned without it. Enable via
    `RESEARCH_NEUMF_PRETRAIN=1`, which `tuning_config.NEUMF_KWARGS` reads --
    that switch also moves `learner` to SGD, which matters (see below).

    Four things this has to get right, none of them obvious:

      - **`self.alpha` is set *by* `from_pretrained`.** The requested value
        therefore cannot be stored there before the call, hence
        `pretrain_alpha`.
      - **The towers must be fitted before `super().fit()`.** `from_pretrained`
        only records references; the backend copies weights out of
        `pretrained_gmf.model` / `pretrained_mlp.model` when it builds the fused
        module, so `.model` has to exist by then.
      - **Shapes must line up.** GMF's `num_factors` and MLP's `layers` are taken
        from this instance for exactly that reason; a mismatch fails at module
        construction, not at config time.
      - **cornac does not switch optimisers when pretrained.** `_fit_pt` always
        uses `get_optimizer(learning_rate=self.lr, learner=self.learner)`, so
        He 2017 §3.4.1's "vanilla SGD, rather than Adam" for the fine-tune has to
        be passed in by the caller. The towers keep Adam, per the same section.

    Negative sampling needs no special handling: `NCFBase.fit` draws through
    `train_set.uir_iter(...)` and GMF/MLP subclass it, so both pre-training
    phases sample through `TimeAwareDataset` exactly as NeuMF does. One
    consequence worth knowing -- the probe counters are reset once per cell by
    `ablation_harness.set_recipe`, so with pre-training on, rho and
    `collision_rate` span all three fits rather than NeuMF's alone.

    GMF and MLP inherit the same broken `NCFBase.save`, but it is never reached:
    they are fitted directly here, not routed through an `Experiment`.
    """

    def __init__(self, *args, pretrain=False, pretrain_alpha=0.5,
                 pretrain_learner="adam", **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain = pretrain
        # Deliberately not `self.alpha`: `from_pretrained` owns that name.
        self.pretrain_alpha = pretrain_alpha
        self.pretrain_learner = pretrain_learner

    def _tower_kwargs(self):
        """Settings both pre-training towers inherit from this instance.

        Shared rather than re-specified so the fused module's shapes cannot drift
        from the towers', and so a tuned `batch_size` / `num_neg` / `lr` applies
        to every phase rather than only the last one.
        """
        return dict(
            num_epochs=self.num_epochs,
            batch_size=self.batch_size,
            num_neg=self.num_neg,
            lr=self.lr,
            reg=self.reg,
            learner=self.pretrain_learner,
            backend=self.backend,
            early_stopping=self.early_stopping,
            seed=self.seed,
            verbose=self.verbose,
        )

    def fit(self, train_set, val_set=None):
        # `not self.pretrained` so an explicit from_pretrained() by a caller is
        # respected rather than silently rebuilt.
        if self.pretrain and not self.pretrained:
            if self.verbose:
                print(f"[{self.name}] pre-training GMF...", flush=True)
            gmf = GMF(name=f"{self.name}/gmf", num_factors=self.num_factors,
                      **self._tower_kwargs())
            gmf.fit(train_set, val_set)

            if self.verbose:
                print(f"[{self.name}] pre-training MLP...", flush=True)
            mlp = MLP(name=f"{self.name}/mlp", layers=self.layers,
                      act_fn=self.act_fn, **self._tower_kwargs())
            mlp.fit(train_set, val_set)

            self.from_pretrained(gmf, mlp, alpha=self.pretrain_alpha)
            if self.verbose:
                print(f"[{self.name}] fusing towers (alpha={self.pretrain_alpha}) "
                      f"and fine-tuning with learner={self.learner}", flush=True)
        return super().fit(train_set, val_set)

    def save(self, save_dir=None, *args, **kwargs):
        if save_dir is None:
            return None
        return Recommender.save(self, save_dir, *args, **kwargs)
