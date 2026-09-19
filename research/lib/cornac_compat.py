from cornac.models import GMF, MLP
from cornac.models import NeuMF as _CornacNeuMF
from cornac.models.recommender import Recommender


class NeuMF(_CornacNeuMF):
    def __init__(self, *args, pretrain=False, pretrain_alpha=0.5,
                 pretrain_learner="adam", **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrain = pretrain
        self.pretrain_alpha = pretrain_alpha
        self.pretrain_learner = pretrain_learner

    def _tower_kwargs(self):
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
