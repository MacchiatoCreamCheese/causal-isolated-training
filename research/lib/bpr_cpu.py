"""Pure-NumPy mini-batch BPR with a swappable negative sampler.

Why a fresh implementation: cornac's BPR is Cython, so toggling the
negative-sampling rule means recompiling. For research iteration speed
we want one Python codebase where vanilla vs leakage-aware is a single
flag, so any accuracy/leakage delta is attributable to the sampler, not
the optimizer.

Algorithm: mini-batch SGD on the BPR pairwise log-likelihood, identical
gradients to Rendle 2009, just batched. The "sampler" parameter chooses
which items are eligible to be negatives:

  - "uniform":      any item in the catalog (vanilla BPR).
  - "causal":       only items whose first appearance in the global
                    dataset is at-or-before the positive's timestamp.
                    Implementation: items pre-sorted by first_seen;
                    binary-search the prefix length for each positive's
                    timestamp; sample uniformly from that prefix.
"""

from typing import Literal, Optional

import numpy as np

from cornac.models.recommender import Recommender

from .causal_negative_sampler import NumpyCausalSampler


SamplerKind = Literal["uniform", "causal"]


class BPRMiniBatch(Recommender):
    def __init__(
        self,
        name: str = "BPR-MB",
        k: int = 64,
        n_epochs: int = 20,
        batch_size: int = 4096,
        learning_rate: float = 0.05,
        lambda_u: float = 1e-4,
        lambda_i: float = 1e-4,
        lambda_j: float = 1e-4,
        use_bias: bool = False,
        sampler: SamplerKind = "uniform",
        # NewBPR §5.2 training protocol: patience=13.
        early_stopping: Optional[dict] = None,
        early_stop_every: int = 1,
        seed: int = 42,
        verbose: bool = True,
        trainable: bool = True,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.k = k
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        # NewBPR §5.3 (global temporal split): separated regularization on
        # user, positive-item, and negative-item embeddings.
        self.reg_u = lambda_u
        self.reg_i = lambda_i
        self.reg_j = lambda_j
        self.use_bias = use_bias
        self.sampler = sampler
        self.early_stopping = early_stopping
        self.early_stop_every = early_stop_every
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.u_factors = None
        self.i_factors = None
        self.i_biases = None
        # Negative sampler (uniform/causal) built per-fit; see
        # causal_negative_sampler.NumpyCausalSampler.
        self._sampler = None

    def _init_params(self, n_users, n_items):
        scale = 1.0 / self.k
        self.u_factors = (self.rng.random((n_users, self.k), dtype=np.float32) - 0.5) * scale
        self.i_factors = (self.rng.random((n_items, self.k), dtype=np.float32) - 0.5) * scale
        self.i_biases = np.zeros(n_items, dtype=np.float32)

    @property
    def counterfactual_rate(self) -> float:
        """Fraction of sampled negatives that violated the causality constraint
        (item didn't exist at the time of the positive). Available after fit()."""
        return self._sampler.counterfactual_rate if self._sampler is not None else 0.0

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        n_users = train_set.num_users
        n_items = train_set.num_items
        if self.verbose:
            print(f"[{self.name}] fit start: users={n_users} items={n_items} rows={train_set.num_ratings}", flush=True)
        self._init_params(n_users, n_items)

        u_arr, i_arr, _ = train_set.uir_tuple
        u_arr = u_arr.astype(np.int64)
        i_arr = i_arr.astype(np.int64)
        ts_arr = np.asarray(train_set.timestamps, dtype=np.int64)

        # Causal/uniform negative sampler — reads per-item first-seen from the
        # (TimeAware) training set. Uses the model's own rng for reproducibility.
        self._sampler = NumpyCausalSampler(train_set, self.sampler, self.rng)

        n_samples = len(u_arr)
        self._reset_early_stop()
        for epoch in range(self.n_epochs):
            perm = self.rng.permutation(n_samples)
            running_correct = 0
            n_seen = 0
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u = u_arr[idx]
                pos = i_arr[idx]
                ts = ts_arr[idx]
                neg = self._sampler.sample(ts)

                # Compute scores
                u_emb = self.u_factors[u]            # (B, k)
                pos_emb = self.i_factors[pos]        # (B, k)
                neg_emb = self.i_factors[neg]        # (B, k)
                pos_score = np.einsum("bk,bk->b", u_emb, pos_emb)
                neg_score = np.einsum("bk,bk->b", u_emb, neg_emb)
                if self.use_bias:
                    pos_score = pos_score + self.i_biases[pos]
                    neg_score = neg_score + self.i_biases[neg]
                diff = pos_score - neg_score
                # sigmoid(-diff) is the gradient coefficient
                z = 1.0 / (1.0 + np.exp(diff))  # shape (B,)
                running_correct += int((z < 0.5).sum())
                n_seen += len(u)

                # Gradients (Rendle 2009) with NewBPR §5.3 separated λs.
                z_col = z[:, None].astype(np.float32)
                grad_u = z_col * (pos_emb - neg_emb) - self.reg_u * u_emb
                grad_pos = z_col * u_emb - self.reg_i * pos_emb
                grad_neg = -z_col * u_emb - self.reg_j * neg_emb

                # Apply updates via np.add.at to handle repeat indices in a batch.
                np.add.at(self.u_factors, u, self.lr * grad_u)
                np.add.at(self.i_factors, pos, self.lr * grad_pos)
                np.add.at(self.i_factors, neg, self.lr * grad_neg)
                if self.use_bias:
                    np.add.at(self.i_biases, pos, self.lr * (z - self.reg_i * self.i_biases[pos]))
                    np.add.at(self.i_biases, neg, self.lr * (-z - self.reg_j * self.i_biases[neg]))

            if self.verbose:
                pct = 100.0 * running_correct / max(n_seen, 1)
                print(f"[{self.name}] epoch {epoch+1}/{self.n_epochs}  correct={pct:.2f}%", flush=True)
            if (self.early_stopping is not None and val_set is not None
                    and (epoch + 1) % self.early_stop_every == 0):
                if self.early_stop(train_set, val_set, **self.early_stopping):
                    if self.verbose:
                        print(f"[{self.name}] early-stop at epoch {epoch+1} "
                              f"(best={self.best_value:.4f} @ epoch {self.best_epoch})",
                              flush=True)
                    break
        return self

    def _reset_early_stop(self):
        self.current_epoch = 0
        self.best_value = float("-inf")
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def monitor_value(self, train_set, val_set):
        """NDCG@20 on val_set — matches our tuner selection metric and the
        reported tables. NewBPR §4.1.6 optimizes NDCG@100; we use @20 for
        consistency with what `aggregate_ablation.py` reports."""
        if val_set is None:
            return None
        from cornac.metrics import NDCG
        from cornac.eval_methods.base_method import ranking_eval
        ndcg_20 = ranking_eval(
            model=self,
            metrics=[NDCG(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]
        return ndcg_20

    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return self.i_biases + self.i_factors @ self.u_factors[user_idx]
        return float(self.i_biases[item_idx] + self.i_factors[item_idx] @ self.u_factors[user_idx])
