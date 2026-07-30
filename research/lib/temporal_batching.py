"""Phase 3: temporal batching.

Sort training rows by timestamp, partition into K windows, train
sequentially window-by-window with warm-start (no parameter reset).
Compare against vanilla (random-shuffled) BPR.

Cross with Phase 2's causal-negative sampler so all four cells of the
2x2 design are filled:

  shuffle  + uniform_negatives   = pure baseline (= Phase 2 BPR-uniform)
  shuffle  + causal_negatives    = Phase 2 BPR-causal
  windowed + uniform_negatives   = order matters, sampler doesn't
  windowed + causal_negatives    = full causal training

Ablate K in {2, 5, 10, 20}. Total training rows are constant across
runs; what changes is the order and within-window negative pool.
"""

import numpy as np

import cornac

from .bpr_cpu import BPRMiniBatch
from .causal_negative_sampler import NumpyCausalSampler


class BPRWindowed(BPRMiniBatch):
    """BPR trained sequentially through K time windows.

    Per window, we run `epochs_per_window` epochs of mini-batch SGD using
    ONLY the rows in that window. After the last window, we have a model
    that has seen every train row in chronological order, with no
    reshuffling across windows.
    """

    def __init__(self, *args, n_windows: int = 10, epochs_per_window: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_windows = n_windows
        self.epochs_per_window = epochs_per_window
        # Override the parent's n_epochs so the budget is comparable:
        self.n_epochs = epochs_per_window  # one window's worth — used inside fit-loop

    def fit(self, train_set, val_set=None):
        # Set up like the parent's fit, but iterate over windows.
        cornac.models.recommender.Recommender.fit(self, train_set, val_set)
        n_users = train_set.num_users
        n_items = train_set.num_items
        self._init_params(n_users, n_items)

        u_arr, i_arr, _ = train_set.uir_tuple
        u_arr = u_arr.astype(np.int64)
        i_arr = i_arr.astype(np.int64)
        ts_arr = np.asarray(train_set.timestamps, dtype=np.int64)

        # Causal/uniform negative sampler — reads per-item first-seen from the
        # (TimeAware) training set. Uses the model's own rng for reproducibility.
        self._sampler = NumpyCausalSampler(train_set, self.sampler, self.rng)

        # Equal-width windows on timestamp value.
        order = np.argsort(ts_arr, kind="stable")
        u_sorted = u_arr[order]
        i_sorted = i_arr[order]
        ts_sorted = ts_arr[order]
        boundaries = np.linspace(0, len(u_sorted), self.n_windows + 1, dtype=np.int64)

        for w in range(self.n_windows):
            lo, hi = boundaries[w], boundaries[w + 1]
            if hi <= lo:
                continue
            u_w = u_sorted[lo:hi]
            i_w = i_sorted[lo:hi]
            ts_w = ts_sorted[lo:hi]
            n_samples = len(u_w)
            self._reset_early_stop()
            for epoch in range(self.epochs_per_window):
                perm = self.rng.permutation(n_samples)
                for start in range(0, n_samples, self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    u = u_w[idx]
                    pos = i_w[idx]
                    ts = ts_w[idx]
                    neg = self._sampler.sample(ts)
                    self._sgd_step(u, pos, neg)
                if (self.early_stopping is not None and val_set is not None
                        and (epoch + 1) % self.early_stop_every == 0):
                    if self.early_stop(train_set, val_set, **self.early_stopping):
                        if self.verbose:
                            print(f"[{self.name}] early-stop window {w+1} at epoch {epoch+1}",
                                  flush=True)
                        break
            if self.verbose:
                print(f"[{self.name}] window {w+1}/{self.n_windows} done  rows={n_samples}")
        return self

    def _sgd_step(self, u, pos, neg):
        u_emb = self.u_factors[u]
        pos_emb = self.i_factors[pos]
        neg_emb = self.i_factors[neg]
        pos_score = np.einsum("bk,bk->b", u_emb, pos_emb)
        neg_score = np.einsum("bk,bk->b", u_emb, neg_emb)
        if self.use_bias:
            pos_score = pos_score + self.i_biases[pos]
            neg_score = neg_score + self.i_biases[neg]
        z = 1.0 / (1.0 + np.exp(pos_score - neg_score))
        z_col = z[:, None].astype(np.float32)
        # NewBPR §5.3 separated λs (inherited from BPRMiniBatch).
        grad_u = z_col * (pos_emb - neg_emb) - self.reg_u * u_emb
        grad_pos = z_col * u_emb - self.reg_i * pos_emb
        grad_neg = -z_col * u_emb - self.reg_j * neg_emb
        np.add.at(self.u_factors, u, self.lr * grad_u)
        np.add.at(self.i_factors, pos, self.lr * grad_pos)
        np.add.at(self.i_factors, neg, self.lr * grad_neg)
        if self.use_bias:
            np.add.at(self.i_biases, pos, self.lr * (z - self.reg_i * self.i_biases[pos]))
            np.add.at(self.i_biases, neg, self.lr * (-z - self.reg_j * self.i_biases[neg]))
