"""GPU-accelerated BPR with the same API as bpr_mini_batch.py.

Used for the healthcare prequential runs where the numpy version is
too slow (~8h per cell). Drop-in compatible: same constructor kwargs,
same `fit/score/recommend` surface, same `counterfactual_rate`
property. Numbers should match the numpy version up to floating-point
noise.

GPU specifics:
- Float32 throughout.
- np.add.at replaced by torch.index_add_ which is properly vectorized
  on GPU.
- Negative sampling done on GPU via torch.searchsorted + uniform random.
"""

from typing import Literal, Optional

import numpy as np
import torch

from cornac.models.recommender import Recommender

from .causal_negative_sampler import TorchCausalSampler


SamplerKind = Literal["uniform", "causal"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPRMiniBatchGPU(Recommender):
    def __init__(
        self,
        name: str = "BPR-GPU",
        k: int = 64,
        n_epochs: int = 20,
        batch_size: int = 16384,
        learning_rate: float = 0.05,
        lambda_u: float = 1e-4,
        lambda_i: float = 1e-4,
        lambda_j: float = 1e-4,
        use_bias: bool = False,
        sampler: SamplerKind = "uniform",
        # NewBPR §5.2 training protocol: patience=13 (in early_stop calls).
        # When `early_stopping` is None the loop runs the full `n_epochs`.
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
        self.u_factors = None
        self.i_factors = None
        self.i_biases = None
        # Negative sampler (uniform/causal) built per-fit; see
        # causal_negative_sampler.TorchCausalSampler.
        self._sampler = None
        self._gen = None  # torch.Generator

    @property
    def counterfactual_rate(self) -> float:
        return self._sampler.counterfactual_rate if self._sampler is not None else 0.0

    def _init_params(self, n_users, n_items):
        scale = 1.0 / self.k
        self._gen = torch.Generator(device="cpu").manual_seed(self.seed)
        u = (torch.rand((n_users, self.k), generator=self._gen) - 0.5) * scale
        i = (torch.rand((n_items, self.k), generator=self._gen) - 0.5) * scale
        self.u_factors = u.to(DEVICE).float().contiguous()
        self.i_factors = i.to(DEVICE).float().contiguous()
        self.i_biases = torch.zeros(n_items, device=DEVICE, dtype=torch.float32)

    def _sgd_step(self, u, pos, neg):
        u_emb = self.u_factors[u]
        pos_emb = self.i_factors[pos]
        neg_emb = self.i_factors[neg]
        score_diff = (u_emb * (pos_emb - neg_emb)).sum(dim=-1)
        if self.use_bias:
            score_diff = score_diff + self.i_biases[pos] - self.i_biases[neg]
        z = torch.sigmoid(-score_diff)  # gradient coefficient

        z_col = z.unsqueeze(-1)
        # NewBPR §5.3 separated λs.
        grad_u = z_col * (pos_emb - neg_emb) - self.reg_u * u_emb
        grad_pos = z_col * u_emb - self.reg_i * pos_emb
        grad_neg = -z_col * u_emb - self.reg_j * neg_emb

        # index_add_ handles duplicate indices correctly (sums them).
        self.u_factors.index_add_(0, u, self.lr * grad_u)
        self.i_factors.index_add_(0, pos, self.lr * grad_pos)
        self.i_factors.index_add_(0, neg, self.lr * grad_neg)
        if self.use_bias:
            self.i_biases.index_add_(0, pos, self.lr * (z - self.reg_i * self.i_biases[pos]))
            self.i_biases.index_add_(0, neg, self.lr * (-z - self.reg_j * self.i_biases[neg]))

    def _reset_early_stop(self):
        """Reset cornac's early-stop counters so a fresh training segment
        (the shuffle run, or each windowed window) gets its own budget."""
        self.current_epoch = 0
        self.best_value = float("-inf")
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def monitor_value(self, train_set, val_set):
        """NDCG@20 on val_set. Matches our tuner selection metric and the
        reported tables. NewBPR §4.1.6 optimizes NDCG@100 — we use @20 for
        consistency with what `aggregate_2x2.py` reports."""
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

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        n_users = train_set.num_users
        n_items = train_set.num_items
        if self.verbose:
            print(f"[{self.name}] fit start (GPU): users={n_users} items={n_items} rows={train_set.num_ratings}", flush=True)
        self._init_params(n_users, n_items)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        u_arr_np = train_set.uir_tuple[0].astype(np.int64)
        i_arr_np = train_set.uir_tuple[1].astype(np.int64)
        ts_np = np.asarray(train_set.timestamps, dtype=np.int64)

        u_arr = torch.from_numpy(u_arr_np).to(DEVICE)
        i_arr = torch.from_numpy(i_arr_np).to(DEVICE)
        ts_arr = torch.from_numpy(ts_np).to(DEVICE)

        # Causal/uniform negative sampler — reads per-item first-seen from the
        # (TimeAware) training set; on-device draws for speed.
        self._sampler = TorchCausalSampler(train_set, self.sampler, DEVICE)

        gen = torch.Generator(device=DEVICE).manual_seed(self.seed)
        n_samples = len(u_arr)
        self._reset_early_stop()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_samples, generator=gen, device=DEVICE)
            for start in range(0, n_samples, self.batch_size):
                idx = perm[start:start + self.batch_size]
                u = u_arr[idx]
                pos = i_arr[idx]
                ts = ts_arr[idx]
                neg = self._sampler.sample(ts)
                self._sgd_step(u, pos, neg)
            if self.verbose:
                print(f"[{self.name}] epoch {epoch+1}/{self.n_epochs}", flush=True)
            if (self.early_stopping is not None and val_set is not None
                    and (epoch + 1) % self.early_stop_every == 0):
                if self.early_stop(train_set, val_set, **self.early_stopping):
                    if self.verbose:
                        print(f"[{self.name}] early-stop at epoch {epoch+1} "
                              f"(best={self.best_value:.4f} @ epoch {self.best_epoch})",
                              flush=True)
                    break
        return self

    @torch.no_grad()
    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            scores = self.i_biases + (self.i_factors @ self.u_factors[user_idx])
            return scores.detach().cpu().numpy()
        return float(self.i_biases[item_idx] + (self.i_factors[item_idx] * self.u_factors[user_idx]).sum())


class BPRWindowedGPU(BPRMiniBatchGPU):
    """Same as BPRWindowed but GPU. Windowed training, warm-start per window."""

    def __init__(self, *args, n_windows: int = 10, epochs_per_window: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_windows = n_windows
        self.epochs_per_window = epochs_per_window

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        n_users = train_set.num_users
        n_items = train_set.num_items
        if self.verbose:
            print(f"[{self.name}] fit start (GPU windowed): K={self.n_windows} epw={self.epochs_per_window}", flush=True)
        self._init_params(n_users, n_items)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        u_arr_np = train_set.uir_tuple[0].astype(np.int64)
        i_arr_np = train_set.uir_tuple[1].astype(np.int64)
        ts_np = np.asarray(train_set.timestamps, dtype=np.int64)

        u_arr = torch.from_numpy(u_arr_np).to(DEVICE)
        i_arr = torch.from_numpy(i_arr_np).to(DEVICE)
        ts_arr = torch.from_numpy(ts_np).to(DEVICE)

        self._sampler = TorchCausalSampler(train_set, self.sampler, DEVICE)

        # Chronological window partition.
        order = torch.argsort(ts_arr, stable=True)
        u_sorted = u_arr[order]
        i_sorted = i_arr[order]
        ts_sorted = ts_arr[order]
        boundaries = np.linspace(0, len(u_sorted), self.n_windows + 1, dtype=np.int64)

        gen = torch.Generator(device=DEVICE).manual_seed(self.seed)
        for w in range(self.n_windows):
            lo, hi = boundaries[w], boundaries[w + 1]
            if hi <= lo:
                continue
            u_w = u_sorted[lo:hi]
            i_w = i_sorted[lo:hi]
            ts_w = ts_sorted[lo:hi]
            self._reset_early_stop()
            for ep in range(self.epochs_per_window):
                perm = torch.randperm(len(u_w), generator=gen, device=DEVICE)
                for start in range(0, len(u_w), self.batch_size):
                    idx = perm[start:start + self.batch_size]
                    u = u_w[idx]
                    pos = i_w[idx]
                    ts = ts_w[idx]
                    neg = self._sampler.sample(ts)
                    self._sgd_step(u, pos, neg)
                if (self.early_stopping is not None and val_set is not None
                        and (ep + 1) % self.early_stop_every == 0):
                    if self.early_stop(train_set, val_set, **self.early_stopping):
                        if self.verbose:
                            print(f"[{self.name}] early-stop window {w+1} at epoch {ep+1}",
                                  flush=True)
                        break
            if self.verbose:
                print(f"[{self.name}] window {w+1}/{self.n_windows}  rows={int(hi-lo)}", flush=True)
        return self
