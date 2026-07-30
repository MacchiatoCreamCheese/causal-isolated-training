"""Phase 6: does our training-side faithfulness recipe transfer to NeuMF?

This is the FAITHFUL NeuMF — two-tower (GMF + MLP) fused at the last layer,
with the paper's three-phase training:
  1. Pre-train GMF separately (Adam, BCE on positives + sampled negatives).
  2. Pre-train MLP separately (Adam, BCE on positives + sampled negatives).
  3. Init NeuMF from GMF and MLP with mixing alpha=0.5, fine-tune with
     vanilla SGD (paper Section 3.4.1).

The nn.Module definitions (`GMF`, `MLP`, `NeuMF`) are imported directly from
cornac's PyTorch backend `cornac/models/ncf/backend_pt.py`, so the
architecture is bit-identical to cornac's published NeuMF. We only add our
two research mechanisms around the training loop:
  - causal negative sampling (`sampler="causal"`)
  - temporal batching with K chronological windows (`order="windowed"`)
Each of the three training phases respects the cell's order/sampler
config, so a `win10+causal` cell pre-trains GMF and MLP under win10+causal
as well, and a `shuffle+uniform` cell pre-trains everything shuffle+uniform.

GPU-accelerated.
"""

import time

import numpy as np
import torch
import torch.nn as nn

from cornac.models.recommender import Recommender
from cornac.models.ncf.backend_pt import GMF, MLP, NeuMF as NeuMFNet

from .causal_negative_sampler import TorchCausalSampler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuMFRecommender(Recommender):
    """Faithful NeuMF (GMF + MLP fusion + pre-training) with the two research
    mechanisms applied to all three training phases."""

    def __init__(
        self,
        name: str = "NeuMF",
        num_factors: int = 8,                          # paper default (§4.1 grid: {8,16,32,64})
        layers: tuple = (64, 32, 16, 8),               # paper default tower (§3.3 halving)
        act_fn: str = "relu",                          # paper §3.3
        n_epochs: int = 20,                            # paper §4.1
        batch_size: int = 256,                         # paper §4.1 grid: {128,256,512,1024}
        learning_rate: float = 1e-3,                   # paper §4.1 grid
        num_neg: int = 4,                              # paper §4.3 optimum 3-6
        alpha: float = 0.5,                            # paper §4.1 NeuMF mixing
        order: str = "shuffle",                        # "shuffle" | "windowed"
        sampler: str = "uniform",                      # "uniform" | "causal"
        n_windows: int = 10,
        seed: int = 42,
        verbose: bool = True,
        trainable: bool = True,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        assert layers[-1] == num_factors, (
            f"NeuMF requires layers[-1] == num_factors (got layers[-1]={layers[-1]}, "
            f"num_factors={num_factors}); paper §3.4 concat-then-project requires equal dims.")
        self.num_factors = num_factors
        self.layers = tuple(layers)
        self.act_fn = act_fn
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.num_neg = num_neg
        self.alpha = alpha
        self.order = order
        self.sampler = sampler
        self.n_windows = n_windows
        self.seed = seed

        # Final trained NeuMF module, used for scoring.
        self.net = None

        # Negative sampler (uniform/causal) built per-fit; see
        # causal_negative_sampler.TorchCausalSampler.
        self._sampler = None

    @property
    def counterfactual_rate(self) -> float:
        return self._sampler.counterfactual_rate if self._sampler is not None else 0.0

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    # ------------------------------------------------------------------
    # Generic training pass over an nn.Module with BCE.
    # ------------------------------------------------------------------
    def _train_passes(self, net, u_pos, i_pos, ts_pos, opt, criterion):
        """One pass through (u_pos, i_pos), generating num_neg negatives per
        positive on-the-fly, computing BCE on sigmoid outputs."""
        n = len(u_pos)
        loss_sum = 0.0
        for start in range(0, n, self.batch_size):
            ub = u_pos[start:start + self.batch_size]
            ib = i_pos[start:start + self.batch_size]
            tb = ts_pos[start:start + self.batch_size]
            b = len(ub)
            negs = self._sampler.sample(tb, self.num_neg)                    # (b, num_neg)
            u_rep = ub.unsqueeze(1).expand(-1, self.num_neg + 1).reshape(-1)
            i_all = torch.cat([ib.unsqueeze(1), negs], dim=1).reshape(-1)
            labels = torch.zeros(b, self.num_neg + 1, device=DEVICE)
            labels[:, 0] = 1.0
            labels = labels.reshape(-1)

            preds = net(u_rep, i_all)                                        # sigmoid output
            loss = criterion(preds, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * b
        return loss_sum / max(n, 1)

    def _train_module(self, net, optimizer_name, u_arr, i_arr, ts_arr, ts_np, label):
        """Train `net` for self.n_epochs under our (order, sampler) config.

        optimizer_name: "adam" (for GMF and MLP pre-training, paper §3.4.1)
                        or "sgd"  (for fused NeuMF fine-tuning, paper §3.4.1).
        """
        rng = np.random.default_rng(self.seed)
        criterion = nn.BCELoss()
        if optimizer_name == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        elif optimizer_name == "sgd":
            opt = torch.optim.SGD(net.parameters(), lr=self.lr)
        else:
            raise ValueError(f"unknown optimizer {optimizer_name}")

        t0 = time.time()
        if self.order == "shuffle":
            for epoch in range(self.n_epochs):
                perm = torch.from_numpy(rng.permutation(len(u_arr))).to(DEVICE)
                loss = self._train_passes(net, u_arr[perm], i_arr[perm],
                                          ts_arr[perm], opt, criterion)
                if self.verbose:
                    print(f"[{self.name}/{label}] epoch {epoch+1}/{self.n_epochs}  "
                          f"loss={loss:.4f}  t={time.time()-t0:.1f}s", flush=True)
        else:  # windowed
            order_idx = np.argsort(ts_np, kind="stable")
            order_t = torch.from_numpy(order_idx).to(DEVICE)
            u_sorted = u_arr[order_t]
            i_sorted = i_arr[order_t]
            ts_sorted = ts_arr[order_t]
            boundaries = np.linspace(0, len(u_sorted), self.n_windows + 1, dtype=int)
            for w in range(self.n_windows):
                lo, hi = boundaries[w], boundaries[w + 1]
                u_w = u_sorted[lo:hi]
                i_w = i_sorted[lo:hi]
                ts_w = ts_sorted[lo:hi]
                for ep in range(self.n_epochs):
                    perm = torch.from_numpy(rng.permutation(len(u_w))).to(DEVICE)
                    loss = self._train_passes(net, u_w[perm], i_w[perm],
                                              ts_w[perm], opt, criterion)
                if self.verbose:
                    print(f"[{self.name}/{label}] window {w+1}/{self.n_windows}  "
                          f"last_loss={loss:.4f}  t={time.time()-t0:.1f}s", flush=True)
        return net

    # ------------------------------------------------------------------
    # fit(): three-phase NeuMF training (paper §3.4.1).
    # ------------------------------------------------------------------
    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        self._set_seeds()

        n_users = train_set.num_users
        n_items = train_set.num_items

        u_arr = torch.from_numpy(train_set.uir_tuple[0].astype(np.int64)).to(DEVICE)
        i_arr = torch.from_numpy(train_set.uir_tuple[1].astype(np.int64)).to(DEVICE)
        ts_np = np.asarray(train_set.timestamps, dtype=np.int64)
        ts_arr = torch.from_numpy(ts_np).to(DEVICE)

        # Causal/uniform negative sampler — reads per-item first-seen from the
        # (TimeAware) training set. Shared across all three training phases.
        self._sampler = TorchCausalSampler(train_set, self.sampler, DEVICE)

        # ---- Phase 1: pre-train GMF (Adam, BCE).
        gmf = GMF(num_users=n_users, num_items=n_items,
                  num_factors=self.num_factors).to(DEVICE)
        self._train_module(gmf, "adam", u_arr, i_arr, ts_arr, ts_np, label="GMF")

        # ---- Phase 2: pre-train MLP (Adam, BCE).
        mlp = MLP(num_users=n_users, num_items=n_items,
                  layers=list(self.layers), act_fn=self.act_fn).to(DEVICE)
        self._train_module(mlp, "adam", u_arr, i_arr, ts_arr, ts_np, label="MLP")

        # ---- Phase 3: build NeuMF, init from pre-trained GMF/MLP, fine-tune with SGD.
        neumf = NeuMFNet(num_users=n_users, num_items=n_items,
                         num_factors=self.num_factors,
                         layers=list(self.layers), act_fn=self.act_fn).to(DEVICE)
        neumf.from_pretrained(gmf, mlp, alpha=self.alpha)
        self._train_module(neumf, "sgd", u_arr, i_arr, ts_arr, ts_np, label="NeuMF")

        self.net = neumf
        # Release intermediate sub-models.
        del gmf, mlp
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    @torch.no_grad()
    def score(self, user_idx, item_idx=None):
        self.net.eval()
        all_i = torch.arange(self.num_items, device=DEVICE)
        u_rep = torch.full_like(all_i, int(user_idx))
        scores = self.net(u_rep, all_i).cpu().numpy()
        if item_idx is None:
            return scores
        return float(scores[item_idx])
