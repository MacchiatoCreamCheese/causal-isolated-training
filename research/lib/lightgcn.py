"""Phase 7: LightGCN with all three faithful-recipe mechanisms.

Wraps cornac's LightGCN (cornac/models/lightgcn/lightgcn.py) and adds:
  - causal negative sampling (Mechanism 1)
  - temporal batching with K=10 chronological windows (Mechanism 2)
  - per-window graph rebuild so the GCN propagation respects causal
    edge availability (Mechanism 2, LightGCN-specific)

Prequential evaluation (Mechanism 3) is handled by the eval method, not
by the model.

The 2x2 ablation cells are the same as for BPR/NeuMF:
  - shuffle+uniform : vanilla LightGCN-style
  - shuffle+causal  : single graph, causal negatives
  - win10+uniform   : per-window graph rebuild, uniform negatives
  - win10+causal    : faithful recipe (all three mechanisms)

GPU-accelerated. Requires `dgl` (cornac LightGCN dependency).
"""

import time
from typing import Optional

import numpy as np
import torch

from cornac.models.recommender import Recommender

from .causal_negative_sampler import TorchCausalSampler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _WindowView:
    """Minimal duck-type accepted by cornac's construct_graph. It only
    reads `uir_tuple`, so we expose exactly that."""

    def __init__(self, u, i, r):
        self.uir_tuple = (u, i, r)


class LightGCNRecommender(Recommender):
    def __init__(
        self,
        name: str = "LightGCN",
        emb_size: int = 64,
        num_layers: int = 3,
        learning_rate: float = 1e-3,
        lambda_reg: float = 1e-4,
        batch_size: int = 1024,
        n_epochs: int = 20,            # total for shuffle; per-window for windowed
        order: str = "shuffle",        # "shuffle" | "windowed"
        sampler: str = "uniform",      # "uniform" | "causal"
        n_windows: int = 10,
        early_stopping: Optional[dict] = None,  # e.g. {"min_delta": 0.0, "patience": 50}
        early_stop_every: int = 10,             # check every N epochs to amortize eval cost
        seed: int = 42,
        verbose: bool = True,
        trainable: bool = True,
    ):
        super().__init__(name=name, trainable=trainable, verbose=verbose)
        self.emb_size = emb_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.order = order
        self.sampler = sampler
        self.n_windows = n_windows
        self.early_stopping = early_stopping
        self.early_stop_every = early_stop_every
        self.seed = seed

        # Negative sampler (uniform/causal) built per-fit; see
        # causal_negative_sampler.TorchCausalSampler.
        self._sampler = None

        # Cached embeddings for score().
        self.U = None  # numpy, (n_users, emb_size)
        self.V = None  # numpy, (n_items, emb_size)

    @property
    def counterfactual_rate(self) -> float:
        return self._sampler.counterfactual_rate if self._sampler is not None else 0.0

    def _set_seeds(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

    def _train_one_pass(self, model, graph, opt, u_t, i_t, ts_t):
        n = len(u_t)
        loss_sum = 0.0
        for start in range(0, n, self.batch_size):
            ub = u_t[start:start + self.batch_size]
            ib = i_t[start:start + self.batch_size]
            tb = ts_t[start:start + self.batch_size]
            jb = self._sampler.sample(tb)

            u_emb, pos_emb, neg_emb = model(graph, ub, ib, jb)
            loss, _, _ = model.loss_fn(u_emb, pos_emb, neg_emb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(ub)
        return loss_sum / max(n, 1)

    def fit(self, train_set, val_set=None):
        Recommender.fit(self, train_set, val_set)
        from cornac.models.lightgcn.lightgcn import construct_graph, Model

        self._set_seeds()
        n_users = self.total_users
        n_items = self.total_items

        u_np = train_set.uir_tuple[0].astype(np.int64)
        i_np = train_set.uir_tuple[1].astype(np.int64)
        ts_np = np.asarray(train_set.timestamps, dtype=np.int64)

        # Causal/uniform negative sampler — reads per-item first-seen from the
        # (TimeAware) training set; on-device draws for speed.
        self._sampler = TorchCausalSampler(train_set, self.sampler, DEVICE)

        if self.verbose:
            print(f"[{self.name}] fit start: users={n_users} items={n_items} "
                  f"rows={len(u_np)}  order={self.order} sampler={self.sampler}",
                  flush=True)

        t0 = time.time()
        if self.order == "shuffle":
            # Single graph from all training rows.
            full_view = _WindowView(u_np, i_np, np.ones_like(u_np, dtype=np.float32))
            graph = construct_graph(full_view, n_users, n_items).to(DEVICE)
            model = Model(graph, self.emb_size, self.num_layers, self.lambda_reg).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            u_t = torch.from_numpy(u_np).to(DEVICE)
            i_t = torch.from_numpy(i_np).to(DEVICE)
            ts_t = torch.from_numpy(ts_np).to(DEVICE)

            self._reset_early_stop()
            for epoch in range(self.n_epochs):
                perm = torch.randperm(len(u_t), device=DEVICE)
                model.train()
                loss = self._train_one_pass(model, graph, opt,
                                            u_t[perm], i_t[perm], ts_t[perm])
                if self.verbose:
                    print(f"[{self.name}] epoch {epoch+1}/{self.n_epochs}  "
                          f"loss={loss:.4f}  t={time.time()-t0:.1f}s", flush=True)
                if (self.early_stopping is not None and val_set is not None
                        and (epoch + 1) % self.early_stop_every == 0):
                    self._refresh_embeddings(model, graph)
                    if self.early_stop(train_set, val_set, **self.early_stopping):
                        if self.verbose:
                            print(f"[{self.name}] early-stop at epoch {epoch+1}", flush=True)
                        break
        else:
            # Windowed: per-window graph rebuild on cumulative prefix.
            order_idx = np.argsort(ts_np, kind="stable")
            u_sorted = u_np[order_idx]
            i_sorted = i_np[order_idx]
            ts_sorted = ts_np[order_idx]
            boundaries = np.linspace(0, len(u_sorted), self.n_windows + 1, dtype=int)

            # Build initial graph from window 1's cumulative prefix to instantiate Model.
            first_view = _WindowView(
                u_sorted[:boundaries[1]],
                i_sorted[:boundaries[1]],
                np.ones(boundaries[1], dtype=np.float32),
            )
            graph = construct_graph(first_view, n_users, n_items).to(DEVICE)
            model = Model(graph, self.emb_size, self.num_layers, self.lambda_reg).to(DEVICE)
            opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

            for w in range(self.n_windows):
                lo, hi = boundaries[w], boundaries[w + 1]
                cumul_end = hi
                # Rebuild graph for this window (cumulative through end-of-window-k).
                view = _WindowView(
                    u_sorted[:cumul_end],
                    i_sorted[:cumul_end],
                    np.ones(cumul_end, dtype=np.float32),
                )
                # Free previous graph before allocating new.
                del graph
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                graph = construct_graph(view, n_users, n_items).to(DEVICE)

                u_w = torch.from_numpy(u_sorted[lo:hi]).to(DEVICE)
                i_w = torch.from_numpy(i_sorted[lo:hi]).to(DEVICE)
                ts_w = torch.from_numpy(ts_sorted[lo:hi]).to(DEVICE)

                self._reset_early_stop()
                for ep in range(self.n_epochs):
                    perm = torch.randperm(len(u_w), device=DEVICE)
                    model.train()
                    loss = self._train_one_pass(model, graph, opt,
                                                u_w[perm], i_w[perm], ts_w[perm])
                    if (self.early_stopping is not None and val_set is not None
                            and (ep + 1) % self.early_stop_every == 0):
                        self._refresh_embeddings(model, graph)
                        if self.early_stop(train_set, val_set, **self.early_stopping):
                            if self.verbose:
                                print(f"[{self.name}] early-stop window {w+1} "
                                      f"at epoch {ep+1}", flush=True)
                            break
                if self.verbose:
                    print(f"[{self.name}] rebuilt graph + trained window {w+1}/{self.n_windows}  "
                          f"window_rows={hi-lo}  cumul_edges={cumul_end}  "
                          f"last_loss={loss:.4f}  t={time.time()-t0:.1f}s", flush=True)

        # Cache final embeddings for inference.
        model.eval()
        with torch.no_grad():
            u_embs, i_embs, _ = model(graph)
        self.U = u_embs.cpu().detach().numpy()
        self.V = i_embs.cpu().detach().numpy()
        return self

    def score(self, user_idx, item_idx=None):
        if item_idx is None:
            return self.V.dot(self.U[user_idx, :])
        return float(self.V[item_idx, :].dot(self.U[user_idx, :]))

    def _reset_early_stop(self):
        """Reset cornac's early-stop counters so each training segment
        (the shuffle run, or each windowed window) gets its own budget."""
        self.current_epoch = 0
        self.best_value = float("-inf")
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.wait = 0

    def _refresh_embeddings(self, model, graph):
        """Copy current model embeddings into self.U/self.V so monitor_value
        (which uses self.score) reflects the live model state."""
        model.eval()
        with torch.no_grad():
            u_embs, i_embs, _ = model(graph)
        self.U = u_embs.cpu().detach().numpy()
        self.V = i_embs.cpu().detach().numpy()

    def monitor_value(self, train_set, val_set):
        """Validation recall@20 — same monitor used by cornac's own LightGCN
        (`cornac/models/lightgcn/recom_lightgcn.py:227`)."""
        if val_set is None:
            return None
        from cornac.metrics import Recall
        from cornac.eval_methods.base_method import ranking_eval
        recall_20 = ranking_eval(
            model=self,
            metrics=[Recall(k=20)],
            train_set=train_set,
            test_set=val_set,
        )[0][0]
        return recall_20
