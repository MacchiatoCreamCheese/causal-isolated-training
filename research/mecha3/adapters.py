"""Per-model drivers for the prequential stream.

`prequential.py` owns the protocol -- score a batch, then train on it, walking
forward in time. What differs per model is only three operations, and this module
supplies them:

    prepare(train_set)   build parameters
    score(users)         scores over the whole catalogue, for evaluation
    observe(rows)        one training step on these rows

Nothing here reimplements a model. cornac's architectures and losses are
imported and used as-is -- `NeuMF._build_model_pt()`, LightGCN's `Model` and
`Model.loss_fn`, its `construct_graph`. What is written here is the five-line
"forward, loss, backward, step" that cornac keeps *inside* `fit()`, because
prequential has to interrupt between batches and `fit()` offers no seam. The part
that must not drift from cornac -- the model definition -- still comes from
cornac, so the repo's "imported, never vendored" rule holds.

Negatives come from `NumpyCausalSampler` for every model here, rather than from
the dataset iterators cornac's models normally use. That is a different code path
from the ablation's, so `test_prequential.py` checks rho on it rather than
assuming; both read the same per-item first-seen index, so "causal" means one
thing across the project.
"""

import numpy as np

from ..lib.causal_negative_sampler import NumpyCausalSampler


class Adapter:
    """Common plumbing: row arrays and the shared negative sampler."""

    #: Rows scored per batch. `None` means all of them. NeuMF overrides this --
    #: see its docstring.
    eval_sample = None

    def __init__(self, model, sampler_kind):
        self.model = model
        self.sampler_kind = sampler_kind
        self.sampler = None

    def bind(self, train_set):
        self.train_set = train_set
        u, i, _ = train_set.uir_tuple
        self.u_arr = np.asarray(u, dtype=np.int64)
        self.i_arr = np.asarray(i, dtype=np.int64)
        self.ts_arr = np.asarray(train_set.timestamps, dtype=np.int64)
        self.num_items = train_set.num_items
        self.num_users = train_set.num_users

    def negatives(self, rows, num_neg=1):
        return self.sampler.sample(self.ts_arr[rows], self.u_arr[rows], num_neg)


class BPRAdapter(Adapter):
    """Our NumPy BPR. The only model whose training loop we already own."""

    def prepare(self, train_set):
        self.bind(train_set)
        m = self.model
        m._init_params(self.num_users, self.num_items)
        self.sampler = NumpyCausalSampler(train_set, self.sampler_kind, m.rng)
        m._sampler = self.sampler

    def score(self, users):
        m = self.model
        scores = np.asarray(m.u_factors[users] @ m.i_factors.T, dtype=np.float64)
        if getattr(m, "use_bias", False):
            scores = scores + m.i_biases[None, :]
        return scores

    def observe(self, rows):
        self.model.train_batch(self.u_arr[rows], self.i_arr[rows],
                               self.negatives(rows))


class NeuMFAdapter(Adapter):
    """cornac's NeuMF, stepped one batch at a time.

    The architecture is cornac's `_build_model_pt()`; the loss is `nn.BCELoss`,
    as in `NCFBase._fit_pt`. Only the loop is written here.

    **Scoring is sampled.** NeuMF is an MLP over a concatenated (user, item)
    pair, not a dot product, so scoring one user against the catalogue costs
    `num_items` forward passes rather than one matrix row. Scoring every row of a
    4096-row batch against 22k items would be ~93M forwards per point. So a
    random subset of each batch is scored instead: the metric is an unbiased
    estimate either way, just noisier, and the curve is smoothed for reading
    anyway. `eval_sample` is the knob.
    """

    eval_sample = 256

    def __init__(self, model, sampler_kind, device=None):
        super().__init__(model, sampler_kind)
        self.device = device

    def prepare(self, train_set):
        import torch
        from cornac.models.ncf.backend_pt import optimizer_dict
        from torch import nn

        self.bind(train_set)
        m = self.model
        m.num_users, m.num_items = self.num_users, self.num_items
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if m.seed is not None:
            torch.manual_seed(m.seed)

        self.torch = torch
        self.net = m._build_model_pt().to(self.device)
        self.opt = optimizer_dict[m.learner](self.net.parameters(), lr=m.lr,
                                             weight_decay=m.reg)
        self.criteria = nn.BCELoss()
        self.sampler = NumpyCausalSampler(train_set, self.sampler_kind,
                                          np.random.default_rng(m.seed))

    def score(self, users):
        torch = self.torch
        items = torch.arange(self.num_items, device=self.device)
        out = np.empty((len(users), self.num_items), dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            for row, uid in enumerate(users):
                u = torch.full((self.num_items,), int(uid), dtype=torch.long,
                               device=self.device)
                out[row] = self.net(u, items).cpu().numpy()
        self.net.train()
        return out

    def observe(self, rows):
        torch = self.torch
        m = self.model
        u, pos = self.u_arr[rows], self.i_arr[rows]
        # cornac's uir_iter layout: positives first, then num_neg negatives per
        # positive, labelled 0. Reproduced so the loss sees what it normally sees.
        u_rep = np.repeat(u, m.num_neg)
        neg = self.negatives(rows, m.num_neg).reshape(-1)
        users = np.concatenate([u, u_rep])
        items = np.concatenate([pos, neg])
        labels = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

        bu = torch.from_numpy(users).long().to(self.device)
        bi = torch.from_numpy(items).long().to(self.device)
        br = torch.tensor(labels, dtype=torch.float, device=self.device)

        self.opt.zero_grad()
        loss = self.criteria(self.net(bu, bi), br)
        loss.backward()
        self.opt.step()


class LightGCNAdapter(Adapter):
    """cornac's LightGCN, stepped one batch at a time.

    Architecture, loss and graph construction are all cornac's (`Model`,
    `Model.loss_fn`, `construct_graph`).

    **The graph is training data.** `construct_graph` builds a dgl graph from a
    dataset's entire `uir_tuple`, and LightGCN's whole mechanism is propagating
    signal along those edges. Handing it the full training graph would let
    information from *future* interactions reach a current prediction through
    convolution -- leakage of exactly the kind this project exists to remove, and
    invisible in the output because the curve would simply look good.

    So the graph is rebuilt from the rows seen so far as the stream advances.
    `graph_every` trades precision for speed: with `graph_every > 1` the graph
    lags by up to that many batches, which is *safe* -- a stale graph holds fewer
    past edges, never future ones -- but slightly understates what the model
    could know. Rebuilding is a dgl heterograph construction, so it is the
    dominant cost of a LightGCN prequential run.
    """

    def __init__(self, model, sampler_kind, graph_every=1, device=None):
        super().__init__(model, sampler_kind)
        self.graph_every = max(1, int(graph_every))
        self.device = device
        self._since_rebuild = 0

    def prepare(self, train_set):
        import torch
        from cornac.models.lightgcn.lightgcn import Model

        self.bind(train_set)
        m = self.model
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if m.seed is not None:
            torch.manual_seed(m.seed)

        self.torch = torch
        self._seen_rows = []
        self.graph = self._build_graph()
        self.net = Model(self.graph, m.emb_size, m.num_layers,
                         m.lambda_reg).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=m.learning_rate)
        self.sampler = NumpyCausalSampler(train_set, self.sampler_kind,
                                          np.random.default_rng(m.seed))

    def _build_graph(self):
        """A dgl graph over the rows observed so far, and only those."""
        from cornac.models.lightgcn.lightgcn import construct_graph

        rows = (np.concatenate(self._seen_rows) if self._seen_rows
                else np.empty(0, dtype=np.int64))
        # construct_graph only reads `.uir_tuple`, so a stand-in is enough and
        # avoids rebuilding a whole cornac Dataset per rebuild.
        view = type("RowView", (), {"uir_tuple": (self.u_arr[rows],
                                                  self.i_arr[rows],
                                                  None)})()
        return construct_graph(view, self.num_users,
                               self.num_items).to(self.device)

    def _refresh_graph(self, rows):
        self._seen_rows.append(np.asarray(rows))
        self._since_rebuild += 1
        if self._since_rebuild >= self.graph_every:
            self.graph = self._build_graph()
            self._since_rebuild = 0

    def score(self, users):
        torch = self.torch
        self.net.eval()
        with torch.no_grad():
            u_all, i_all, _ = self.net(self.graph)
            scores = (u_all[torch.from_numpy(np.asarray(users)).long()
                            .to(self.device)] @ i_all.T)
            out = scores.cpu().numpy().astype(np.float64)
        self.net.train()
        return out

    def observe(self, rows):
        torch = self.torch
        u, pos = self.u_arr[rows], self.i_arr[rows]
        neg = self.negatives(rows)
        u_emb, pos_emb, neg_emb = self.net(
            self.graph,
            torch.from_numpy(u).long().to(self.device),
            torch.from_numpy(pos).long().to(self.device),
            torch.from_numpy(np.asarray(neg)).long().to(self.device),
        )
        loss, _bpr, _reg = self.net.loss_fn(u_emb, pos_emb, neg_emb)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._refresh_graph(rows)


def build_adapter(model_name, model, sampler_kind, **kw):
    if model_name == "bpr":
        return BPRAdapter(model, sampler_kind)
    if model_name == "neumf":
        return NeuMFAdapter(model, sampler_kind)
    if model_name == "lightgcn":
        return LightGCNAdapter(model, sampler_kind,
                               graph_every=kw.get("graph_every", 1))
    raise ValueError(f"unknown model: {model_name}")
