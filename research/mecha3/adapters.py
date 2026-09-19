import numpy as np

from ..lib.causal_negative_sampler import NumpyCausalSampler


class Adapter:
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
        from cornac.models.lightgcn.lightgcn import construct_graph

        rows = (np.concatenate(self._seen_rows) if self._seen_rows
                else np.empty(0, dtype=np.int64))
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
