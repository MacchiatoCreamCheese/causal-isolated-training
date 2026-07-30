# Hyperparameter inventory

Every knob in every model, with its current value, where it came from, and whether we're tuning it. Source of truth: `research/lib/tuning_config.py`.

**Status labels.**
- **PAPER_FIXED** — original paper explicitly fixes this value; do not search.
- **PAPER_GRID** — original paper specifies a search grid.
- **PAPER_DEFAULT** — paper mentions a default but does not search.
- **CORNAC_DEFAULT** — cornac's published default; no paper claim.
- **TORCH_DEFAULT** — pytorch library default; we inherit it.
- **EYEBALL** — Claude's pick. To be revisited.
- **RESEARCH** — our research mechanism (not in any paper).
- **NOT_IMPLEMENTED** — knob exists conceptually but our code does not handle it.

Reference papers:
- **LightGCN:** He et al., SIGIR '20 (`LightGCN.pdf`)
- **NeuMF:** He et al., WWW '17 (`NeuMF.pdf`)
- **BPR:** Rendle et al., UAI '09; Milogradskii et al., RecSys '24 (`NewBPR.pdf`) for the modern hyperparameter recipe.

---

## LightGCN

### Fixed by the paper (do not tune)

| Knob | Value | Source quote |
|---|---|---|
| `emb_size` | 64 | §4.1.2: *"the embedding size is fixed to 64 for all models and the embedding parameters are initialized with the Xavier method [12]."* |
| `alpha_k_formula` | `1/(K+1)` | §4.1.2: *"The layer combination coefficient αk is uniformly set to 1/(1+K) where K is the number of layers."* + §3.1.2: *"We do not design special component to optimize αk, to avoid complicating LightGCN unnecessarily and to keep its simplicity."* |
| `normalization_kind` | `symmetric_sqrt` | §4.4.2: *"The best setting in general is using sqrt normalization at both sides (i.e., the current design of LightGCN)."* |
| `self_connection` | `False` | §3.1.1: *"in LGC, we aggregate only the connected neighbors and do not integrate the target node itself"* |
| `dropout_p` | `0.0` | §3.3: *"Note that we do not introduce dropout mechanisms... enforcing L2 regularization on the embedding layer is sufficient to prevent overfitting."* |
| `loss` | BPR | §3.3: *"We employ the Bayesian Personalized Ranking (BPR) loss [32], which is a pairwise loss..."* |
| `optimizer_kind` | `adam` | §4.1.2: *"We optimize LightGCN with Adam [22]."* |
| `weight_init_kind` | `xavier_uniform` | §4.1.2: *"embedding parameters are initialized with the Xavier method"* |
| `batch_size_baby` | 1024 | §4.1.2 default 1024; Baby (1.24M rows) ~ Yelp2018 scale (1.56M). |
| `batch_size_cellphone` | 2048 | §4.1.2: *"on Amazon-Book, we increase the mini-batch size to 2048 for speed"*; Cellphone (2.75M) ~ Amazon-Book (2.98M). |
| `batch_size_movielens1m` | 1024 | §4.1.2 default 1024; MovieLens-1M (1.00M) ~ Gowalla (1.03M). |
| `n_epochs_max` | 1000 | §4.1.2: *"Typically, 1000 epochs are sufficient for LightGCN to converge."* |
| `monitor_metric` | `recall@20` | §4.1.2 + cornac LightGCN.monitor_value: recall@20 on val_set (same as NGCF). |

### Paper grids (paper itself searches over these)

| Knob | Current | Grid | Source quote |
|---|---|---|---|
| `num_layers` (K) | 3 | {1, 2, 3, 4} | §4.1.2: *"We test K in the range of 1 to 4, and satisfactory performance can be achieved when K equals to 3."* |
| `lambda_reg` (L2 λ) | 1e-4 | {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | §4.1.2: *"The L2 regularization coefficient λ is searched in the range of {1e-6, 1e-5, ..., 1e-2}, and in most cases the optimal value is 1e-4."* §4.5: *"the most important hyper-parameter to tune is the L2 regularization coefficient λ"* |

### Paper default (not searched in the paper)

| Knob | Value | Source |
|---|---|---|
| `learning_rate` | 1e-3 | §4.1.2: *"use the default learning rate of 0.001"* — paper does not search lr. We treat it as a tuning knob anyway because real datasets vary. |

### PyTorch defaults (paper does not specify)

| Knob | Value | Why |
|---|---|---|
| `adam_beta1` | 0.9 | `torch.optim.Adam` default. |
| `adam_beta2` | 0.999 | `torch.optim.Adam` default. |
| `adam_eps` | 1e-8 | `torch.optim.Adam` default. |
| `xavier_gain` | 1.0 | `torch.nn.init.xavier_uniform_` default. |

### Eyeballed (we chose; revisit)

| Knob | Current | Why eyeball |
|---|---|---|
| `batch_size_healthcare` | 2048 | Healthcare (7.18M) is 2.4× Amazon-Book. Paper only mentions 2048 as max; could justify 4096. |
| `early_stop_min_delta` | 0.0 | Paper §4.1.2 says "same as NGCF" without quoting min_delta. |
| `early_stop_patience` | 5 | Paper does not quote a patience value. patience=5 with check_every=10 ⇒ stop after 50 epochs without recall@20 improvement. |
| `early_stop_check_every` | 10 | Trade-off between eval cost and stopping precision; cornac's own LightGCN checks every epoch. |

### Research (our mechanisms)

| Knob | Values | Notes |
|---|---|---|
| `order` | `shuffle` \| `windowed` | Mechanism 2 (temporal batching). |
| `sampler` | `uniform` \| `causal` | Mechanism 1 (causal negative sampling). |
| `n_windows` | 10 | Our Phase 4a K-ablation; K=10 was the Pareto sweet spot on Baby/BPR. Grid {2, 5, 10, 20, 50}. |
| `epochs_per_window` | 1000 | Equal to `n_epochs_max` for per-row exposure parity with the shuffle baseline. |

---

## NeuMF

### Fixed by the paper (do not tune)

| Knob | Value | Source quote |
|---|---|---|
| `act_fn` | `relu` | §3.3: *"we opt for ReLU... empirical results show that ReLU yields slightly better performance than tanh."* |
| `layer_pattern` | tower halving | §3.3: *"empirically implement the tower structure, halving the layer size for each successive higher layer"* — e.g., factors=8 ⇒ [32, 16, 8]. |
| `pretrain` | `True` | §4.2.1 + Table 2: NeuMF-with-pretrain wins on both datasets (~1–2% better in most cells). We do not ablate; pretrain is always on. |
| `alpha_mixing` (NeuMF α) | 0.5 | §4.1: *"For the NeuMF with pre-training, α was set to 0.5, allowing the pre-trained GMF and MLP to contribute equally to NeuMF's initialization."* |
| `loss` | BCE (log loss) | §3.1.1: *"binary cross-entropy loss, also known as log loss."* |
| `optimizer_pretrain_kind` | `adam` | §3.4.1: *"For training GMF and MLP from scratch, we adopt the Adaptive Moment Estimation (Adam)"* |
| `optimizer_finetune_kind` | `sgd` (vanilla) | §3.4.1: *"we optimize it with the vanilla SGD, rather than Adam. This is because Adam needs to save momentum information for updating parameters properly."* |
| `optimizer_finetune_sgd_momentum` | 0.0 | §3.4.1: *"vanilla SGD"* (no momentum). |
| `weight_decay` | 0.0 | He 2017 does not use weight decay; cornac NCF default `reg=0.0`. |
| `weight_init_kind` | gaussian | §4.1: *"randomly initialized model parameters with a Gaussian distribution"* |
| `weight_init_mean` | 0.0 | §4.1: *"mean of 0"* |
| `weight_init_std` | 0.01 | §4.1: *"standard deviation of 0.01"* |
| `monitor_metric` | HR@10 | §4.1: *"truncated the ranked list at 10 ... Hit Ratio (HR)"* |

### Paper grids (paper itself searches over these)

| Knob | Current | Grid | Source quote |
|---|---|---|---|
| `num_factors` | 8 | {8, 16, 32, 64} | §4.1: *"evaluated the factors of [8, 16, 32, 64]."* |
| `mlp_hidden_count` | 3 | {0, 1, 2, 3, 4} | §4.4 + Tables 3/4: MLP-0 through MLP-4; *"MLP-4 generally best on both datasets."* |
| `batch_size` | 256 | {128, 256, 512, 1024} | §4.1: *"We tested the batch size of [128, 256, 512, 1024]."* |
| `learning_rate` | 1e-3 | {1e-4, 5e-4, 1e-3, 5e-3} | §4.1: *"the learning rate of [0.0001, 0.0005, 0.001, 0.005]."* |
| `num_neg` | 4 | {3, 4, 5, 6} | §4.3 + Fig 7: *"optimal sampling ratio is around 3 to 6."* (Paper tested 1–10; we restrict to the optimum band.) |

### Paper default (not searched in the paper)

| Knob | Value | Source |
|---|---|---|
| `mlp_layers` | (64, 32, 16, 8) | §3.3 tower-halving with `layers[-1] == num_factors`; cornac NCF default `layers=[64,32,16,8]`. |

### Cornac default

| Knob | Value | Why |
|---|---|---|
| `n_epochs_neumf` | 20 | cornac NCF default `num_epochs=20`. He 2017 does not quote a number; Fig 6 shows convergence ~20 iterations on MovieLens. |

### PyTorch defaults

| Knob | Value |
|---|---|
| `optimizer_pretrain_adam_beta1` | 0.9 |
| `optimizer_pretrain_adam_beta2` | 0.999 |
| `optimizer_pretrain_adam_eps` | 1e-8 |
| `optimizer_finetune_sgd_nesterov` | False |

### Eyeballed (we chose; revisit)

| Knob | Current | Why eyeball |
|---|---|---|
| `n_windows` | 10 | Reused from our BPR K-ablation. Not re-tuned for NeuMF. |
| `epochs_per_window` | 20 | Equal to `n_epochs_neumf` so per-row exposure matches shuffle. |

(Pre-train epoch counts are *not* a separate knob — our `NeuMFRecommender.fit` reuses one shared `n_epochs` across all three phases, matching cornac's typical "set num_epochs=20 on each of GMF/MLP/NeuMF" pattern. See `DEFERRED.md §1` for the refactor that would expose per-phase counts.)

### NOT_IMPLEMENTED (knob exists conceptually but our code does not handle it)

| Knob | Why |
|---|---|
| `early_stop_min_delta` | Early stopping not wired into our custom NeuMFRecommender training loop. |
| `early_stop_patience` | Same. NewBPR (BPR side) uses patience=13; we'd want similar here. |
| `early_stop_check_every` | Same. |

### Research (our mechanisms)

| Knob | Values | Notes |
|---|---|---|
| `order` | `shuffle` \| `windowed` | Applied to all three training phases (GMF, MLP, NeuMF). |
| `sampler` | `uniform` \| `causal` | Applied to all three training phases. |

---

## BPR

### Fixed by the paper (do not tune)

| Knob | Value | Source quote |
|---|---|---|
| `use_bias` | False | NewBPR §5.3 (global temporal split): *"disabled item biases ... performed best on the MSD dataset"*. Was True (cornac default) until 2026-05-29; flipped. |
| `regularization_scheme` | separated (λu, λi, λj) | NewBPR §5.2/§5.3: *"distinct regularization lambdas for users, positive and negative items, are crucial for specific datasets"*; the global-temporal-split winner uses *"separated regularization factors (λu, λi, λj)"*. Implemented 2026-06-06. |
| `loss` | BPR pairwise | Rendle 2009 / NewBPR §3.2: pairwise log-likelihood −Σ log σ(score_ui − score_uj). |
| `optimizer_kind` | SGD (manual) | NewBPR §5.3: *"standard SGD demonstrates commendable performance ... preferred optimizer for the BPR model on the global temporal split"*. Rendle 2009 also SGD. We do manual SGD updates inside the training loop; not `torch.optim.SGD`. |
| `sgd_momentum` | 0.0 | NewBPR §5.3: *"standard SGD"* (not Momentum SGD which they evaluated separately). |
| `monitor_metric` | NDCG@100 | NewBPR §4.1.6: *"We search for the best hyperparameters using NDCG@100 on all datasets except Netflix"*. |

### Paper grids (paper itself searches over these)

| Knob | Current | Grid | Source quote |
|---|---|---|---|
| `k_embed_dim` | 64 | {32, 64, 128, 256, 512, 1024} | NewBPR §4.1.7: *"with varying dimensions: 32, 64, 128, 256, 512, and 1024 for ML-20M time-splitted and Yelp."* Finding: *"Cornac achieves the best results with embedding dimensions of 512 and 1024 on both datasets"* (Fig. 3). |
| `lambda_u` | 1e-4 | {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | NewBPR §5.2: *"distinct regularization lambdas for users, positive and negative items, are crucial for specific datasets"*. Range adapted from the LightGCN §4.1.2 grid that NewBPR also references. |
| `lambda_i` | 1e-4 | {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | NewBPR §5.2/§5.3 — separate λ for positive-item embeddings. |
| `lambda_j` | 1e-4 | {1e-6, 1e-5, 1e-4, 1e-3, 1e-2} | NewBPR §5.2/§5.3 — separate λ for negative-item embeddings. |

### Cornac defaults

| Knob | Value | Notes |
|---|---|---|
| `weight_init_kind` | uniform_centered | `cornac/models/bpr/recom_bpr.pyx:149-151`: `(uniform(0,1) - 0.5) / k`. Rendle 2009 does not specify. |
| `weight_init_low` | `-0.5/k` | Cornac default. |
| `weight_init_high` | `0.5/k` | Cornac default. |

### Eyeballed (we chose; revisit)

| Knob | Current | Why eyeball |
|---|---|---|
| `learning_rate` | 0.05 | Cornac default `lr=0.001` (paired with 100 iterations). NewBPR tunes via Optuna without enumerated bounds. |
| `batch_size` | 16384 | GPU throughput choice. NewBPR (PyTorch) and Elliot's Batched BPRMF both batch; neither pins a value. |
| `early_stop_min_delta` | 0.0 | NewBPR doesn't quote a min_delta; 0.0 = strict "any improvement". |
| `early_stop_check_every` | 1 | Every epoch, so `patience=13` literally means "13 epochs without improvement" (NewBPR-faithful translation). |
| `monitor_metric_early_stop` | NDCG@20 | NewBPR optimizes NDCG@100 via Optuna; we monitor NDCG@20 to align with our tuner selection metric and the reported tables. |

### Paper-fixed: training schedule

| Knob | Value | Source |
|---|---|---|
| `n_epochs` (ceiling) | 1000 | NewBPR §5.2: *"1000 epochs for the best model on the ML-20M dataset"*. Budget ceiling; early stopping decides the actual stop. |
| `early_stop_patience` | 13 | NewBPR §5.2: *"training protocol employs the Early Stopping criterion with patience equal to 13 epochs"*. |

### Research (our mechanisms)

| Knob | Values | Notes |
|---|---|---|
| `sampler_kind` | `uniform` \| `causal` | Our causal mechanism. NewBPR §5.3 recommends adaptive sampling for global temporal split; we have NOT implemented adaptive — orthogonal to our axis. |
| `order` | `shuffle` \| `windowed` | `BPRWindowedGPU` subclass. |
| `n_windows` | 10 | Our paper Phase 4a K-ablation (`sample.tex` Table VI / Fig 4). K=10 = Pareto sweet spot on Baby/BPR. Grid {2, 5, 10, 20, 50}. |
| `epochs_per_window` | 20 | Equal to `n_epochs` for per-row exposure parity. |

---

## Coordinate-descent tuning sweep

These are the knobs we're actually sweeping (`lib/tuning_config.py:TUNE_ORDER`), in order. Order = most-impactful first per each paper's own ablation. Each grid **omits the default** so the baseline run is reused (no duplicate training).

Tuning runs on **Baby × shuffle+uniform × seed=42**, selecting on **NDCG@20**. Winner config is then transplanted to all 4 datasets × 4 recipes × 3 seeds.

### LightGCN — 10 trials

| Order | Knob | Default | Grid (excl. default) | Rationale |
|---|---|---|---|---|
| 1 | `lambda_reg` | 1e-4 | 1e-6, 1e-5, 1e-3, 1e-2 | He 2020 §4.5: *"most important hyper-parameter to tune is λ"*. |
| 2 | `num_layers` | 3 | 1, 2, 4 | He 2020 §4.1.2 paper grid. |
| 3 | `learning_rate` | 1e-3 | 1e-4, 5e-4, 5e-3 | Paper does not search lr; we add a small grid. |

### NeuMF — 15 trials

| Order | Knob | Default | Grid (excl. default) | Rationale |
|---|---|---|---|---|
| 1 | `num_factors` | 8 | 16, 32, 64 | He 2017 §4.1 paper grid. |
| 2 | `mlp_hidden_count` | 3 | 1, 2, 4 | He 2017 §4.4 paper grid (excl. 0; degenerate tower). |
| 3 | `learning_rate` | 1e-3 | 1e-4, 5e-4, 5e-3 | He 2017 §4.1 paper grid. |
| 4 | `num_neg` | 4 | 3, 5, 6 | He 2017 §4.3 optimum band 3–6. |
| 5 | `batch_size` | 256 | 128, 512, 1024 | He 2017 §4.1 paper grid. |

### BPR — 17 trials

| Order | Knob | Default | Grid (excl. default) | Rationale |
|---|---|---|---|---|
| 1 | `k_embed_dim` | 64 | 32, 128, 256, 512, 1024 | NewBPR §4.1.7 paper grid. |
| 2 | `learning_rate` | 0.05 | 1e-3, 1e-2, 1e-1 | NewBPR uses Optuna; we eyeball a small grid. |
| 3 | `lambda_u` | 1e-4 | 1e-6, 1e-5, 1e-3, 1e-2 | NewBPR §5.2/§5.3: separated λs tuned independently. |
| 4 | `lambda_i` | 1e-4 | 1e-6, 1e-5, 1e-3, 1e-2 | Same. |
| 5 | `lambda_j` | 1e-4 | 1e-6, 1e-5, 1e-3, 1e-2 | Same. |

`n_epochs` is **not tuned**: the ceiling is fixed at 1000 (NewBPR §5.2) and early stopping (patience=13, NDCG@20 monitor) decides the actual stop per trial.
