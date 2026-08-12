"""Hyperparameter inventory and search spaces — every knob, atomized.

This file is the single source of truth for what each model sets. Every
leaf hyperparameter has its own entry; compound objects (like early
stopping config) are split into one entry per scalar so you can change
each in isolation.

Entry schema:
    {
        "current": <value our code uses right now>,
        "status":  <one of the labels below>,
        "source":  <paper section / file path / "Claude">,
        "grid":    <only for PAPER_GRID>,
        "notes":   <optional>,
    }

Status labels:
    PAPER_FIXED      — original paper explicitly fixes this value.
    PAPER_GRID       — original paper specifies a search grid.
    PAPER_DEFAULT    — paper mentions a default but does not search.
    CORNAC_DEFAULT   — cornac's published default; no paper claim.
    TORCH_DEFAULT    — pytorch's library default; we inherit it.
    EYEBALL          — Claude's pick, to be revisited.
    RESEARCH         — our research mechanism, not in any paper.
    NOT_IMPLEMENTED  — knob exists conceptually but our code does not handle it.

References:
    LightGCN:  He et al., SIGIR '20  (LightGCN.pdf)
    NeuMF:     He et al., WWW '17    (NeuMF.pdf)
    BPR:       Rendle et al., UAI '09;
               Milogradskii et al., RecSys '24 (NewBPR.pdf)
"""

# ===========================================================================
# LightGCN
# ===========================================================================

LIGHTGCN = {
    # --- Architecture ---
    "emb_size": {
        "current": 64,
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'embedding size is fixed to 64 for all models'",
    },
    "num_layers": {
        "current": 3,
        "status": "PAPER_GRID",
        "grid": [1, 2, 3, 4],
        "source": "He 2020 §4.1.2: 'We test K in the range of 1 to 4, and satisfactory performance can be achieved when K equals to 3'",
    },
    "alpha_k_formula": {
        "current": "1/(K+1)",
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'uniformly set to 1/(1+K)' / §3.1.2: 'we do not design special component to optimize αk'",
        "notes": "α_k is the weight of the k-th propagation layer in the final embedding.",
    },
    "normalization_kind": {
        "current": "symmetric_sqrt",
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.4.2: 'best setting in general is using sqrt normalization at both sides'",
    },
    "self_connection": {
        "current": False,
        "status": "PAPER_FIXED",
        "source": "He 2020 §3.1.1: 'in LGC, we aggregate only the connected neighbors and do not integrate the target node itself'",
    },
    "dropout_p": {
        "current": 0.0,
        "status": "PAPER_FIXED",
        "source": "He 2020 §3.3: 'we do not introduce dropout mechanisms'",
    },

    # --- Loss ---
    "loss": {
        "current": "BPR",
        "status": "PAPER_FIXED",
        "source": "He 2020 §3.3: 'We employ the Bayesian Personalized Ranking (BPR) loss'",
    },

    # --- Optimizer ---
    "optimizer_kind": {
        "current": "adam",
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'We optimize LightGCN with Adam'",
    },
    "adam_beta1": {
        "current": 0.9,
        "status": "TORCH_DEFAULT",
        "source": "pytorch torch.optim.Adam default beta1=0.9. Paper does not specify.",
    },
    "adam_beta2": {
        "current": 0.999,
        "status": "TORCH_DEFAULT",
        "source": "pytorch torch.optim.Adam default beta2=0.999. Paper does not specify.",
    },
    "adam_eps": {
        "current": 1e-8,
        "status": "TORCH_DEFAULT",
        "source": "pytorch torch.optim.Adam default eps=1e-8.",
    },

    # --- Weight init ---
    "weight_init_kind": {
        "current": "xavier_uniform",
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'embedding parameters are initialized with the Xavier method'",
    },
    "xavier_gain": {
        "current": 1.0,
        "status": "TORCH_DEFAULT",
        "source": "pytorch torch.nn.init.xavier_uniform_ default gain=1.0. Paper does not specify a gain.",
    },

    # --- Tunable scalars ---
    "learning_rate": {
        "current": 1e-3,
        "status": "PAPER_DEFAULT",
        "source": "He 2020 §4.1.2: 'use the default learning rate of 0.001' — paper does not search lr.",
    },
    "lambda_reg": {
        "current": 1e-4,
        "status": "PAPER_GRID",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "He 2020 §4.1.2: 'L2 regularization coefficient λ is searched in the range of {1e-6, 1e-5, ..., 1e-2}, and in most cases the optimal value is 1e-4'",
    },
    "batch_size_musical": {
        "current": 1024,
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2 default 1024. Musical Instruments (0.51M) is smaller than every dataset in the paper; the default applies.",
    },
    "batch_size_baby": {
        "current": 1024,
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2 default 1024. Baby (1.24M) ~ Yelp2018 scale (1.56M).",
    },
    "batch_size_cellphone": {
        "current": 2048,
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'on Amazon-Book, we increase the mini-batch size to 2048 for speed'. Cellphone (2.75M) ~ Amazon-Book (2.98M).",
    },
    "batch_size_healthcare": {
        "current": 2048,
        "status": "EYEBALL",
        "source": "Claude. Healthcare (7.18M) is 2.4× Amazon-Book; paper only mentions 2048 max. Could justify 4096.",
    },

    # --- Training schedule ---
    "n_epochs_max": {
        "current": 1000,
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'Typically, 1000 epochs are sufficient for LightGCN to converge'",
    },
    "early_stop_min_delta": {
        "current": 0.0,
        "status": "EYEBALL",
        "source": "Claude. Paper §4.1.2 says 'same as NGCF' without quoting min_delta.",
    },
    "early_stop_patience": {
        "current": 5,
        "status": "EYEBALL",
        "source": "Claude. Paper does not quote a patience value.",
        "notes": "patience=5 with check_every=10 ⇒ stop after 50 epochs without recall@20 improvement.",
    },
    "early_stop_check_every": {
        "current": 10,
        "status": "EYEBALL",
        "source": "Claude. Trade-off between eval cost and stopping precision; cornac's own LightGCN checks every epoch.",
    },
    "monitor_metric": {
        "current": "recall@20",
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2 + cornac LightGCN.monitor_value: recall@20 on val_set (same as NGCF)",
    },

    # --- Our research mechanism ---
    "sampler": {
        "current": "uniform | causal",
        "status": "RESEARCH",
        "source": "Our causal-negative-sampling mechanism. Set on the training split, not on the model: cornac's LightGCN takes its negatives from train_set.uij_iter().",
    },
}


# ===========================================================================
# NeuMF
# ===========================================================================

NEUMF = {
    # --- Architecture: GMF + MLP towers ---
    "num_factors": {
        "current": 8,
        "status": "PAPER_GRID",
        "grid": [8, 16, 32, 64],
        "source": "He 2017 §4.1: 'evaluated the factors of [8, 16, 32, 64]'",
    },
    "mlp_layers": {
        "current": (64, 32, 16, 8),
        "status": "PAPER_DEFAULT",
        "source": "He 2017 §3.3 tower-halving pattern, cornac NCF default layers=[64,32,16,8]. Constraint: layers[-1] == num_factors.",
        "notes": "Tower-halving means each layer is half the previous. layers[0] = 2 * embedding_dim (concat of user+item embeddings).",
    },
    "mlp_hidden_count": {
        "current": 3,
        "status": "PAPER_GRID",
        "grid": [0, 1, 2, 3, 4],
        "source": "He 2017 §4.4 + Tables 3/4: MLP-0 through MLP-4. 'MLP-4 generally best on both datasets'.",
        "notes": "len(layers) - 1 = number of hidden layers beyond the embedding concat.",
    },
    "act_fn": {
        "current": "relu",
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.3: 'we opt for ReLU, which is more biologically plausible and proven to be non-saturated'",
    },
    "layer_pattern": {
        "current": "tower_halving",
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.3: 'tower structure, halving the layer size for each successive higher layer'",
    },

    # --- Pre-training + fusion ---
    "pretrain": {
        "current": True,
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.2.1 + Table 2: NeuMF-with-pretrain wins over NeuMF-from-scratch on both datasets. We do not ablate this; pretrain is always on.",
    },
    "alpha_mixing": {
        "current": 0.5,
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.1: 'α was set to 0.5, allowing the pre-trained GMF and MLP to contribute equally'",
    },

    # --- Loss ---
    "loss": {
        "current": "BCE",
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.1.1: 'binary cross-entropy loss, also known as log loss'",
    },

    # --- Optimizers ---
    "optimizer_pretrain_kind": {
        "current": "adam",
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.4.1: 'For training GMF and MLP from scratch, we adopt the Adaptive Moment Estimation (Adam)'",
    },
    "optimizer_pretrain_adam_beta1": {
        "current": 0.9,
        "status": "TORCH_DEFAULT",
        "source": "pytorch default. Paper does not specify Adam betas.",
    },
    "optimizer_pretrain_adam_beta2": {
        "current": 0.999,
        "status": "TORCH_DEFAULT",
        "source": "pytorch default.",
    },
    "optimizer_pretrain_adam_eps": {
        "current": 1e-8,
        "status": "TORCH_DEFAULT",
        "source": "pytorch default.",
    },
    "optimizer_finetune_kind": {
        "current": "sgd",
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.4.1: 'we optimize it with the vanilla SGD, rather than Adam'",
    },
    "optimizer_finetune_sgd_momentum": {
        "current": 0.0,
        "status": "PAPER_FIXED",
        "source": "He 2017 §3.4.1: 'vanilla SGD' (no momentum) — 'Adam needs to save momentum ... unsuitable to further optimize NeuMF with momentum-based methods'",
    },
    "optimizer_finetune_sgd_nesterov": {
        "current": False,
        "status": "TORCH_DEFAULT",
        "source": "pytorch SGD default nesterov=False.",
    },
    "weight_decay": {
        "current": 0.0,
        "status": "PAPER_FIXED",
        "source": "He 2017 does not use weight decay. cornac NCF default reg=0.0.",
    },

    # --- Weight init ---
    "weight_init_kind": {
        "current": "gaussian",
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.1: 'Gaussian distribution (with a mean of 0 and standard deviation of 0.01)'",
    },
    "weight_init_mean": {
        "current": 0.0,
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.1: 'mean of 0'",
    },
    "weight_init_std": {
        "current": 0.01,
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.1: 'standard deviation of 0.01'",
    },

    # --- Tunable scalars ---
    "batch_size": {
        "current": 256,
        "status": "PAPER_GRID",
        "grid": [128, 256, 512, 1024],
        "source": "He 2017 §4.1: 'We tested the batch size of [128, 256, 512, 1024]'",
    },
    "learning_rate": {
        "current": 1e-3,
        "status": "PAPER_GRID",
        "grid": [1e-4, 5e-4, 1e-3, 5e-3],
        "source": "He 2017 §4.1: 'learning rate of [0.0001, 0.0005, 0.001, 0.005]'",
    },
    "num_neg": {
        "current": 4,
        "status": "PAPER_GRID",
        "grid": [3, 4, 5, 6],
        "source": "He 2017 §4.3 + Fig 7: 'optimal sampling ratio is around 3 to 6'. (They tested 1–10; we use optimum band.)",
    },

    # --- Training schedule ---
    "n_epochs_neumf": {
        "current": 20,
        "status": "CORNAC_DEFAULT",
        "source": "cornac NCF default num_epochs=20. He 2017 does not quote a number; Fig 6 shows convergence ~20 iterations on MovieLens.",
    },
    "early_stop_min_delta": {
        "current": None,
        "status": "CORNAC_DEFAULT",
        "source": "cornac NCFBase accepts early_stopping={'min_delta':..., 'patience':...}; we leave it None (off). Available for free now that we use cornac's NeuMF.",
    },
    "early_stop_patience": {
        "current": None,
        "status": "CORNAC_DEFAULT",
        "source": "As above — off by default; He 2017 does not pin a patience.",
    },
    "monitor_metric": {
        "current": "HR@10",
        "status": "PAPER_FIXED",
        "source": "He 2017 §4.1: 'truncated the ranked list at 10 ... Hit Ratio (HR)'",
    },

    # --- Our research mechanism ---
    "sampler": {
        "current": "uniform | causal",
        "status": "RESEARCH",
        "source": "Our causal-negative-sampling mechanism. Set on the training split, not on the model: cornac's NCF family takes its negatives from train_set.uir_iter(..., num_zeros=num_neg), so all three training phases (GMF, MLP, NeuMF) inherit it.",
    },
}


# ===========================================================================
# BPR  (research/lib/bpr_cpu.py: BPRMiniBatch)
# ===========================================================================

BPR = {
    # --- Architecture ---
    "k_embed_dim": {
        "current": 64,
        "status": "EYEBALL",
        "source": "Claude. Rendle 2009 doesn't fix k. Cornac default k=10. NewBPR §4.1.7 grid for ML-20M-time-split / Yelp: {32, 64, 128, 256, 512, 1024}; Cornac's best in their study was 512–1024.",
    },
    "use_bias": {
        "current": False,
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.3 (global temporal split): 'disabled item biases ... performed best on the MSD dataset'. Cornac default is True (Rendle's original).",
        "notes": "Was True until 2026-05-29; flipped to match NewBPR's global-temporal-split recipe.",
    },

    # --- Loss ---
    "loss": {
        "current": "BPR_pairwise",
        "status": "PAPER_FIXED",
        "source": "Rendle 2009 / NewBPR §3.2: pairwise log-likelihood −Σ log σ(score_ui − score_uj).",
    },

    # --- Optimizer ---
    "optimizer_kind": {
        "current": "sgd_manual",
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.3: 'standard SGD demonstrates commendable performance ... preferred optimizer for the BPR model on the global temporal split'. Rendle 2009 also SGD.",
        "notes": "We do manual SGD updates inside the training loop; not torch.optim.SGD.",
    },
    "sgd_momentum": {
        "current": 0.0,
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.3: 'standard SGD' (not Momentum SGD which they evaluated separately).",
    },

    # --- Regularization (NewBPR §5.3: separated λu, λi, λj) ---
    "regularization_scheme": {
        "current": "separated",
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.2/§5.3 finds three separate (λu, λi, λj) best for global temporal split.",
    },
    "lambda_u": {
        "current": 1e-4,
        "status": "PAPER_GRID",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "NewBPR §5.2: 'distinct regularization lambdas for users, positive and negative items, are crucial for specific datasets'. Range adapted from LightGCN §4.1.2 grid that NewBPR also references.",
    },
    "lambda_i": {
        "current": 1e-4,
        "status": "PAPER_GRID",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "NewBPR §5.2/§5.3 — separate λ for positive-item embeddings.",
    },
    "lambda_j": {
        "current": 1e-4,
        "status": "PAPER_GRID",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "NewBPR §5.2/§5.3 — separate λ for negative-item embeddings.",
    },

    # --- Weight init ---
    "weight_init_kind": {
        "current": "uniform_centered",
        "status": "CORNAC_DEFAULT",
        "source": "cornac/models/bpr/recom_bpr.pyx:149-151: '(uniform(0,1) - 0.5) / k'. Rendle 2009 does not specify.",
    },
    "weight_init_low": {
        "current": "-0.5/k",
        "status": "CORNAC_DEFAULT",
        "source": "cornac/models/bpr/recom_bpr.pyx:149.",
    },
    "weight_init_high": {
        "current": "0.5/k",
        "status": "CORNAC_DEFAULT",
        "source": "cornac/models/bpr/recom_bpr.pyx:149.",
    },

    # --- Tunable scalars ---
    "learning_rate": {
        "current": 0.05,
        "status": "EYEBALL",
        "source": "Claude. Cornac default lr=0.001 (paired with 100 iterations). 0.05 chosen for our 20-epoch budget; NewBPR tunes via Optuna without enumerated bounds.",
    },
    "batch_size": {
        "current": 16384,
        "status": "EYEBALL",
        "source": "Claude. GPU throughput choice. NewBPR (PyTorch) and Elliot's Batched BPRMF both batch; neither pins a value.",
    },

    # --- Negative sampling ---
    "sampler_kind": {
        "current": "uniform | causal",
        "status": "RESEARCH",
        "source": "Our causal mechanism. Unlike NeuMF/LightGCN this is a model argument, because cornac's BPR trains in compiled Cython over train_set.matrix and never asks the data loader for negatives — so our NumPy BPR carries its own sampler. NewBPR §5.3 recommends adaptive sampling for global temporal split; we have NOT implemented adaptive — orthogonal to our axis.",
    },

    # --- Training schedule ---
    "n_epochs": {
        "current": 1000,
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.2: '1000 epochs for the best model on the ML-20M dataset' — budget ceiling, with early stopping deciding the actual stop.",
    },
    "early_stop_min_delta": {
        "current": 0.0,
        "status": "EYEBALL",
        "source": "Claude. NewBPR doesn't quote a min_delta; 0.0 = strict 'any improvement'.",
    },
    "early_stop_patience": {
        "current": 13,
        "status": "PAPER_FIXED",
        "source": "NewBPR §5.2: 'training protocol employs the Early Stopping criterion with patience equal to 13 epochs.'",
    },
    "early_stop_check_every": {
        "current": 1,
        "status": "EYEBALL",
        "source": "Claude. Check every epoch so patience=13 means 13 epochs without improvement (NewBPR-faithful translation).",
    },
    "monitor_metric_early_stop": {
        "current": "NDCG@20",
        "status": "EYEBALL",
        "source": "Claude. NewBPR optimizes NDCG@100 via Optuna; we monitor NDCG@20 to align with our tuner selection metric and reported tables.",
    },
    "monitor_metric": {
        "current": "NDCG@100",
        "status": "PAPER_FIXED",
        "source": "NewBPR §4.1.6: 'We search for the best hyperparameters using NDCG@100 on all datasets except Netflix'.",
    },

}


# ===========================================================================
# Coordinate-descent tuning: knob order + grids (excluding the default value).
# ===========================================================================
# For each model, tune knobs in the listed order. The grid for each knob OMITS
# the current default so the baseline run is reused. See research/tune.py.

TUNE_ORDER = {
    "lightgcn": [
        # He 2020 §4.5: "the most important hyper-parameter to tune is λ".
        ("lambda_reg",    [1e-6, 1e-5, 1e-3, 1e-2]),     # default 1e-4
        ("num_layers",    [1, 2, 4]),                    # default 3
        ("learning_rate", [1e-4, 5e-4, 5e-3]),           # default 1e-3
    ],
    "neumf": [
        ("num_factors",      [16, 32, 64]),              # default 8
        ("mlp_hidden_count", [1, 2, 4]),                 # default 3 (skip 0 — degenerate tower)
        ("learning_rate",    [1e-4, 5e-4, 5e-3]),        # default 1e-3
        ("num_neg",          [3, 5, 6]),                 # default 4
        ("batch_size",       [128, 512, 1024]),          # default 256
    ],
    "bpr": [
        ("k_embed_dim",   [32, 128, 256, 512, 1024]),    # default 64
        ("learning_rate", [1e-3, 1e-2, 1e-1]),           # default 0.05
        # NewBPR §5.3: λu, λi, λj are tuned independently.
        ("lambda_u",      [1e-6, 1e-5, 1e-3, 1e-2]),     # default 1e-4
        ("lambda_i",      [1e-6, 1e-5, 1e-3, 1e-2]),     # default 1e-4
        ("lambda_j",      [1e-6, 1e-5, 1e-3, 1e-2]),     # default 1e-4
        # n_epochs is the budget ceiling (1000); early stopping picks the stop.
    ],
}


# Baseline config = the defaults that tune.py starts from. Keys match what
# tune.py passes through to each model's constructor. Stays in lockstep with
# the "current" values in the dicts above.
BASELINE_CONFIG = {
    "lightgcn": {
        "lambda_reg":    LIGHTGCN["lambda_reg"]["current"],
        "num_layers":    LIGHTGCN["num_layers"]["current"],
        "learning_rate": LIGHTGCN["learning_rate"]["current"],
    },
    "neumf": {
        "num_factors":      NEUMF["num_factors"]["current"],
        "mlp_hidden_count": NEUMF["mlp_hidden_count"]["current"],
        "learning_rate":    NEUMF["learning_rate"]["current"],
        "num_neg":          NEUMF["num_neg"]["current"],
        "batch_size":       NEUMF["batch_size"]["current"],
    },
    "bpr": {
        "k_embed_dim":   BPR["k_embed_dim"]["current"],
        "learning_rate": BPR["learning_rate"]["current"],
        "lambda_u":      BPR["lambda_u"]["current"],
        "lambda_i":      BPR["lambda_i"]["current"],
        "lambda_j":      BPR["lambda_j"]["current"],
        # n_epochs is the ceiling; we set it via BPR_EARLY_STOP, not tuned.
        "n_epochs":      BPR["n_epochs"]["current"],
    },
}


# Convenience config: pass these to BPR ctors to get the NewBPR-faithful
# early-stopping protocol.
BPR_EARLY_STOP = {
    "early_stopping": {
        "min_delta": BPR["early_stop_min_delta"]["current"],
        "patience":  BPR["early_stop_patience"]["current"],
    },
    "early_stop_every": BPR["early_stop_check_every"]["current"],
}


# Convenience config: pass these to NeuMF ctors for the He 2017 defaults, whose
# provenance is inventoried in NEUMF above. Lives here rather than in the runner
# because `runners/tuning.py` needs it too, and a runner importing from another
# runner just to reach a constant is the wrong direction.
#
# backend="pytorch" because cornac's default TF backend has no GPU support on
# native Windows; both backends take the same uir_iter path, so the ablation is
# unaffected by the choice.
NEUMF_KWARGS = dict(
    num_factors=NEUMF["num_factors"]["current"],
    layers=NEUMF["mlp_layers"]["current"],
    act_fn=NEUMF["act_fn"]["current"],
    num_epochs=NEUMF["n_epochs_neumf"]["current"],
    batch_size=NEUMF["batch_size"]["current"],
    lr=NEUMF["learning_rate"]["current"],
    num_neg=NEUMF["num_neg"]["current"],
    backend="pytorch",
)
