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

import json
import os

from ..paths import RESULTS_DIR


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
        "status": "PAPER_FIXED",
        "source": "He 2020 §4.1.2: 'L2 regularization coefficient λ is searched in the range of {1e-6, 1e-5, ..., 1e-2}, and in most cases the optimal value is 1e-4'",
        "notes": "Adopted, not re-searched. The paper ran this grid and published 1e-4 as the answer; repeating it on our most expensive model would only reproduce a known result. Was PAPER_GRID and swept by TUNE_ORDER until 2026-09-09 -- dropping it removed 4 of LightGCN's 10 trials. He 2020 §4.5 does call λ 'the most important hyper-parameter to tune', so revisit if LightGCN underperforms on a dataset unlike theirs.",
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
        "current": 50,
        "status": "EYEBALL",
        "source": "Claude. Paper does not quote a patience value.",
        "notes": "50 epochs without a recall@20 improvement. cornac's LightGCN evaluates every epoch and exposes no check-interval knob, so patience is counted in epochs directly. Recorded as 5 until 2026-09-09, paired with an unimplementable check_every=10 that multiplied out to the same 50; the runner always passed 50.",
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
        "notes": "TUNE_ORDER caps the sweep at 256, so 512 and 1024 -- where NewBPR reports cornac's best -- are NOT searched. A documented ceiling, not a completed search. k is swept first under coordinate descent, so its winner sets the per-trial cost of every later BPR trial; capping bounds those at 4x the k=64 baseline instead of 16x, which is where the saving comes from.",
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
    "lambda_shared": {
        "current": 1e-4,
        "status": "EYEBALL",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "Claude. One lambda tied across lambda_u/lambda_i/lambda_j, which is what TUNE_ORDER actually sweeps.",
        "notes": "NewBPR §5.2's claim is that *separated* lambdas matter -- a claim about how the three interact. Coordinate descent cannot test it: it tunes each with the other two pinned, so it never visits the joint optimum the paper is describing. Sweeping all three that way cost 12 of BPR's 18 trials and answered a different question than the one asked. Tuning one tied lambda is the honest reduction; a real test of NewBPR's claim needs a joint search (random/Optuna), which is a separate piece of work. `_build_model` fans this value out to all three constructor arguments.",
    },
    "lambda_u": {
        "current": 1e-4,
        "status": "PAPER_GRID",
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        "source": "NewBPR §5.2: 'distinct regularization lambdas for users, positive and negative items, are crucial for specific datasets'. Range adapted from LightGCN §4.1.2 grid that NewBPR also references.",
        "notes": "Tied to lambda_i/lambda_j during tuning -- see lambda_shared. The separated form is still what BPR_KWARGS passes and what the model supports.",
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
        "current": 4096,
        "status": "EYEBALL",
        "source": "Claude. Throughput choice for the pure-NumPy BPR in lib/bpr_cpu.py. NewBPR (PyTorch) and Elliot's Batched BPRMF both batch; neither pins a value.",
        "notes": "Read by BPR_KWARGS and BASELINE_CONFIG below, so this value is what actually runs. Recorded as 16384 until 2026-09-09 — a leftover from a GPU backend that no longer exists, while every call site passed 4096.",
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

# Trimmed 2026-09-09 from 48 to 26 trials per (dataset, arm) cell so the full
# 3-dataset x 2-arm sweep is ~4 days rather than ~9. Every knob dropped below is
# one the source paper already answers; each entry's "notes" in the inventory
# above records which sentence, so the omissions are citable rather than silent.
TUNE_ORDER = {
    "lightgcn": [
        # lambda_reg is NOT swept: He 2020 §4.1.2 ran exactly this grid and
        # published 1e-4, which is already the default -- see LIGHTGCN["lambda_reg"].
        # learning_rate is NOT swept: PAPER_DEFAULT, He 2020 does not search it.
        ("num_layers",    [1, 2, 4]),                    # default 3
    ],
    "neumf": [
        # num_factors is the one that matters: 8 against BPR/LightGCN's 64 is the
        # likeliest reason NeuMF scores ~1/3 of LightGCN's HR@20.
        ("num_factors",      [16, 32, 64]),              # default 8
        ("mlp_hidden_count", [4]),                       # default 3; He 2017 §4.4: "MLP-4 generally best"
        ("learning_rate",    [1e-4, 5e-4, 5e-3]),        # default 1e-3
        # num_neg is NOT swept: He 2017 Fig 7 puts the optimum at 3-6 and the
        # default of 4 is already inside that band, while cost scales with it.
        ("batch_size",       [128, 512, 1024]),          # default 256
    ],
    "bpr": [
        # Capped at 256; 512/1024 (NewBPR's best for cornac) are not searched.
        # k is swept first, so its winner sets the cost of every later BPR trial.
        ("k_embed_dim",   [32, 128, 256]),               # default 64
        ("learning_rate", [1e-3, 1e-2, 1e-1]),           # default 0.05
        # One tied lambda, not three: coordinate descent cannot test NewBPR
        # §5.2's *interaction* claim. See BPR["lambda_shared"].
        ("lambda_shared", [1e-6, 1e-5, 1e-3, 1e-2]),     # default 1e-4
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
        # One tied value; _build_model fans it out to lambda_u/i/j.
        "lambda_shared": BPR["lambda_shared"]["current"],
        "batch_size":    BPR["batch_size"]["current"],
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


# Convenience config: the architecture/optimizer knobs `runners/ablation_bpr.py`
# passes to BPRMiniBatch, read off the inventory above so the runner cannot drift
# from it -- `batch_size` was recorded here as 16384 for a while whilst every call
# site passed 4096, which is exactly what this closes. Pair with BPR_EARLY_STOP;
# `sampler`, `seed`, `name`, and `verbose` stay per-call.
#
# n_epochs is the NewBPR §5.2 budget ceiling (1000), not a target: early stopping
# decides the actual stop epoch, which is data-dependent.
BPR_KWARGS = dict(
    k=BPR["k_embed_dim"]["current"],
    batch_size=BPR["batch_size"]["current"],
    learning_rate=BPR["learning_rate"]["current"],
    lambda_u=BPR["lambda_u"]["current"],
    lambda_i=BPR["lambda_i"]["current"],
    lambda_j=BPR["lambda_j"]["current"],
    n_epochs=BPR["n_epochs"]["current"],
)


# Convenience config: the LightGCN knobs both runners need, derived from the
# LIGHTGCN inventory above. Lives here rather than in `runners/ablation_lightgcn.py`
# for the same reason NEUMF_KWARGS does -- `runners/tuning.py` needs them too, and
# a runner importing from another runner just to reach a constant is the wrong
# direction.
#
# cornac's LightGCN evaluates every epoch and takes no check-interval argument, so
# `patience` here is counted in epochs directly.
LIGHTGCN_BATCH = {
    "musical":    LIGHTGCN["batch_size_musical"]["current"],
    "baby":       LIGHTGCN["batch_size_baby"]["current"],
    "cellphone":  LIGHTGCN["batch_size_cellphone"]["current"],
    "healthcare": LIGHTGCN["batch_size_healthcare"]["current"],
}
LIGHTGCN_EPOCHS = LIGHTGCN["n_epochs_max"]["current"]
LIGHTGCN_EARLY_STOP = {
    "min_delta": LIGHTGCN["early_stop_min_delta"]["current"],
    "patience":  LIGHTGCN["early_stop_patience"]["current"],
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


# ===========================================================================
# Applying tuned winners
# ===========================================================================
#
# `runners/tuning.py` writes one winner per (model, dataset, arm, seed) to
#   results/tuning/<model>/<dataset>/<recipe>/seed<n>/winner.json
# The helpers below read those back so the ablation runners can use them, and
# fall back to the `"current"` values above whenever a cell has not been tuned --
# so the runners work identically before and after a sweep.
#
# WHICH ARM'S WINNERS TO APPLY is a methodological choice, not a detail:
#
#   "uniform" (default) -- both arms run at the *uniform* arm's hyperparameters.
#       The arms then differ in exactly one thing, the negative-item pool, so any
#       accuracy difference is attributable to the sampler. This is the ablation
#       the paper's claim rests on.
#
#   "per_arm" -- each arm runs at its own winners. A legitimate question ("which
#       method is better when each is deployed properly?") but a different one:
#       two things now differ, so the comparison is a benchmark, not an ablation.
#       Report it as a separate table; never merge the two.
#
# Selected by RESEARCH_TUNED_ARM so the secondary experiment needs no code edit.

TUNED_ARM = os.environ.get("RESEARCH_TUNED_ARM", "uniform")
TUNING_SEED = 42


def _tuning_source(recipe: str) -> str:
    """Which arm's winners apply to a cell running `recipe`. See TUNED_ARM."""
    return recipe if TUNED_ARM == "per_arm" else TUNED_ARM


def winner_config(model: str, dataset: str, recipe: str, seed: int = TUNING_SEED):
    """Tuned config for one cell, or None if it was never tuned.

    Returns the `config` dict `runners/tuning.py` wrote, which carries every key
    in that model's BASELINE_CONFIG -- including knobs the sweep did not touch,
    since coordinate descent starts from the baseline and only overwrites what it
    sweeps. Callers can therefore read any baseline key off it unconditionally.
    """
    path = (RESULTS_DIR / "tuning" / model / dataset / _tuning_source(recipe)
            / f"seed{seed}" / "winner.json")
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)["config"]


def neumf_layers(num_factors: int, hidden: int) -> tuple:
    """He 2017 §3.3 tower-halving, with layers[-1] == num_factors.

    hidden=3, num_factors=8 -> (64, 32, 16, 8). Shared by the tuner and the
    ablation runner so a tuned `mlp_hidden_count` means the same thing in both.
    """
    return tuple(num_factors * (2 ** i) for i in range(hidden, -1, -1))


def bpr_kwargs(dataset: str, recipe: str) -> dict:
    """BPRMiniBatch ctor kwargs, tuned if available. Pair with BPR_EARLY_STOP."""
    w = winner_config("bpr", dataset, recipe)
    if w is None:
        return dict(BPR_KWARGS)
    lam = float(w["lambda_shared"])
    return dict(
        k=int(w["k_embed_dim"]),
        batch_size=int(w["batch_size"]),
        learning_rate=float(w["learning_rate"]),
        lambda_u=lam, lambda_i=lam, lambda_j=lam,
        n_epochs=int(w["n_epochs"]),
    )


def neumf_kwargs(dataset: str, recipe: str) -> dict:
    """cornac NeuMF ctor kwargs, tuned if available."""
    w = winner_config("neumf", dataset, recipe)
    if w is None:
        return dict(NEUMF_KWARGS)
    num_factors = int(w["num_factors"])
    kwargs = dict(NEUMF_KWARGS)
    kwargs.update(
        num_factors=num_factors,
        layers=neumf_layers(num_factors, int(w["mlp_hidden_count"])),
        batch_size=int(w["batch_size"]),
        lr=float(w["learning_rate"]),
        num_neg=int(w["num_neg"]),
    )
    return kwargs


def lightgcn_kwargs(dataset: str, recipe: str) -> dict:
    """cornac LightGCN ctor kwargs, tuned if available.

    batch_size is per-dataset from the inventory and is not a tuned knob, so it
    comes from LIGHTGCN_BATCH either way.
    """
    base = dict(
        emb_size=LIGHTGCN["emb_size"]["current"],
        num_layers=LIGHTGCN["num_layers"]["current"],
        learning_rate=LIGHTGCN["learning_rate"]["current"],
        lambda_reg=LIGHTGCN["lambda_reg"]["current"],
    )
    w = winner_config("lightgcn", dataset, recipe)
    if w is not None:
        base.update(
            num_layers=int(w["num_layers"]),
            learning_rate=float(w["learning_rate"]),
            lambda_reg=float(w["lambda_reg"]),
        )
    base.update(
        batch_size=LIGHTGCN_BATCH.get(dataset, 1024),
        num_epochs=LIGHTGCN_EPOCHS,
        early_stopping=LIGHTGCN_EARLY_STOP,
    )
    return base
