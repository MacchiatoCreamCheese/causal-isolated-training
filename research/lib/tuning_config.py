import json
import os

from ..paths import RESULTS_DIR


LIGHTGCN = {
    "emb_size": {
        "current": 64,
    },
    "num_layers": {
        "current": 3,
        "grid": [1, 2, 3, 4],
    },
    "alpha_k_formula": {
        "current": "1/(K+1)",
    },
    "normalization_kind": {
        "current": "symmetric_sqrt",
    },
    "self_connection": {
        "current": False,
    },
    "dropout_p": {
        "current": 0.0,
    },

    "loss": {
        "current": "BPR",
    },

    "optimizer_kind": {
        "current": "adam",
    },
    "adam_beta1": {
        "current": 0.9,
    },
    "adam_beta2": {
        "current": 0.999,
    },
    "adam_eps": {
        "current": 1e-8,
    },

    "weight_init_kind": {
        "current": "xavier_uniform",
    },
    "xavier_gain": {
        "current": 1.0,
    },

    "learning_rate": {
        "current": 1e-3,
    },
    "lambda_reg": {
        "current": 1e-4,
    },
    "batch_size_musical": {
        "current": 1024,
    },
    "batch_size_baby": {
        "current": 1024,
    },
    "batch_size_cellphone": {
        "current": 512,
    },
    "batch_size_healthcare": {
        "current": 2048,
    },

    "n_epochs_max": {
        "current": 1000,
    },
    "early_stop_min_delta": {
        "current": 0.0,
    },
    "early_stop_patience": {
        "current": 50,
    },
    "monitor_metric": {
        "current": "recall@20",
    },

    "sampler": {
        "current": "uniform | causal",
    },
}


NEUMF = {
    "num_factors": {
        "current": 8,
        "grid": [8, 16, 32, 64],
    },
    "mlp_layers": {
        "current": (64, 32, 16, 8),
    },
    "mlp_hidden_count": {
        "current": 3,
        "grid": [0, 1, 2, 3, 4],
    },
    "act_fn": {
        "current": "relu",
    },
    "layer_pattern": {
        "current": "tower_halving",
    },

    "pretrain": {
        "current": False,
    },
    "pretrain_epochs": {
        "current": 20,
    },
    "alpha_mixing": {
        "current": 0.5,
    },

    "loss": {
        "current": "BCE",
    },

    "optimizer_pretrain_kind": {
        "current": "adam",
    },
    "optimizer_pretrain_adam_beta1": {
        "current": 0.9,
    },
    "optimizer_pretrain_adam_beta2": {
        "current": 0.999,
    },
    "optimizer_pretrain_adam_eps": {
        "current": 1e-8,
    },
    "optimizer_finetune_kind": {
        "current": "sgd",
    },
    "optimizer_finetune_sgd_momentum": {
        "current": 0.0,
    },
    "optimizer_finetune_sgd_nesterov": {
        "current": False,
    },
    "weight_decay": {
        "current": 0.0,
    },

    "weight_init_kind": {
        "current": "gaussian",
    },
    "weight_init_mean": {
        "current": 0.0,
    },
    "weight_init_std": {
        "current": 0.01,
    },

    "batch_size": {
        "current": 256,
        "grid": [128, 256, 512, 1024],
    },
    "learning_rate": {
        "current": 1e-3,
        "grid": [1e-4, 5e-4, 1e-3, 5e-3],
    },
    "num_neg": {
        "current": 4,
        "grid": [3, 4, 5, 6],
    },

    "n_epochs_neumf": {
        "current": 20,
    },
    "early_stop_min_delta": {
        "current": None,
    },
    "early_stop_patience": {
        "current": None,
    },
    "monitor_metric": {
        "current": "HR@10",
    },

    "sampler": {
        "current": "uniform | causal",
    },
}


BPR = {
    "k_embed_dim": {
        "current": 64,
    },
    "use_bias": {
        "current": False,
    },

    "loss": {
        "current": "BPR_pairwise",
    },

    "optimizer_kind": {
        "current": "sgd_manual",
    },
    "sgd_momentum": {
        "current": 0.0,
    },

    "regularization_scheme": {
        "current": "separated",
    },
    "lambda_shared": {
        "current": 1e-4,
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },
    "lambda_u": {
        "current": 1e-4,
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },
    "lambda_i": {
        "current": 1e-4,
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },
    "lambda_j": {
        "current": 1e-4,
        "grid": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    },

    "weight_init_kind": {
        "current": "uniform_centered",
    },
    "weight_init_low": {
        "current": "-0.5/k",
    },
    "weight_init_high": {
        "current": "0.5/k",
    },

    "learning_rate": {
        "current": 0.05,
    },
    "batch_size": {
        "current": 4096,
    },

    "sampler_kind": {
        "current": "uniform | causal",
    },

    "n_epochs": {
        "current": 1000,
    },
    "early_stop_min_delta": {
        "current": 0.0,
    },
    "early_stop_patience": {
        "current": 13,
    },
    "early_stop_check_every": {
        "current": 1,
    },
    "monitor_metric_early_stop": {
        "current": "NDCG@20",
    },
    "monitor_metric": {
        "current": "NDCG@100",
    },

}


TUNE_ORDER = {
    "lightgcn": [
        ("num_layers",    [1, 2, 4]),
    ],
    "neumf": [
        ("num_factors",      [16, 32, 64]),
        ("mlp_hidden_count", [4]),
        ("learning_rate",    [1e-4, 5e-4, 5e-3]),
        ("batch_size",       [128, 512, 1024]),
    ],
    "bpr": [
        ("k_embed_dim",   [32, 128, 256]),
        ("learning_rate", [1e-3, 1e-2, 1e-1]),
        ("lambda_shared", [1e-6, 1e-5, 1e-3, 1e-2]),
    ],
}


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
        "lambda_shared": BPR["lambda_shared"]["current"],
        "batch_size":    BPR["batch_size"]["current"],
        "n_epochs":      BPR["n_epochs"]["current"],
        "early_stopping": "off",
    },
}


BPR_EARLY_STOP = {}


BPR_KWARGS = dict(
    k=BPR["k_embed_dim"]["current"],
    batch_size=BPR["batch_size"]["current"],
    learning_rate=BPR["learning_rate"]["current"],
    lambda_u=BPR["lambda_u"]["current"],
    lambda_i=BPR["lambda_i"]["current"],
    lambda_j=BPR["lambda_j"]["current"],
    n_epochs=BPR["n_epochs"]["current"],
)


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


NEUMF_PRETRAIN = os.environ.get("RESEARCH_NEUMF_PRETRAIN", "") == "1"

NEUMF_KWARGS = dict(
    pretrain=NEUMF_PRETRAIN,
    learner="sgd" if NEUMF_PRETRAIN else "adam",
    num_factors=NEUMF["num_factors"]["current"],
    layers=NEUMF["mlp_layers"]["current"],
    act_fn=NEUMF["act_fn"]["current"],
    num_epochs=NEUMF["n_epochs_neumf"]["current"],
    batch_size=NEUMF["batch_size"]["current"],
    lr=NEUMF["learning_rate"]["current"],
    num_neg=NEUMF["num_neg"]["current"],
    backend="pytorch",
)


TUNED_ARM = os.environ.get("RESEARCH_TUNED_ARM", "uniform")


def _variant_parts(model: str) -> list:
    parts = []
    if model.lower().startswith("neumf") and NEUMF_PRETRAIN:
        parts.append("pretrain")
    return parts


def tuning_model_dir(model: str) -> str:
    return "-".join([model] + _variant_parts(model))


def ablation_label(model: str) -> str:
    parts = _variant_parts(model)
    if TUNED_ARM != "uniform":
        parts.append("tuned" + TUNED_ARM.replace("_", "-"))
    return "-".join([model] + parts)


def _tuning_source(recipe: str) -> str:
    return recipe if TUNED_ARM == "per_arm" else TUNED_ARM


_WARNED = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(msg, flush=True)


def winner_dir(model: str, dataset: str, arm: str, seed: int):
    return RESULTS_DIR / "tuning" / tuning_model_dir(model) / dataset / arm / f"seed{seed}"


def winner_config(model: str, dataset: str, recipe: str, seed: int):
    path = winner_dir(model, dataset, _tuning_source(recipe), seed) / "winner.json"
    if not path.exists():
        _warn_once(f"[warn] no tuned winner at {path}; using defaults")
        return None
    with open(path, encoding="utf-8-sig") as f:
        payload = json.load(f)
    if payload.get("select_split") != "validation":
        _warn_once(f"[warn] ignoring {path}: selected on the test set; using "
                   f"defaults until this cell is re-tuned")
        return None
    return payload["config"]


def neumf_layers(num_factors: int, hidden: int) -> tuple:
    return tuple(num_factors * (2 ** i) for i in range(hidden, -1, -1))


def bpr_kwargs(dataset: str, recipe: str, seed: int) -> dict:
    w = winner_config("bpr", dataset, recipe, seed)
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


def neumf_kwargs(dataset: str, recipe: str, seed: int) -> dict:
    w = winner_config("neumf", dataset, recipe, seed)
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


def lightgcn_kwargs(dataset: str, recipe: str, seed: int) -> dict:
    base = dict(
        emb_size=LIGHTGCN["emb_size"]["current"],
        num_layers=LIGHTGCN["num_layers"]["current"],
        learning_rate=LIGHTGCN["learning_rate"]["current"],
        lambda_reg=LIGHTGCN["lambda_reg"]["current"],
    )
    w = winner_config("lightgcn", dataset, recipe, seed)
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
