"""Smoke test: verify the counterfactual-negative-rate counter.

Usage:  python -m research.smoke.smoke_counter
"""
from ..lib.bpr_cpu import BPRMiniBatch
from ..lib.data import build_eval_method

ts = build_eval_method("baby")

for sampler in ("uniform", "causal"):
    model = BPRMiniBatch(
        name=f"smoke-{sampler}", k=32, n_epochs=2, batch_size=4096,
        learning_rate=0.05,
        lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
        sampler=sampler, seed=42, verbose=False,
    )
    model.fit(ts.train_set)
    s = model._sampler
    print(f"{sampler}: counterfactual_rate={model.counterfactual_rate*100:.2f}%  "
          f"(total negs={s._total_negs:,}, counterfactual={s._counterfactual_negs:,})", flush=True)
