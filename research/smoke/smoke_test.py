"""Smoke test: confirm BPRMiniBatch runs at all and benchmark one epoch.

Usage:  python -m research.smoke.smoke_test
"""
import time

from ..lib.bpr_cpu import BPRMiniBatch
from ..lib.data import build_eval_method

print("Building split...", flush=True)
ts = build_eval_method("baby", verbose=True)
print(f"Train rows: {ts.train_set.num_ratings}", flush=True)

print("Training 2 epochs...", flush=True)
t0 = time.time()
model = BPRMiniBatch(
    name="smoke",
    k=64,
    n_epochs=2,
    batch_size=4096,
    learning_rate=0.05,
    lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
    sampler="uniform",
    seed=42,
    verbose=True,
)
model.fit(ts.train_set)
print(f"DONE in {time.time()-t0:.1f}s", flush=True)
