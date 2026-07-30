"""Smoke test the GPU BPR — train 2 epochs on baby, check counterfactual rate.

Usage:  python -m research.smoke.smoke_gpu
"""
import time

import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", flush=True)

from ..lib.bpr_gpu import BPRMiniBatchGPU
from ..lib.data import build_eval_method

ts = build_eval_method("baby")
print(f"train rows: {ts.train_set.num_ratings}", flush=True)

for sampler in ("uniform", "causal"):
    t0 = time.time()
    model = BPRMiniBatchGPU(
        name=f"smoke-gpu-{sampler}", k=32, n_epochs=2,
        batch_size=16384, learning_rate=0.05,
        lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4,
        sampler=sampler, seed=42, verbose=False,
    )
    model.fit(ts.train_set)
    elapsed = time.time() - t0
    print(f"{sampler}: cf_rate={model.counterfactual_rate*100:.2f}% in {elapsed:.1f}s", flush=True)
