"""Phase 4a: sweep number of windows for the best (win+causal) variant.

K too small => barely different from shuffle (poor curriculum).
K too large => each window has too few rows to learn from.
We expect a Pareto sweet spot.
"""

import cornac
from cornac.metrics import NDCG, HitRatio, Recall

from ..lib.temporal_batching import BPRWindowed
from ..lib.causal_sampling import TOP_K, item_first_seen, future_items_pct
from ..lib.data import build_eval_method, load_uirt, dataset_path
from ..paths import logs_dir


def main():
    eval_method = build_eval_method("baby")
    rows = load_uirt(dataset_path("baby"))
    item_first = item_first_seen(rows)
    common = dict(k=64, batch_size=4096, learning_rate=0.05,
                  lambda_u=1e-4, lambda_i=1e-4, lambda_j=1e-4, seed=42, verbose=False)
    total_passes = 20  # match Phase 3 budget: per-row exposure = 20 across all K

    rows_summary = []
    for K in [2, 5, 10, 20, 50]:
        epw = max(1, total_passes // K) * K // K  # = epochs_per_window so K*epw = total per-row exposure (approx)
        # The clean choice: epochs_per_window = total_passes, giving K * total_passes effective
        # row-touches PER ROW within its window. To make per-row touches identical to shuffle
        # of total_passes epochs, we need epochs_per_window = total_passes (since each row is
        # in exactly one window). Confirmed correct.
        epw = total_passes
        name = f"win{K}+causal"
        print(f"\n========== {name} (epochs_per_window={epw}) ==========", flush=True)
        model = BPRWindowed(
            name=name, n_windows=K, epochs_per_window=epw, sampler="causal", **common
        )
        cornac.Experiment(
            eval_method=eval_method,
            models=[model],
            metrics=[HitRatio(k=TOP_K), NDCG(k=TOP_K), Recall(k=TOP_K)],
            user_based=True,
            save_dir=logs_dir(),
        ).run()
        leak = future_items_pct(model, eval_method, item_first, k=TOP_K)
        rows_summary.append((K, model))
        print(f"[{name}] future_items: {leak}", flush=True)


if __name__ == "__main__":
    main()
