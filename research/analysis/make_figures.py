"""Generate every figure in the paper from existing JSON checkpoints +
hard-coded constants for the phases that ran before per-cell JSON
checkpointing existed.

Outputs to research/figures/ as PDF (for LaTeX inclusion) and PNG (for
quick previewing). One function per figure; a top-level dispatcher
generates all of them.

Run: python -m research.analysis.make_figures [fig_name ...]
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

from ..paths import PROJECT_ROOT as ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR

FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Styling — IEEEtran-friendly defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
})

# Colors — ColorBrewer set1, grayscale-distinguishable
COLOR_VANILLA = "#e41a1c"   # red
COLOR_FAITHFUL = "#377eb8"  # blue
COLOR_DATASETS = {
    "Baby Products":          "#4daf4a",  # green
    "Cell Phones & Acc.":     "#984ea3",  # purple
    "Health & Household":     "#ff7f00",  # orange
}
COLOR_CELLS = {
    "shuffle+uniform": "#a6cee3",
    "shuffle+causal":  "#1f78b4",
    "win10+uniform":   "#b2df8a",
    "win10+causal":    "#33a02c",
}

SINGLE_COL = (3.4, 2.2)
DOUBLE_COL = (7.0, 2.6)


def save(fig, name):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(ROOT)} + {png.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Hard-coded constants from research/RESULTS_phase*.md
# (phases that pre-date per-cell JSON checkpointing)
# ---------------------------------------------------------------------------

# Phase 4a — K ablation on baby, win+causal (RESULTS_phase4.md)
PHASE4A_K = [2, 5, 10, 20, 50]
PHASE4A_HR = [0.0329, 0.0466, 0.0550, 0.0491, 0.0518]

# Three-model ablation on Baby Products (Tables III, III-B, III-C in sample.tex).
# Per-cell HR@20 at fixed seed=42 under TimestampSplit.
ABLATION_BABY_HR = {
    "BPR": {
        "shuffle+uniform": 0.0208, "shuffle+causal": 0.0314,
        "win10+uniform":   0.0482, "win10+causal":   0.0550,
    },
    "NeuMF": {
        "shuffle+uniform": 0.0262, "shuffle+causal": 0.0260,
        "win10+uniform":   0.0391, "win10+causal":   0.0365,
    },
    "LightGCN": {
        "shuffle+uniform": 0.0278, "shuffle+causal": 0.0297,
        "win10+uniform":   0.0361, "win10+causal":   0.0357,
    },
}

# Map filenames to display labels for prequential figures
DATASET_FILES = {
    "Baby Products":          "preq_Baby_Products_seed42.json",
    "Cell Phones & Acc.":     "preq_Cell_Phones_and_Accessories_gpu_seed42.json",
    "Health & Household":     "preq_Health_and_Household_gpu_seed42.json",
}


def load_preq(name):
    """Load one prequential JSON payload as a dict keyed by cell."""
    path = RESULTS_DIR / DATASET_FILES[name]
    with open(path) as f:
        return json.load(f)


def slice_array(payload, cell_name, key):
    """Extract per-slice values for a metric or counterfactual rate or
    recency stat. `key` is either a metrics key like 'HitRatio@20', the
    string 'counterfactual_rate', or a recency-stat name like
    'recency_days_median'."""
    cells = sorted(payload["cells"][cell_name], key=lambda c: c["slice"])
    out = []
    for c in cells:
        if key in ("counterfactual_rate",):
            out.append(c["counterfactual_rate"])
        elif key.startswith("recency_"):
            out.append(c["recency"].get(key, np.nan))
        else:
            out.append(c["metrics"][key])
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_temporal_batching():
    """Fig. 2 — conceptual schematic of K-windowed training."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    K = 10
    for k in range(K):
        ax.add_patch(mpatches.Rectangle((k, 0.7), 0.9, 0.4,
                                        facecolor="#cccccc", edgecolor="black", lw=0.5))
        ax.text(k + 0.45, 0.9, f"$W_{{{k+1}}}$", ha="center", va="center", fontsize=6.5)
    # arrow showing time
    ax.annotate("", xy=(K + 0.1, 0.45), xytext=(-0.1, 0.45),
                arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.text(K / 2, 0.25, "time $t$", ha="center", fontsize=7.5)
    # warm-start chain below
    for k in range(K):
        ax.add_patch(mpatches.Circle((k + 0.45, -0.05), 0.12,
                                     facecolor=COLOR_FAITHFUL, edgecolor="black", lw=0.4))
    for k in range(K - 1):
        ax.annotate("", xy=(k + 1 + 0.33, -0.05), xytext=(k + 0.57, -0.05),
                    arrowprops=dict(arrowstyle="->", lw=0.6, color="#444"))
    ax.text(K / 2, -0.45, r"warm-started parameters $\theta$", ha="center", fontsize=7)

    ax.set_xlim(-0.5, K + 0.5)
    ax.set_ylim(-0.7, 1.4)
    ax.set_axis_off()
    save(fig, "temporal_batching")


def fig_prequential_protocol():
    """Fig. 3 — conceptual schematic of prequential evaluation."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    N = 6  # 5 prequential test slices, plus warmup S_0
    for step in range(1, N):
        y = N - 1 - step
        # Step label on left
        ax.text(-0.7, y + 0.5, f"step {step}", ha="right", va="center", fontsize=7)
        for s in range(N):
            if s < step:
                color, label = "#377eb8", "train"
            elif s == step:
                color, label = "#e41a1c", "test"
            else:
                color, label = "#dddddd", ""
            ax.add_patch(mpatches.Rectangle((s, y), 0.92, 0.85,
                                            facecolor=color, edgecolor="black", lw=0.4))
            ax.text(s + 0.46, y + 0.42, f"$S_{{{s}}}$", ha="center", va="center", fontsize=6.5)

    # Legend patches
    legend_handles = [
        mpatches.Patch(facecolor="#377eb8", edgecolor="black", label="train"),
        mpatches.Patch(facecolor="#e41a1c", edgecolor="black", label="test"),
        mpatches.Patch(facecolor="#dddddd", edgecolor="black", label="unused"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    ax.set_xlim(-2.2, N + 0.2)
    ax.set_ylim(-0.4, N - 1 + 0.4)
    ax.set_axis_off()
    save(fig, "prequential_protocol")


def fig_k_ablation():
    """Fig. 4 — K ablation on baby (win+causal)."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    ax.plot(PHASE4A_K, PHASE4A_HR, marker="o", color=COLOR_FAITHFUL)
    # Highlight K=10
    k_best = 10
    idx = PHASE4A_K.index(k_best)
    ax.plot(k_best, PHASE4A_HR[idx], marker="o", markersize=8,
            markerfacecolor="none", markeredgecolor=COLOR_VANILLA, markeredgewidth=1.5)
    ax.annotate("Pareto sweet spot",
                xy=(k_best, PHASE4A_HR[idx]),
                xytext=(k_best * 1.6, PHASE4A_HR[idx] - 0.004),
                fontsize=7, color=COLOR_VANILLA,
                arrowprops=dict(arrowstyle="->", color=COLOR_VANILLA, lw=0.8))
    ax.set_xscale("log")
    ax.set_xticks(PHASE4A_K)
    ax.set_xticklabels([str(k) for k in PHASE4A_K])
    ax.set_xlabel("Number of windows $K$")
    ax.set_ylabel("HR@20")
    save(fig, "k_ablation")


def fig_prequential_hr():
    """Fig. 5 — prequential HR@20 per slice across 3 datasets."""
    datasets = list(DATASET_FILES.keys())
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COL, sharex=True)
    for ax, name in zip(axes, datasets):
        payload = load_preq(name)
        vanilla = slice_array(payload, "shuffle+uniform", "HitRatio@20")
        faithful = slice_array(payload, "win10+causal", "HitRatio@20")
        slices = np.arange(1, len(vanilla) + 1)
        ax.plot(slices, vanilla, marker="o", color=COLOR_VANILLA,
                linestyle="-",  label="vanilla")
        ax.plot(slices, faithful, marker="s", color=COLOR_FAITHFUL,
                linestyle="--", label="faithful")
        ax.set_title(name, fontsize=8)
        ax.set_xlabel("prequential slice")
        ax.set_xticks(slices)
    axes[0].set_ylabel("HR@20")
    axes[-1].legend(loc="upper right", frameon=False)
    save(fig, "prequential_hr")


def fig_counterfactual_rate():
    """Fig. 6 — counterfactual-negative rate growth (vanilla only)."""
    fig, ax = plt.subplots(figsize=SINGLE_COL)
    for name, color in COLOR_DATASETS.items():
        payload = load_preq(name)
        rho = slice_array(payload, "shuffle+uniform", "counterfactual_rate") * 100
        slices = np.arange(1, len(rho) + 1)
        ax.plot(slices, rho, marker="o", color=color, label=name)
    ax.set_xlabel("prequential slice")
    ax.set_ylabel(r"counterfactual rate $\rho$ (\%)")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0, 50)
    ax.legend(loc="lower right", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.97, "faithful: $\\rho \\equiv 0\\%$",
            transform=ax.transAxes, fontsize=7,
            color=COLOR_FAITHFUL, va="top")
    save(fig, "counterfactual_rate")


def fig_three_model_ablation_baby():
    """Legacy single-seed Baby-only figure kept for backward compatibility."""
    models = ["BPR", "NeuMF", "LightGCN"]
    cells = ["shuffle+uniform", "shuffle+causal", "win10+uniform", "win10+causal"]
    fig, axes = plt.subplots(1, 3, figsize=DOUBLE_COL, sharey=False)
    for ax, model in zip(axes, models):
        vals = [ABLATION_BABY_HR[model][c] for c in cells]
        x = np.arange(len(cells))
        ax.bar(x, vals, width=0.7,
               color=[COLOR_CELLS[c] for c in cells],
               edgecolor="black", lw=0.4)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.02, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_title(model, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["sh+un", "sh+ca", "w10+un", "w10+ca"],
                           rotation=0, fontsize=6.5)
        ax.set_ylim(0, max(vals) * 1.20)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    axes[0].set_ylabel("HR@20")
    save(fig, "three_model_ablation_baby")


def fig_three_model_ablation():
    """Multi-seed three-model 2x2 ablation across all datasets.

    Reads `research/results/2x2_summary.csv` (produced by aggregate_2x2.py).
    Layout: rows = datasets, columns = models, bars = 4 recipes per panel
    with mean +/- std error bars. Cells with no seeds are skipped.
    """
    import csv as _csv
    summary = RESULTS_DIR / "2x2_summary.csv"
    if not summary.exists():
        print(f"  no {summary}, skipping")
        return
    # {(model, dataset, recipe, metric): (mean, std, n)}
    data = {}
    with open(summary, encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            data[(row["model"], row["dataset"], row["recipe"], row["metric"])] = (
                float(row["mean"]), float(row["std"]), int(row["n"]))

    models = ["BPR", "NeuMF", "LightGCN"]
    datasets = [("baby", "Baby Products"), ("cellphone", "Cell Phones & Acc."),
                ("healthcare", "Health & Household")]
    cells = ["shuffle+uniform", "shuffle+causal", "win10+uniform", "win10+causal"]
    short = ["sh+un", "sh+ca", "w10+un", "w10+ca"]

    fig, axes = plt.subplots(len(datasets), len(models),
                             figsize=(7.0, 6.0), sharex=False)
    for r, (ds_key, ds_label) in enumerate(datasets):
        for c, model in enumerate(models):
            ax = axes[r][c]
            means, stds = [], []
            for recipe in cells:
                m, s, n = data.get((model, ds_key, recipe, "HR@20"), (0.0, 0.0, 0))
                means.append(m if n > 0 else 0.0)
                stds.append(s if n > 0 else 0.0)
            x = np.arange(len(cells))
            ax.bar(x, means, width=0.7,
                   yerr=stds, capsize=2,
                   color=[COLOR_CELLS[k] for k in cells],
                   edgecolor="black", lw=0.4,
                   error_kw=dict(elinewidth=0.6, ecolor="black"))
            ymax = max(means) if max(means) > 0 else 1.0
            for xi, v, s in zip(x, means, stds):
                if v > 0:
                    ax.text(xi, v + s + ymax * 0.04, f"{v:.4f}",
                            ha="center", va="bottom", fontsize=5.5)
                else:
                    ax.text(xi, ymax * 0.5, "TBD", ha="center", va="center",
                            fontsize=6.5, color="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(short, rotation=0, fontsize=6.0)
            ax.set_ylim(0, ymax * 1.35 if ymax > 0 else 1.0)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            if r == 0:
                ax.set_title(model, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{ds_label}\nHR@20", fontsize=7)
    fig.tight_layout()
    save(fig, "three_model_ablation")


def _load_recency_arrays_for_baby_bpr():
    """Used by fig_recency_cdf. Returns dict[slice] -> (vanilla_arr, faithful_arr)
    of raw recency days, derived from pickled BPR models in
    research/results/runs/Baby_Products_cpu_seed42/."""
    import pickle
    from cornac.eval_methods.base_method import BaseMethod
    from ..lib.faithfulness_metrics import recommendation_recency_distribution
    from ..lib.causal_sampling import TOP_K, load_uirt, item_first_seen

    run_dir = RESULTS_DIR / "runs" / "Baby_Products_cpu_seed42"
    csv_path = DATA_DIR / "baby_dataset" / "Baby_Products.csv"
    rows = load_uirt(str(csv_path))
    item_first = item_first_seen(rows)
    rows_sorted = sorted(rows, key=lambda r: r[3])
    boundaries = np.linspace(0, len(rows_sorted), 7, dtype=int)
    slices = [rows_sorted[boundaries[i]:boundaries[i+1]] for i in range(6)]

    def _bm(train_rows, test_rows):
        bm = BaseMethod(fmt="UIRT", rating_threshold=1.0,
                        exclude_unknowns=True, verbose=False, seed=42)
        bm._build_datasets(train_data=train_rows, test_data=test_rows, val_data=None)
        bm._build_modalities()
        return bm

    out = {}
    for i in range(1, 6):
        v_pkl = run_dir / "shuffle_uniform" / f"slice{i}.pkl"
        f_pkl = run_dir / "win10_causal"    / f"slice{i}.pkl"
        if not (v_pkl.exists() and f_pkl.exists()):
            continue
        train_rows = [r for s in slices[:i] for r in s]
        test_rows = slices[i]
        bm = _bm(train_rows, test_rows)
        with open(v_pkl, "rb") as f:
            v_model = pickle.load(f)
        with open(f_pkl, "rb") as f:
            f_model = pickle.load(f)
        v_stats = recommendation_recency_distribution(
            v_model, bm, item_first, k=TOP_K, return_raw=True)
        f_stats = recommendation_recency_distribution(
            f_model, bm, item_first, k=TOP_K, return_raw=True)
        out[i] = (v_stats["recency_days_raw"], f_stats["recency_days_raw"])
    return out


def fig_recency_cdf():
    """Empirical CDFs of recommendation recency, per prequential slice
    on Baby Products (BPR). Each panel shows vanilla vs faithful CDFs
    plus the KS two-sample p-value in-figure. Pairs visually with the
    KS test reported in research/STATS_ks_recency.md."""
    from scipy import stats as sps
    per_slice = _load_recency_arrays_for_baby_bpr()
    if not per_slice:
        print("  warning: no pickles found, skipping recency_cdf")
        return

    n = len(per_slice)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL[0], 2.3), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (i, (v_arr, f_arr)) in zip(axes, sorted(per_slice.items())):
        for arr, color, label in (
            (v_arr, COLOR_VANILLA,  "vanilla"),
            (f_arr, COLOR_FAITHFUL, "faithful"),
        ):
            xs = np.sort(arr)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=color, lw=1.2, label=label)
        ks_stat, ks_p = sps.ks_2samp(v_arr, f_arr)
        p_txt = "p < 1e-12" if ks_p < 1e-12 else f"p = {ks_p:.2e}"
        ax.text(0.05, 0.95,
                f"slice {i}\nKS={ks_stat:.2f}\n{p_txt}",
                transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
        ax.set_xlabel("recency (days)")
        ax.set_xlim(left=0)
    axes[0].set_ylabel("empirical CDF")
    axes[-1].legend(loc="lower right", frameon=False, fontsize=6.5)
    save(fig, "recency_cdf")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALL_FIGS = {
    "temporal_batching":             fig_temporal_batching,
    "prequential_protocol":          fig_prequential_protocol,
    "k_ablation":                    fig_k_ablation,
    "prequential_hr":                fig_prequential_hr,
    "counterfactual_rate":           fig_counterfactual_rate,
    "recency_cdf":                   fig_recency_cdf,
    "three_model_ablation_baby":     fig_three_model_ablation_baby,
    "three_model_ablation":          fig_three_model_ablation,
}


def main(names):
    if not names:
        names = list(ALL_FIGS.keys())
    for n in names:
        if n not in ALL_FIGS:
            print(f"skip: unknown figure {n}")
            continue
        print(f"generating {n}")
        ALL_FIGS[n]()


if __name__ == "__main__":
    main(sys.argv[1:])
