"""Generate the paper's figures from the aggregated ablation CSV.

Outputs to research/figures/ as PDF (for LaTeX inclusion) and PNG (for quick
previewing). One function per figure; a top-level dispatcher generates all.

Requires `research/results/ablation_summary.csv` — run
`python -m research.analysis.aggregate_ablation` first.

Run: python -m research.analysis.make_figures [fig_name ...]
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from ..lib.ablation_harness import RECIPES
from ..paths import PROJECT_ROOT as ROOT, RESULTS_DIR, FIGURES_DIR

FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "ablation_summary.csv"


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

# Colors — ColorBrewer, grayscale-distinguishable
COLOR_CELLS = {
    "uniform": "#a6cee3",
    "causal":  "#1f78b4",
}

MODELS = ["BPR", "NeuMF", "LightGCN"]
DATASETS = [("baby", "Baby Products"), ("cellphone", "Cell Phones & Acc."),
            ("healthcare", "Health & Household")]

SINGLE_COL = (3.4, 2.2)
DOUBLE_COL = (7.0, 2.6)


def save(fig, name):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {pdf.relative_to(ROOT)} + {png.relative_to(ROOT)}")


def load_summary():
    """{(model, dataset, recipe, metric): (mean, std, n)} or None if absent."""
    if not SUMMARY_CSV.exists():
        print(f"  no {SUMMARY_CSV}, skipping "
              f"(run `python -m research.analysis.aggregate_ablation` first)")
        return None
    data = {}
    with open(SUMMARY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            data[(row["model"], row["dataset"], row["recipe"], row["metric"])] = (
                float(row["mean"]), float(row["std"]), int(row["n"]))
    return data


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_three_model_ablation():
    """Multi-seed uniform-vs-causal ablation across models and datasets.

    Layout: rows = datasets, columns = models, two bars per panel with
    mean +/- std error bars. Cells with no seeds are marked TBD.
    """
    data = load_summary()
    if data is None:
        return

    fig, axes = plt.subplots(len(DATASETS), len(MODELS),
                             figsize=(7.0, 6.0), sharex=False)
    for r, (ds_key, ds_label) in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            means, stds = [], []
            for recipe in RECIPES:
                m, s, n = data.get((model, ds_key, recipe, "HR@20"), (0.0, 0.0, 0))
                means.append(m if n > 0 else 0.0)
                stds.append(s if n > 0 else 0.0)
            x = np.arange(len(RECIPES))
            ax.bar(x, means, width=0.6,
                   yerr=stds, capsize=2,
                   color=[COLOR_CELLS[k] for k in RECIPES],
                   edgecolor="black", lw=0.4,
                   error_kw=dict(elinewidth=0.6, ecolor="black"))
            ymax = max(means) if max(means) > 0 else 1.0
            for xi, v, s in zip(x, means, stds):
                if v > 0:
                    ax.text(xi, v + s + ymax * 0.04, f"{v:.4f}",
                            ha="center", va="bottom", fontsize=6.0)
                else:
                    ax.text(xi, ymax * 0.5, "TBD", ha="center", va="center",
                            fontsize=6.5, color="gray")
            ax.set_xticks(x)
            ax.set_xticklabels(list(RECIPES), rotation=0, fontsize=6.5)
            ax.set_ylim(0, ymax * 1.35 if ymax > 0 else 1.0)
            if r == 0:
                ax.set_title(model, fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{ds_label}\nHR@20", fontsize=7)
    fig.tight_layout()
    save(fig, "three_model_ablation")


def fig_counterfactual_rate():
    """Counterfactual-negative rate ρ under the vanilla (uniform) arm.

    ρ is 0 by construction under `causal`, so only the uniform arm carries
    information: it measures how much of vanilla training is spent on
    comparisons that could not have been made at the time.
    """
    data = load_summary()
    if data is None:
        return

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    width = 0.25
    x = np.arange(len(DATASETS))
    for i, model in enumerate(MODELS):
        vals = []
        for ds_key, _ in DATASETS:
            m, _s, n = data.get((model, ds_key, "uniform", "rho"), (0.0, 0.0, 0))
            vals.append(m * 100 if n > 0 else 0.0)
        ax.bar(x + (i - 1) * width, vals, width=width, label=model,
               edgecolor="black", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in DATASETS], fontsize=7)
    ax.set_ylabel(r"counterfactual rate $\rho$ (\%)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.97, "causal: $\\rho \\equiv 0\\%$",
            transform=ax.transAxes, fontsize=7,
            color=COLOR_CELLS["causal"], va="top")
    save(fig, "counterfactual_rate")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALL_FIGS = {
    "three_model_ablation":  fig_three_model_ablation,
    "counterfactual_rate":   fig_counterfactual_rate,
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
