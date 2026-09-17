"""Generate the paper's figures from the aggregated ablation CSV.

Outputs to research/figures/ as PDF (for LaTeX inclusion) and PNG (for quick
previewing). One function per figure; a top-level dispatcher generates all.

Requires `research/results/ablation_summary.csv` — run
`python -m research.analysis.aggregate_ablation` first.

Run: python -m research.analysis.make_figures [fig_name ...]
"""

import csv
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

from ..lib.ablation_harness import RECIPES
from ..lib.data import DATASETS as DATASETS_ALL
from ..paths import PROJECT_ROOT as ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR

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
    # ACM rejects Type 3 fonts; 42 embeds TrueType instead.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
# The datasets the runners actually sweep by default -- kept in step with
# `ablation_harness.DEFAULT_DATASETS`. Healthcare (7.18M) is opt-in via an
# explicit `--datasets healthcare` and is left out of the figures for now;
# add its row back here once it has been run.
DATASETS = [("musical", "Musical Instruments"), ("baby", "Baby Products"),
            ("cellphone", "Cell Phones & Acc.")]

# Every dataset the paper reports on, for figures that describe the data itself
# and so do not wait on a finished run.
DATASETS_FIG = DATASETS + [("philadelphia", "Yelp (Philadelphia)"),
                           ("movielens", "MovieLens-10M")]

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


def _load_cohorts(ds_key):
    """(years, interactions_that_year, mean_interactions_per_item_born_that_year).

    Read straight from the raw UIRT CSV with the stdlib: this figure describes the
    data alone, so it must not depend on a split, a loader, or a finished run. The
    log is streamed rather than materialized -- MovieLens-10M is ten million rows,
    and holding them as Python lists costs more than the plot is worth.
    """
    import datetime as _dt

    path = DATA_DIR / DATASETS_ALL[ds_key]["path"]
    per_item = {}          # item -> [first-seen ms, interaction count]
    per_year = Counter()   # calendar year -> interactions that year
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            item, ts = row[1], int(row[3])
            rec = per_item.get(item)
            if rec is None:
                per_item[item] = [ts, 1]
            else:
                rec[1] += 1
                if ts < rec[0]:
                    rec[0] = ts
            per_year[_dt.datetime.fromtimestamp(ts / 1000, _dt.timezone.utc).year] += 1

    born = {}              # launch year -> [items born, their total interactions]
    for first_ts, count in per_item.values():
        y = _dt.datetime.fromtimestamp(first_ts / 1000, _dt.timezone.utc).year
        acc = born.setdefault(y, [0, 0])
        acc[0] += 1
        acc[1] += count

    years = sorted(set(per_year) | set(born))
    inter = [per_year.get(y, 0) for y in years]
    mean_per = [born[y][1] / born[y][0] if y in born else np.nan for y in years]
    return years, inter, mean_per


def fig_item_cohorts():
    """What the catalogue's growth looks like, per dataset, with no model involved.

    Bars: interactions per calendar year. Line: mean interactions per item, items
    grouped by their launch year tau(i). The bars are catalogue and traffic growth;
    the falling line is observation-window censoring, since an item appearing late
    in the log has less remaining time in which to accumulate interactions. That
    censoring is why every tau-derived quantity carries a cohort bias, and the
    shape of the bars against the line is what sets rho (Proposition 3.1).
    """
    keys = [k for k, _ in DATASETS_FIG]
    ncol = 3
    nrow = -(-len(keys) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (ds_key, label) in zip(axes, DATASETS_FIG):
        path = DATA_DIR / DATASETS_ALL[ds_key]["path"]
        if not path.exists():
            ax.set_axis_off()
            print(f"  no {path}, skipping {ds_key}")
            continue
        years, inter, mean_per = _load_cohorts(ds_key)
        # Drop the long sparse pre-history so the readable range is visible.
        keep = [k for k, v in enumerate(inter) if v >= max(inter) * 0.005]
        lo = keep[0]
        years, inter, mean_per = years[lo:], inter[lo:], mean_per[lo:]

        ax.bar(years, inter, color=COLOR_CELLS["uniform"], alpha=0.85,
               edgecolor="none")
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("year")
        ax.tick_params(axis="x", rotation=90)
        ax.grid(False)

        ax2 = ax.twinx()
        ax2.plot(years, mean_per, color="#333333", marker="o", markersize=2.5,
                 lw=1.2)
        finite = [v for v in mean_per if np.isfinite(v)]
        if finite:
            ax2.set_ylim(0, max(finite) * 1.15)
        ax2.grid(False)
        ax2.spines["right"].set_visible(True)
        ax2.set_ylabel("mean interactions\nper item (by launch year)",
                       fontsize=6.5)

    for ax in axes[len(DATASETS_FIG):]:
        ax.set_axis_off()
    for ax in axes[::ncol]:
        ax.set_ylabel("interactions")
    fig.tight_layout()
    save(fig, "item_cohorts")


#: The four ablation steps, in order, as `mecha2/runner.ARMS` names them, with the
#: labels the paper uses.
STEPS = [("uniform", "uniform"), ("causal", "past-only"),
         ("causal+coherent", "+coherent"), ("causal+temporal", "+temporal")]


def _load_ablation_cells():
    """Every finished four-step cell: {(model, dataset, seed): {step: metrics}}.

    Reads the per-cell JSON the ablation runner writes. Cells still in flight have
    fewer than four steps recorded and are skipped, so the figure only ever shows
    what actually completed.
    """
    cells = {}
    for path in sorted((RESULTS_DIR / "ablation").glob("*tunedper-arm*.json")):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        recipes = j.get("recipes", {})
        if not all(name in recipes for name, _ in STEPS):
            print(f"  skipping {path.name}: {len(recipes)}/4 steps so far")
            continue
        model = j["model"].split("-m2")[0]
        cells[(model, j["dataset"], j["seed"])] = recipes
    return cells


def fig_ablation_steps(metrics=("NDCG@20", "HitRatio@20")):
    """The cumulative ablation, one line per finished cell.

    Each line walks the four steps left to right, so the shape is the result: the
    lift from the past-only pool is the first segment, and whether time-coherent
    or chronological batches add anything is the rest. Values are test metrics at
    each step's own tuned hyperparameters (steps 1-2 from tuning, 3-4 trained).
    """
    cells = _load_ablation_cells()
    if not cells:
        print("  no finished ablation cells yet, skipping")
        return

    # One panel per (model, dataset), one line per seed. Every cell on one pair of
    # axes made the seed spread unreadable once more than a couple had finished,
    # and the spread is half of what the ablation is for: a step that moves less
    # than its own seeds do has not moved.
    metric = metrics[0]
    panels = sorted({(m, ds) for m, ds, _ in cells})
    ncol = min(3, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.6 * nrow), squeeze=False)
    flat = axes.ravel()
    x = np.arange(len(STEPS))

    for ax, (model, ds) in zip(flat, panels):
        for (m, d, seed), recipes in sorted(cells.items()):
            if (m, d) != (model, ds):
                continue
            ax.plot(x, [recipes[name][metric] for name, _ in STEPS],
                    marker="o", lw=1.4, label=f"seed {seed}")
        # Steps 1-2 come from tuning, 3-4 are trained here: mark the seam so the
        # reader knows where the hyperparameters stop changing.
        ax.axvline(1.5, color="#999999", lw=0.6, ls=":", zorder=0)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in STEPS], rotation=30, ha="right",
                           fontsize=6.5)
        ax.set_title(f"{model} / {ds}", fontsize=8)
        ax.margins(y=0.18)
        ax.legend(loc="best", frameon=False, fontsize=6)
    for ax in flat[len(panels):]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel(f"test {metric}")
    fig.tight_layout()
    save(fig, "ablation_steps")


def fig_prequential_grid(metric="HitRatio@20", window=9):
    """Every finished timeline on one sheet: datasets down, seeds across.

    `fig_prequential` draws one cell per file, which is the right unit for a
    close reading of a single run but hopeless for seeing whether a pattern
    repeats. This puts them on shared axes per dataset so the seeds can be
    compared at a glance, with the per-batch points dropped and only the rolling
    means kept -- at this size the raw scatter is noise.
    """
    import json as _json

    payloads = {}
    for path in sorted((RESULTS_DIR / "prequential").glob("*tunedper-arm*.json")):
        with open(path, encoding="utf-8") as f:
            j = _json.load(f)
        payloads[(j["dataset"], int(j["seed"]))] = j["arms"]
    if not payloads:
        print("  no results/prequential/*.json, skipping")
        return

    datasets = sorted({d for d, _ in payloads})
    seeds = sorted({s for _, s in payloads})

    # Each cell keeps the two-panel shape of the single-run figure: the metric
    # above, `causal - uniform` below. The difference panel is the one that
    # carries the claim -- batches differ in difficulty, and both samplers are
    # measured on identical ones, so their difference cancels that out.
    fig = plt.figure(figsize=(7.0, 2.7 * len(datasets)))
    outer = fig.add_gridspec(len(datasets), len(seeds), hspace=0.55, wspace=0.3)

    for r, ds in enumerate(datasets):
        for c, seed in enumerate(seeds):
            arms = payloads.get((ds, seed))
            inner = outer[r, c].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.08)
            top = fig.add_subplot(inner[0])
            bot = fig.add_subplot(inner[1], sharex=top)
            if not arms:
                top.set_axis_off()
                bot.set_axis_off()
                continue

            days = [(p["ts_end"] - arms["uniform"][0]["ts_start"]) / 86_400_000
                    for p in arms["uniform"]]
            for arm, colour in (("uniform", COLOR_CELLS["uniform"]),
                                ("causal", COLOR_CELLS["causal"])):
                y = [p[metric] for p in arms[arm]]
                top.plot(days, y, color=colour, lw=0.4, alpha=0.35)
                top.plot(days, _rolling(y, window), color=colour, lw=1.3, label=arm)

            diff = [b[metric] - a[metric]
                    for a, b in zip(arms["uniform"], arms["causal"])]
            bot.axhline(0, color="black", lw=0.6)
            bot.plot(days, diff, color="#999999", lw=0.4, alpha=0.5)
            smoothed = _rolling(diff, window)
            bot.plot(days, smoothed, color="black", lw=1.0)
            bot.fill_between(days, 0, smoothed, color=COLOR_CELLS["uniform"],
                             alpha=0.5)

            top.tick_params(labelsize=5.5, labelbottom=False)
            bot.tick_params(labelsize=5.5)
            if r == 0:
                top.set_title(f"seed {seed}", fontsize=8)
            if c == 0:
                top.set_ylabel(f"{ds}\n{metric}", fontsize=6.5)
                bot.set_ylabel("causal $-$\nuniform", fontsize=5.5)
            if r == len(datasets) - 1:
                bot.set_xlabel("days since first evaluated batch", fontsize=6.5)
            if (r, c) == (0, 0):
                top.legend(loc="best", frameon=False, fontsize=5.5)
    save(fig, "prequential_grid")


def _rolling(a, w):
    """Centred rolling mean, with the window shrinking at the edges.

    A per-batch metric over a few hundred interactions is jumpy by construction;
    the smoothed line is what is readable and the raw points are what is honest,
    so the figure shows both.
    """
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i - w // 2), min(len(a), i + w // 2 + 1)
        out[i] = a[lo:hi].mean()
    return out


def fig_prequential(metric="HitRatio@20", window=9):
    """Performance along the deployment timeline, one line per arm.

    Top panel: each arm's metric per batch, raw points faint behind a rolling
    mean. Bottom panel: `causal - uniform` against a zero line.

    The bottom panel is the one that carries the claim. Batches differ in
    difficulty -- different users, different catalogue size, different counts --
    so an absolute curve confounds "the arms diverged" with "this stretch of the
    timeline was harder". Both arms are measured on *identical* batches, so their
    difference cancels that out, and a crossover is simply a sign change.
    """
    import json

    paths = sorted((RESULTS_DIR / "prequential").glob("*.json"))
    if not paths:
        print("  no results/prequential/*.json, skipping "
              "(run `python -m research.mecha3.runner` first)")
        return

    for path in paths:
        with open(path, encoding="utf-8-sig") as f:
            payload = json.load(f)
        arms = payload["arms"]
        if not {"uniform", "causal"} <= set(arms):
            continue

        # Batches are identical across arms by construction; take the x-axis from
        # either and assert rather than assume.
        u_rec, c_rec = arms["uniform"], arms["causal"]
        n = min(len(u_rec), len(c_rec))
        mid = np.array([(r["ts_start"] + r["ts_end"]) / 2 for r in u_rec[:n]])
        x = (mid - mid.min()) / (1000 * 60 * 60 * 24)   # days since the first point
        u = np.array([r[metric] for r in u_rec[:n]])
        c = np.array([r[metric] for r in c_rec[:n]])

        fig, (ax, dax) = plt.subplots(
            2, 1, figsize=(7.0, 4.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]})

        for vals, arm in ((u, "uniform"), (c, "causal")):
            ax.plot(x, vals, lw=0.5, alpha=0.25, color=COLOR_CELLS[arm])
            ax.plot(x, _rolling(vals, window), lw=1.6,
                    color=COLOR_CELLS[arm], label=arm)
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
        ax.set_title(f"{payload['model']} / {payload['dataset']} / "
                     f"seed {payload['seed']}", fontsize=8)

        gap = c - u
        dax.axhline(0.0, color="black", lw=0.6)
        dax.plot(x, gap, lw=0.5, alpha=0.25, color="black")
        dax.plot(x, _rolling(gap, window), lw=1.6, color="black")
        dax.fill_between(x, 0, _rolling(gap, window),
                         where=_rolling(gap, window) >= 0,
                         color=COLOR_CELLS["causal"], alpha=0.25, lw=0)
        dax.set_ylabel("causal - uniform")
        dax.set_xlabel("days since first evaluated batch")

        fig.tight_layout()
        # The model label belongs in the filename: two models on the same
        # dataset and seed are two different figures, and keying only on
        # dataset+seed silently overwrote the first with the second.
        save(fig, f"prequential_{payload['model'].lower()}_"
                  f"{payload['dataset']}_seed{payload['seed']}")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

ALL_FIGS = {
    "item_cohorts":          fig_item_cohorts,
    "ablation_steps":        fig_ablation_steps,
    "prequential_grid":      fig_prequential_grid,
    "three_model_ablation":  fig_three_model_ablation,
    "counterfactual_rate":   fig_counterfactual_rate,
    "prequential":           fig_prequential,
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
