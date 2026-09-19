import csv
import json
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..lib.ablation_harness import RECIPES
from ..lib.data import DATASETS as DATASETS_ALL
from ..paths import PROJECT_ROOT as ROOT, DATA_DIR, RESULTS_DIR, FIGURES_DIR

FIG_DIR = FIGURES_DIR
FIG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = RESULTS_DIR / "ablation_summary.csv"


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

COLOR_CELLS = {
    "uniform": "#a6cee3",
    "causal":  "#1f78b4",
}

MODELS = ["BPR", "NeuMF", "LightGCN"]
SEEDS = (42, 123, 2026)
DATASETS = [("musical", "Musical Instruments"), ("baby", "Baby Products"),
            ("cellphone", "Cell Phones & Acc.")]

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
    if not SUMMARY_CSV.exists():
        print(f"  no {SUMMARY_CSV}, skipping "
              f"(run `python -m research.analysis.aggregate_ablation` first)")
        return None
    data = {}
    with open(SUMMARY_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gone = (row.get("missing_seeds") or "").split()
            data[(row["model"], row["dataset"], row["recipe"], row["metric"])] = (
                float(row["mean"]), float(row["std"]), 0 if gone else int(row["n"]), gone)
    return data


def fig_three_model_ablation():
    data = load_summary()
    if data is None:
        return

    fig, axes = plt.subplots(len(DATASETS), len(MODELS),
                             figsize=(7.0, 6.0), sharex=False)
    for r, (ds_key, ds_label) in enumerate(DATASETS):
        for c, model in enumerate(MODELS):
            ax = axes[r][c]
            means, stds, gones = [], [], []
            for recipe in RECIPES:
                m, s, n, gone = data.get((model, ds_key, recipe, "HR@20"),
                                         (0.0, 0.0, 0, [f"seed{x}" for x in SEEDS]))
                means.append(m if n > 0 else 0.0)
                stds.append(s if n > 0 else 0.0)
                gones.append(gone)
            x = np.arange(len(RECIPES))
            ax.bar(x, means, width=0.6,
                   yerr=stds, capsize=2,
                   color=[COLOR_CELLS[k] for k in RECIPES],
                   edgecolor="black", lw=0.4,
                   error_kw=dict(elinewidth=0.6, ecolor="black"))
            ymax = max(means) if max(means) > 0 else 1.0
            for xi, v, s, gone in zip(x, means, stds, gones):
                if v > 0:
                    ax.text(xi, v + s + ymax * 0.04, f"{v:.4f}",
                            ha="center", va="bottom", fontsize=6.0)
                else:
                    ax.text(xi, ymax * 0.5, "missing\n" + "\n".join(gone),
                            ha="center", va="center", fontsize=5.5, color="gray")
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
    data = load_summary()
    if data is None:
        return

    fig, ax = plt.subplots(figsize=DOUBLE_COL)
    width = 0.25
    x = np.arange(len(DATASETS))
    for i, model in enumerate(MODELS):
        vals, errs = [], []
        for ds_key, _ in DATASETS:
            m, s, n, _gone = data.get((model, ds_key, "uniform", "rho"), (0.0, 0.0, 0, []))
            vals.append(m * 100 if n > 0 else 0.0)
            errs.append(s * 100 if n > 0 else 0.0)
        ax.bar(x + (i - 1) * width, vals, width=width, yerr=errs, capsize=2,
               label=model, edgecolor="black", lw=0.4)
        for xi, v in zip(x + (i - 1) * width, vals):
            if v == 0:
                ax.text(xi, 1, "missing", rotation=90, ha="center", va="bottom",
                        fontsize=5.5, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in DATASETS], fontsize=7)
    ax.set_ylabel(r"counterfactual rate $\rho$ (\%)")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.text(0.02, 0.97, "causal: $\\rho \\equiv 0\\%$",
            transform=ax.transAxes, fontsize=7,
            color=COLOR_CELLS["causal"], va="top")
    save(fig, "counterfactual_rate")


def _load_cohorts(ds_key):
    import datetime as _dt

    path = DATA_DIR / DATASETS_ALL[ds_key]["path"]
    per_item = {}
    per_year = Counter()
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

    born = {}
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


STEPS = [("uniform", "uniform"), ("causal", "past-only"),
         ("causal+coherent", "+coherent"), ("causal+temporal", "+temporal")]



def _missing(seeds_present):
    return [f"seed{s}" for s in SEEDS if s not in seeds_present]


def _missing_text(ax, gone, **kw):
    ax.text(0.5, 0.5, f"missing [{', '.join(gone)}]", transform=ax.transAxes,
            ha="center", va="center", fontsize=6.5, color="gray", **kw)


def _load_ablation_cells():
    cells = {}
    for path in sorted((RESULTS_DIR / "ablation").glob("*tunedper-arm*.json")):
        with open(path, encoding="utf-8") as f:
            j = json.load(f)
        recipes = j.get("recipes", {})
        if not all(name in recipes for name, _ in STEPS):
            print(f"  skipping {path.name}: {len(recipes)}/4 steps so far")
            continue
        model = j["model"].split("-m2")[0]
        cells.setdefault((model, j["dataset"]), {})[int(j["seed"])] = recipes
    return cells


def fig_ablation_steps(metrics=("NDCG@20", "HitRatio@20")):
    cells = _load_ablation_cells()
    if not cells:
        print("  no finished ablation cells yet, skipping")
        return

    metric = metrics[0]
    panels = sorted(cells)
    ncol = min(3, len(panels))
    nrow = -(-len(panels) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.6 * nrow), squeeze=False)
    flat = axes.ravel()
    x = np.arange(len(STEPS))

    for ax, (model, ds) in zip(flat, panels):
        by_seed = cells[(model, ds)]
        ax.set_title(f"{model} / {ds}", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in STEPS], rotation=30, ha="right",
                           fontsize=6.5)
        gone = _missing(by_seed)
        if gone:
            _missing_text(ax, gone)
            continue
        y = np.array([[by_seed[s][name][metric] for name, _ in STEPS] for s in SEEDS])
        ax.errorbar(x, y.mean(axis=0), yerr=y.std(axis=0, ddof=1), marker="o", lw=1.4,
                    capsize=2.5, elinewidth=0.8, color=COLOR_CELLS["causal"],
                    label=f"mean ± SD, seeds {', '.join(map(str, SEEDS))}")
        ax.axvline(1.5, color="#999999", lw=0.6, ls=":", zorder=0)
        ax.margins(y=0.18)
        ax.legend(loc="best", frameon=False, fontsize=6)
    for ax in flat[len(panels):]:
        ax.set_axis_off()
    for ax in axes[:, 0]:
        ax.set_ylabel(f"test {metric}")
    fig.tight_layout()
    save(fig, "ablation_steps")


def _load_prequential(pattern):
    """{(model, dataset): {seed: arms}}"""
    groups = {}
    for path in sorted((RESULTS_DIR / "prequential").glob(pattern)):
        with open(path, encoding="utf-8-sig") as f:
            j = json.load(f)
        if not {"uniform", "causal"} <= set(j["arms"]):
            continue
        groups.setdefault((j["model"], j["dataset"]), {})[int(j["seed"])] = j["arms"]
    return groups


def _seed_stack(by_seed, metric):
    """Per-arm (seeds x batches) arrays; batches must be identical across seeds."""
    ref = by_seed[SEEDS[0]]["uniform"]
    key = [(p["ts_start"], p["ts_end"]) for p in ref]
    for s in SEEDS:
        for arm in ("uniform", "causal"):
            got = [(p["ts_start"], p["ts_end"]) for p in by_seed[s][arm]]
            if got != key:
                raise ValueError(f"seed {s} / {arm}: batches differ from seed {SEEDS[0]}")
    days = np.array([(p["ts_end"] - ref[0]["ts_start"]) / 86_400_000 for p in ref])
    u = np.array([[p[metric] for p in by_seed[s]["uniform"]] for s in SEEDS])
    c = np.array([[p[metric] for p in by_seed[s]["causal"]] for s in SEEDS])
    return days, u, c


def _plot_seed_mean(top, bot, days, u, c, window):
    for y, arm in ((u, "uniform"), (c, "causal")):
        colour = COLOR_CELLS[arm]
        m, sd = y.mean(axis=0), y.std(axis=0, ddof=1)
        top.plot(days, m, color=colour, lw=0.4, alpha=0.35)
        rm, rsd = _rolling(m, window), _rolling(sd, window)
        top.fill_between(days, rm - rsd, rm + rsd, color=colour, alpha=0.25, lw=0)
        top.plot(days, rm, color=colour, lw=1.3,
                 label="past-only" if arm == "causal" else arm)

    diff = c - u
    m, sd = diff.mean(axis=0), diff.std(axis=0, ddof=1)
    rm, rsd = _rolling(m, window), _rolling(sd, window)
    bot.axhline(0, color="black", lw=0.6)
    bot.plot(days, m, color="#999999", lw=0.4, alpha=0.5)
    bot.fill_between(days, rm - rsd, rm + rsd, color="#999999", alpha=0.35, lw=0)
    bot.plot(days, rm, color="black", lw=1.0)


def fig_prequential_grid(metric="HitRatio@20", window=9):
    groups = _load_prequential("*tunedper-arm*.json")
    if not groups:
        print("  no results/prequential/*.json, skipping")
        return

    panels = sorted(groups)
    ncol = min(3, len(panels))
    nrow = -(-len(panels) // ncol)
    fig = plt.figure(figsize=(7.0, 2.7 * nrow))
    outer = fig.add_gridspec(nrow, ncol, hspace=0.55, wspace=0.3)

    for k, (model, ds) in enumerate(panels):
        r, c = divmod(k, ncol)
        inner = outer[r, c].subgridspec(2, 1, height_ratios=(3, 1), hspace=0.08)
        top = fig.add_subplot(inner[0])
        bot = fig.add_subplot(inner[1], sharex=top)
        top.set_title(ds, fontsize=8)
        gone = _missing(groups[(model, ds)])
        if gone:
            _missing_text(top, gone)
            bot.set_axis_off()
            continue
        days, u, cz = _seed_stack(groups[(model, ds)], metric)
        _plot_seed_mean(top, bot, days, u, cz, window)

        top.tick_params(labelsize=5.5, labelbottom=False)
        bot.tick_params(labelsize=5.5)
        if c == 0:
            top.set_ylabel(metric, fontsize=6.5)
            bot.set_ylabel("past-only $-$\nuniform", fontsize=5.5)
        if r == nrow - 1:
            bot.set_xlabel("days since first evaluated batch", fontsize=6.5)
        if k == 0:
            top.legend(loc="best", frameon=False, fontsize=5.5)
    save(fig, "prequential_grid")


def _rolling(a, w):
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    for i in range(len(a)):
        lo, hi = max(0, i - w // 2), min(len(a), i + w // 2 + 1)
        out[i] = a[lo:hi].mean()
    return out


def fig_prequential(metric="HitRatio@20", window=9):
    groups = _load_prequential("*.json")
    if not groups:
        print("  no results/prequential/*.json, skipping "
              "(run `python -m research.mecha3.runner` first)")
        return

    for (model, ds), by_seed in sorted(groups.items()):
        gone = _missing(by_seed)
        if gone:
            print(f"  skipping {model} / {ds}: missing [{', '.join(gone)}]")
            continue
        days, u, c = _seed_stack(by_seed, metric)
        fig, (ax, dax) = plt.subplots(
            2, 1, figsize=(7.0, 4.2), sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1]})
        _plot_seed_mean(ax, dax, days, u, c, window)
        ax.set_ylabel(metric)
        ax.legend(loc="upper right", frameon=False, fontsize=6.5)
        ax.set_title(f"{model} / {ds} / mean ± SD over seeds "
                     f"{', '.join(map(str, SEEDS))}", fontsize=8)
        dax.set_ylabel("past-only - uniform")
        dax.set_xlabel("days since first evaluated batch")
        fig.tight_layout()
        save(fig, f"prequential_{model.lower()}_{ds}")


MODEL_STYLE = {"bpr": ("BPR", "#1b9e77", "o"),
               "neumf-pretrain": ("NeuMF", "#d95f02", "s"),
               "lightgcn": ("LightGCN", "#7570b3", "^")}


def fig_user_leakage(axis="review"):
    path = RESULTS_DIR / "diagnostics" / "user_leakage.json"
    if not path.exists():
        print(f"  no {path.name}, skipping (run `python -m research.analysis.user_leakage`)")
        return
    with open(path, encoding="utf-8") as f:
        ul = json.load(f)

    datasets = [(k, lbl) for k, lbl in DATASETS_FIG if k in ul["datasets"]]
    fig, axes = plt.subplots(3, len(datasets), figsize=(7.0, 5.6), squeeze=False,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    for c, (ds, label) in enumerate(datasets):
        d = ul["datasets"][ds]
        groups = d["axes"][axis]
        names = [g["group"] for g in groups]
        fine = axis in ("fine", "join_fine")
        ticks = [g["range"] if fine else f"{g['group']}\n{g['range']}" for g in groups]
        if axis == "join":
            ticks = [g["group"] for g in groups]
        if axis == "join_fine":
            ticks = [g["range"].split(" – ")[0] for g in groups]   # group's first month
        x = np.arange(len(groups))

        ax = axes[0][c]
        ax.bar(x, [100 * g["rho"] for g in groups], width=0.6,
               color=COLOR_CELLS["causal"], edgecolor="black", lw=0.4)
        ax.axhline(100 * d["rho_total"], color="black", lw=0.6, ls="--")
        ax.set_title(label, fontsize=7.5)

        ax = axes[1][c]
        w = 0.36
        ax.bar(x - w / 2, [100 * g["share_users"] for g in groups], width=w,
               color="#dddddd", edgecolor="black", lw=0.4, label="share of users")
        ax.bar(x + w / 2, [100 * g["share_leak"] for g in groups], width=w,
               color=COLOR_CELLS["causal"], edgecolor="black", lw=0.4,
               label="share of future negatives")

        ax = axes[2][c]
        ax.axhline(0, color="black", lw=0.6)
        rows = {(g["model"], g["group"]): g for g in ul["gains"]
                if g["dataset"] == ds and g["axis"] == axis and g.get("diff") is not None}
        present = [m for m in MODEL_STYLE if any(k[0] == m for k in rows)]
        if not present:
            ax.text(0.5, 0.5, "no trained\nmodels", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6, color="gray")
        for i, m in enumerate(present):
            name, colour, marker = MODEL_STYLE[m]
            off = (i - (len(present) - 1) / 2) * 0.18
            ys = [rows.get((m, n)) for n in names]
            for xi, g in zip(x, ys):
                if g is None:
                    continue
                ax.errorbar(xi + off, 1000 * g["diff"],
                            yerr=[[1000 * (g["diff"] - g["ci_low"])],
                                  [1000 * (g["ci_high"] - g["diff"])]],
                            fmt=marker, ms=3, color=colour, elinewidth=0.7, capsize=1.5,
                            mfc=colour if g["significant"] else "white",
                            label=name if xi == 0 else None)

        for r in range(3):
            axes[r][c].set_xlim(-0.6, len(x) - 0.4)
            axes[r][c].set_xticks(x)
            axes[r][c].set_xticklabels(ticks if r == 2 else [""] * len(x), fontsize=5.5,
                                       rotation=90 if fine else 0)
            axes[r][c].tick_params(axis="y", labelsize=6)

    axes[0][0].set_ylabel(r"$\rho$ (%)", fontsize=7)
    axes[1][0].set_ylabel("share (%)", fontsize=7)
    axes[2][0].set_ylabel(r"$\Delta$ NDCG@20 ($\times10^{-3}$)", fontsize=7)
    handles, labels = [], []
    for ax in [axes[1][0], *axes[2]]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False,
                   fontsize=6, bbox_to_anchor=(0.5, -0.02))
    save(fig, f"user_leakage_{axis}")


def fig_user_leakage_all():
    for axis in ("review", "join", "fine", "join_fine"):
        fig_user_leakage(axis)


ALL_FIGS = {
    "user_leakage":          fig_user_leakage_all,
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
