import argparse
import csv
import json
from itertools import product

import numpy as np
from scipy import stats

from ..lib import user_scores
from ..lib.tuning_config import winner_dir
from ..paths import RESULTS_DIR

MODELS = ("bpr", "neumf-pretrain", "lightgcn")
DATASETS = ("musical", "baby", "cellphone", "philadelphia", "movielens")
SEEDS = (42, 123, 2026)
METRIC = "NDCG@20"
ALPHA = 0.05
N_BOOT = 2000

OUT = RESULTS_DIR / "diagnostics" / "significance.json"
PER_USER = RESULTS_DIR / "ablation" / "per_user"
STACKED = RESULTS_DIR / "diagnostics" / "stacked"


def missing_seeds(paths_a, paths_b):
    return [f"seed{s}" for s, a, b in zip(SEEDS, paths_a, paths_b)
            if not (a.exists() and b.exists())]


def _aligned(runs):
    """Per-user METRIC arrays for every run, restricted to users present in all."""
    index = [{u: i for i, u in enumerate(r["raw_user_id"])} for r in runs]
    common = sorted(set.intersection(*(set(ix) for ix in index)))
    return [np.asarray([r[METRIC][ix[u]] for u in common]) for r, ix in zip(runs, index)]


def stack(paths_a, paths_b, out_csv, names=("a", "b")):
    """Append the three seeds into one table: one row per (user, seed).

    Written to out_csv so the pooled data can be inspected; returns
    (user_idx, seed, a, b) arrays aligned row by row.
    """
    loaded = [user_scores.load(p) for p in list(paths_a) + list(paths_b)]
    runs = _aligned(loaded)
    index = [{u: i for i, u in enumerate(r["raw_user_id"])} for r in loaded]
    users = sorted(set.intersection(*(set(ix) for ix in index)))
    k = len(paths_a)
    n = len(users)
    user_idx = np.tile(np.arange(n), k)
    seed = np.repeat(np.asarray(SEEDS), n)
    a, b = np.concatenate(runs[:k]), np.concatenate(runs[k:])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw_user_id", "seed", f"{names[0]} {METRIC}", f"{names[1]} {METRIC}",
                    "diff"])
        for u, s, x, y in zip(user_idx, seed, a, b):
            w.writerow([users[u], s, f"{x:.8g}", f"{y:.8g}", f"{y - x:.8g}"])
    return user_idx, seed, a, b


def compare(user_idx, seed, a, b, rng):
    """Paired test on the seed-stacked table.

    Primary: users are the unit.  The same user appears once per seed and those
    rows are correlated, so each user's rows are averaged, d_u = mean_s(b_su - a_su),
    and a two-sided t-test over users tests mean(d) = 0; the bootstrap CI resamples
    users.  Also reported: the plain paired t-test over all stacked rows, which
    treats the rows as independent and so understates p.
    """
    diff = b - a
    n = int(user_idx.max()) + 1
    d = np.bincount(user_idx, weights=diff, minlength=n) / np.bincount(user_idx, minlength=n)
    t, p = stats.ttest_1samp(d, 0.0)
    t_rows, p_rows = stats.ttest_rel(b, a)
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_seed = [float(diff[seed == s].mean()) for s in SEEDS]
    agree = sum(np.sign(x) == np.sign(d.mean()) for x in per_seed)
    return {"n_users": n, "n_rows": int(len(diff)), "mean_a": float(a.mean()),
            "mean_b": float(b.mean()), "diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "t": float(t), "p": float(p),
            "t_rows": float(t_rows), "p_rows": float(p_rows),
            "per_seed_diff": dict(zip(map(str, SEEDS), per_seed)),
            "seeds_agree": int(agree)}


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def sampler_groups():
    for model, ds in product(MODELS, DATASETS):
        a = [winner_dir(model, ds, "uniform", s) / "winner.users.npz" for s in SEEDS]
        b = [winner_dir(model, ds, "causal", s) / "winner.users.npz" for s in SEEDS]
        yield ("sampler", model, ds, "uniform", "past-only"), a, b


def order_groups():
    steps = [("causal", "past-only"), ("causal+coherent", "+coherent"),
             ("causal+temporal", "+temporal")]
    stems = set()
    for path in PER_USER.glob("*-m2-tunedper-arm_*_seed*_causal.npz"):
        stem = path.name[:-len("_causal.npz")]
        stems.add(stem.rsplit("_seed", 1)[0])
    for base in sorted(stems):
        label, ds = base.split("_", 1)
        model = label.split("-m2")[0]
        for (s_a, n_a), (s_b, n_b) in zip(steps, steps[1:]):
            a = [PER_USER / f"{base}_seed{s}_{s_a}.npz" for s in SEEDS]
            b = [PER_USER / f"{base}_seed{s}_{s_b}.npz" for s in SEEDS]
            yield ("order", model, ds, n_a, n_b), a, b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", action="store_true",
                   help="also test the batch-order steps (after the equal-budget re-run)")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    rows, missing = [], []
    families = [sampler_groups()] + ([order_groups()] if args.order else [])
    for family in families:
        for key, fa, fb in family:
            ident = {"kind": key[0], "model": key[1], "dataset": key[2],
                     "from": key[3], "to": key[4]}
            gone = missing_seeds(fa, fb)
            if gone:
                missing.append(dict(ident, missing=gone))
                continue
            slug = f"{key[0]}_{key[1]}_{key[2]}"
            if key[0] != "sampler":
                slug += f"_{key[3]}-to-{key[4]}".replace("+", "")
            out_csv = STACKED / f"{slug}.csv"
            stacked = stack(fa, fb, out_csv, names=(key[3], key[4]))
            rows.append(dict(ident, seeds=list(SEEDS), stacked_csv=str(out_csv.name),
                             **compare(*stacked, rng)))
    if not rows:
        raise SystemExit("no complete per-user score groups found")

    # Holm within each family, so adding --order never moves the sampler p-values.
    for kind in sorted({r["kind"] for r in rows}):
        fam = [r for r in rows if r["kind"] == kind]
        for col, flag in (("p", "significant"), ("p_rows", "significant_rows")):
            for r, a in zip(fam, holm(np.asarray([r[col] for r in fam]))):
                r[f"{col}_holm"] = float(a)
                r[flag] = bool(a < ALPHA)
        for r in fam:
            r["holm_family_size"] = len(fam)

    print(f"seeds {', '.join(map(str, SEEDS))} stacked into one table per cell "
          f"({STACKED}); Holm within each family\n"
          f"  p_users: users as the unit (each user's seed rows averaged)\n"
          f"  p_rows:  plain paired t-test over all stacked rows\n")
    print(f"{'kind':<8}{'model':<15}{'dataset':<13}{'step':<22}"
          f"{'users':>7}{'rows':>8}{'diff':>10}{'95% CI':>22}{'p_users':>10}{'p_rows':>10}"
          f"  agree")
    for r in rows:
        ci = f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"
        star = lambda f: "*" if r[f] else " "  # noqa: E731
        print(f"{r['kind']:<8}{r['model']:<15}{r['dataset']:<13}"
              f"{r['from'] + ' -> ' + r['to']:<22}{r['n_users']:>7}{r['n_rows']:>8}"
              f"{r['diff']:>+10.5f}{ci:>22}"
              f"{r['p_holm']:>9.2g}{star('significant')}{r['p_rows_holm']:>9.2g}"
              f"{star('significant_rows')}  {r['seeds_agree']}/{len(SEEDS)}")
    for m in missing:
        print(f"{m['kind']:<8}{m['model']:<15}{m['dataset']:<13}"
              f"{m['from'] + ' -> ' + m['to']:<22}missing [{', '.join(m['missing'])}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"alpha": ALPHA, "metric": METRIC, "n_boot": N_BOOT,
                               "seeds": list(SEEDS),
                               "method": "seeds stacked into one table per cell; "
                                         "p: t-test over users (each user's seed rows "
                                         "averaged), bootstrap CI over users; "
                                         "p_rows: paired t-test over all stacked rows; "
                                         "Holm across complete cells, per family",
                               "rows": rows, "missing": missing}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
