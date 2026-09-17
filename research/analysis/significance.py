"""Paired significance tests on per-user test scores.

The protocol the paper states (Sec. 6.4): for two steps of the ablation, a
two-sided paired t-test over per-user test NDCG@20, within each seed, with a
paired bootstrap 95% confidence interval on the mean difference; p-values are
Holm-corrected across every comparison in the family; and an effect counts only
if it is significant in all three seeds, in the same direction.

Paired because both steps score the same test users on the same split, so each
user is compared with themselves and the between-user variance -- far larger than
any effect here -- drops out. The t-test targets the mean difference, which is the
quantity the tables report.

Comparisons:
  sampler   uniform -> past-only, from the two tuning winners' per-user scores
            (`winner.users.npz`).
  order     past-only -> +coherent -> +temporal, from the ablation's per-user
            files. Off by default: run with --order only once the equal-budget
            re-run has finished, or the test is run on confounded steps.

Users are joined on the dataset's own id, not cornac's index, so two runs that
built their splits separately still line up; a user present in only one file is
dropped rather than guessed.

Usage (repo root):
    python -m research.analysis.significance
    python -m research.analysis.significance --order
"""

import argparse
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


def _paired(a, b):
    """Align two per-user score files on raw user id; returns (x_a, x_b)."""
    ia = {u: i for i, u in enumerate(a["raw_user_id"])}
    ib = {u: i for i, u in enumerate(b["raw_user_id"])}
    common = sorted(set(ia) & set(ib))
    xa = np.asarray([a[METRIC][ia[u]] for u in common])
    xb = np.asarray([b[METRIC][ib[u]] for u in common])
    return xa, xb


def compare(a, b, rng):
    """t-test and bootstrap CI for mean(b - a) over shared users."""
    xa, xb = _paired(a, b)
    d = xb - xa
    t, p = stats.ttest_rel(xb, xa)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_users": int(len(d)), "mean_a": float(xa.mean()),
            "mean_b": float(xb.mean()), "diff": float(d.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "t": float(t), "p": float(p)}


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def sampler_pairs():
    """(key, file_a, file_b) for uniform vs past-only, per finished cell."""
    for model, ds, seed in product(MODELS, DATASETS, SEEDS):
        a = winner_dir(model, ds, "uniform", seed) / "winner.users.npz"
        b = winner_dir(model, ds, "causal", seed) / "winner.users.npz"
        if a.exists() and b.exists():
            yield ("sampler", model, ds, seed, "uniform", "past-only"), a, b


def order_pairs():
    """(key, file_a, file_b) for consecutive batch-order steps."""
    steps = [("causal", "past-only"), ("causal+coherent", "+coherent"),
             ("causal+temporal", "+temporal")]
    for path in sorted(PER_USER.glob("*-m2-tunedper-arm_*_seed*_causal.npz")):
        stem = path.name[:-len("_causal.npz")]
        label, rest = stem.split("_", 1)
        ds, seed = rest.rsplit("_seed", 1)
        model = label.split("-m2")[0]
        for (s_a, n_a), (s_b, n_b) in zip(steps, steps[1:]):
            fa = PER_USER / f"{stem}_{s_a}.npz"
            fb = PER_USER / f"{stem}_{s_b}.npz"
            if fa.exists() and fb.exists():
                yield ("order", model, ds, int(seed), n_a, n_b), fa, fb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", action="store_true",
                   help="also test the batch-order steps (after the equal-budget re-run)")
    args = p.parse_args()

    rng = np.random.default_rng(0)
    rows = []
    families = [sampler_pairs()] + ([order_pairs()] if args.order else [])
    for family in families:
        for key, fa, fb in family:
            res = compare(user_scores.load(fa), user_scores.load(fb), rng)
            rows.append({"kind": key[0], "model": key[1], "dataset": key[2],
                         "seed": key[3], "from": key[4], "to": key[5], **res})
    if not rows:
        raise SystemExit("no per-user score pairs found")

    adj = holm(np.asarray([r["p"] for r in rows]))
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
        r["significant"] = bool(a < ALPHA)

    print(f"{'kind':<8}{'model':<15}{'dataset':<13}{'seed':>5}  {'step':<22}"
          f"{'n':>7}{'diff':>10}{'95% CI':>22}{'p_holm':>10}  sig")
    for r in rows:
        ci = f"[{r['ci_low']:+.5f}, {r['ci_high']:+.5f}]"
        print(f"{r['kind']:<8}{r['model']:<15}{r['dataset']:<13}{r['seed']:>5}  "
              f"{r['from'] + ' -> ' + r['to']:<22}{r['n_users']:>7}{r['diff']:>+10.5f}"
              f"{ci:>22}{r['p_holm']:>10.2g}  {'*' if r['significant'] else ''}")

    # The paper's bar: significant at every seed, all in the same direction.
    print("\nconsistent across all three seeds:")
    groups = {}
    for r in rows:
        groups.setdefault((r["kind"], r["model"], r["dataset"], r["from"], r["to"]), []).append(r)
    verdicts = []
    for (kind, model, ds, a, b), rs in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in rs)
        if len(rs) < len(SEEDS):
            verdict = f"incomplete ({len(rs)}/{len(SEEDS)} seeds)"
        elif all(r["significant"] for r in rs) and len({np.sign(r["diff"]) for r in rs}) == 1:
            verdict = "HOLDS: " + ("gain" if rs[0]["diff"] > 0 else "loss")
        else:
            n_sig = sum(r["significant"] for r in rs)
            verdict = f"does not hold ({n_sig}/{len(rs)} seeds significant)"
        verdicts.append({"kind": kind, "model": model, "dataset": ds,
                         "from": a, "to": b, "seeds": seeds, "verdict": verdict})
        print(f"  {kind:<8}{model:<15}{ds:<13}{a + ' -> ' + b:<22}{verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"alpha": ALPHA, "metric": METRIC, "n_boot": N_BOOT,
                               "rows": rows, "verdicts": verdicts}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
