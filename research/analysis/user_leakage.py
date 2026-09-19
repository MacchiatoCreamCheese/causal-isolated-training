import csv
import datetime as dt
import json

import numpy as np
from scipy import stats

from ..lib.data import build_eval_method
from ..paths import RESULTS_DIR
from .significance import ALPHA, MODELS, N_BOOT, SEEDS, STACKED, holm

DATASETS = ("musical", "baby", "cellphone", "philadelphia", "movielens")

AMAZON = (1, 5, 17)
GROUPS = {"musical": AMAZON, "baby": AMAZON, "cellphone": AMAZON,
          "philadelphia": (1, 5, 33), "movielens": (1, 65, 257)}
GROUP_NAMES = ("few", "medium", "many")
FINE_EDGES = tuple([1] + [2 ** k + 1 for k in range(1, 14)])
MIN_SHARE = 0.01
JOIN_FINE = 6

OUT_JSON = RESULTS_DIR / "diagnostics" / "user_leakage.json"
OUT_USERS = RESULTS_DIR / "diagnostics" / "user_leakage_users.csv"
SIGNIF = RESULTS_DIR / "diagnostics" / "significance.json"


def user_facts(ds):
    train = build_eval_method(ds, neg_sampling="uniform", seed=42).train_set
    u, items, _r = (np.asarray(a) for a in train.uir_tuple)
    u = u.astype(np.int64)
    items = items.astype(np.int64)
    ts = np.asarray(train.timestamps, dtype=np.int64)

    tau = np.full(train.num_items, np.iinfo(np.int64).max)
    np.minimum.at(tau, items, ts)
    p = (train.num_items - np.searchsorted(np.sort(tau), ts, side="right")) / train.num_items

    n_users = train.num_users
    n = np.bincount(u, minlength=n_users)
    leak = np.bincount(u, weights=p, minlength=n_users)
    join = np.full(n_users, np.iinfo(np.int64).max)
    np.minimum.at(join, u, ts)
    raw = np.empty(n_users, dtype=object)
    for r, idx in train.uid_map.items():
        raw[idx] = str(r)
    keep = n > 0
    return {"raw": raw[keep], "n": n[keep], "leak": leak[keep], "join": join[keep],
            "rho_total": float(p.mean())}


def _date(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m")


def _count_label(lo, hi):
    return f"{lo}+" if hi is None else (f"{lo}" if lo == hi else f"{lo}–{hi}")


def count_groups(n, lowers):
    idx = np.searchsorted(np.asarray(lowers), n, side="right") - 1
    uppers = [b - 1 for b in lowers[1:]] + [None]
    labels = [_count_label(lo, hi) for lo, hi in zip(lowers, uppers)]
    return idx, labels


def fine_lowers(n):
    lowers = [b for b in FINE_EDGES if b <= n.max()]
    while True:
        idx, _ = count_groups(n, lowers)
        share = np.bincount(idx, minlength=len(lowers)) / len(n)
        small = [k for k, s in enumerate(share) if s < MIN_SHARE]
        if not small or len(lowers) == 1:
            return lowers
        k = small[0]
        del lowers[k if k > 0 else 1]


def join_groups(join, k_groups=3):
    cuts = np.quantile(join, [q / k_groups for q in range(1, k_groups)])
    idx = np.searchsorted(cuts, join, side="right")
    labels = []
    for k in range(k_groups):
        sel = join[idx == k]
        labels.append(f"{_date(sel.min())} – {_date(sel.max())}")
    return idx, labels


def summarize(facts, idx, labels, names):
    total_users, total_rows = len(facts["n"]), facts["n"].sum()
    total_leak = facts["leak"].sum()
    out = []
    for k, (name, label) in enumerate(zip(names, labels)):
        sel = idx == k
        rows, leak = facts["n"][sel].sum(), facts["leak"][sel].sum()
        out.append({"group": name, "range": label, "users": int(sel.sum()),
                    "share_users": float(sel.mean()),
                    "share_rows": float(rows / total_rows),
                    "rho": float(leak / rows) if rows else float("nan"),
                    "rho_user_mean": float((facts["leak"][sel] / facts["n"][sel]).mean())
                    if sel.any() else float("nan"),
                    "share_leak": float(leak / total_leak)})
    assert sum(g["users"] for g in out) == total_users
    return out


def load_stacked(model, ds):
    path = STACKED / f"sampler_{model}_{ds}.csv"
    if not path.exists():
        return None
    users, seeds, diff = [], [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            users.append(row["raw_user_id"])
            seeds.append(int(row["seed"]))
            diff.append(float(row["diff"]))
    return np.asarray(users), np.asarray(seeds), np.asarray(diff)


def missing_map():
    if not SIGNIF.exists():
        return {}
    j = json.loads(SIGNIF.read_text(encoding="utf-8"))
    return {(m["model"], m["dataset"]): m["missing"]
            for m in j.get("missing", []) if m["kind"] == "sampler"}


def gain_rows(ds, model, stacked, group_of, names, axis, rng):
    users, seeds, diff = stacked
    uniq, inv = np.unique(users, return_inverse=True)
    d_user = np.bincount(inv, weights=diff) / np.bincount(inv)
    g_user = np.asarray([group_of.get(u, -1) for u in uniq])
    g_row = g_user[inv]
    rows = []
    for k, name in enumerate(names):
        d = d_user[g_user == k]
        r = diff[g_row == k]
        base = {"dataset": ds, "model": model, "axis": axis, "group": name,
                "n_users": int(len(d)), "n_rows": int(len(r))}
        if len(d) < 2:
            rows.append(dict(base, diff=None))
            continue
        boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append(dict(base, diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi),
                         p=float(stats.ttest_1samp(d, 0.0).pvalue),
                         p_rows=float(stats.ttest_1samp(r, 0.0).pvalue)))
    unmatched = int((g_user == -1).sum())
    return rows, unmatched


def apply_holm(rows):
    tested = [r for r in rows if r.get("diff") is not None]
    for col, flag in (("p", "significant"), ("p_rows", "significant_rows")):
        for r, a in zip(tested, holm(np.asarray([r[col] for r in tested]))):
            r[f"{col}_holm"] = float(a)
            r[flag] = bool(a < ALPHA)


def main():
    rng = np.random.default_rng(0)
    gone = missing_map()
    result, gains, missing, per_user = {}, [], [], []

    for ds in DATASETS:
        print(f"[user_leakage] {ds}: building the training split", flush=True)
        facts = user_facts(ds)
        axes = {}

        idx_r, lab_r = count_groups(facts["n"], list(GROUPS[ds]))
        axes["review"] = (idx_r, lab_r, list(GROUP_NAMES))
        idx_j, lab_j = join_groups(facts["join"])
        axes["join"] = (idx_j, lab_j, ["early", "middle", "late"])
        lowers = fine_lowers(facts["n"])
        idx_f, lab_f = count_groups(facts["n"], lowers)
        axes["fine"] = (idx_f, lab_f, lab_f)
        idx_jf, lab_jf = join_groups(facts["join"], JOIN_FINE)
        axes["join_fine"] = (idx_jf, lab_jf, lab_jf)

        summary = {axis: summarize(facts, *spec) for axis, spec in axes.items()}
        for g in summary["review"]:
            if g["share_users"] < MIN_SHARE:
                raise SystemExit(f"{ds}: group {g['group']} ({g['range']}) holds only "
                                 f"{g['share_users']:.2%} of users; revise GROUPS")
        rho_check = sum(g["rho"] * g["share_rows"] for g in summary["review"])
        assert abs(rho_check - facts["rho_total"]) < 1e-9, (rho_check, facts["rho_total"])
        result[ds] = {"users": int(len(facts["n"])), "rows": int(facts["n"].sum()),
                      "rho_total": facts["rho_total"], "review_lowers": list(GROUPS[ds]),
                      "fine_lowers": lowers, "axes": summary}

        for model in MODELS:
            stacked = load_stacked(model, ds)
            if stacked is None:
                missing.append({"dataset": ds, "model": model,
                                "missing": gone.get((model, ds),
                                                    [f"seed{s}" for s in SEEDS])})
                continue
            for axis, (idx, _labels, names) in axes.items():
                group_of = dict(zip(facts["raw"], idx.tolist()))
                rows, unmatched = gain_rows(ds, model, stacked, group_of, names, axis, rng)
                for r in rows:
                    r["test_users_without_train"] = unmatched
                gains.extend(rows)

        per_user.extend(zip([ds] * len(facts["n"]), facts["raw"], facts["n"],
                            facts["join"], facts["leak"] / facts["n"], facts["leak"],
                            idx_r, idx_j, idx_f, idx_jf))

    apply_holm([g for g in gains if g["axis"] in ("review", "join")])
    apply_holm([g for g in gains if g["axis"] in ("fine", "join_fine")])

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "method": "p(t) = share of items first seen after t (Proposition 1); rho = "
                  "sum p / rows per group; gain = seed-averaged per-user NDCG@20 "
                  "change past-only - uniform; p: t-test over users, p_rows: t-test "
                  "over stacked rows; Holm per family (review+join, fine)",
        "groups": {ds: list(g) for ds, g in GROUPS.items()}, "min_share": MIN_SHARE,
        "datasets": result, "gains": gains, "missing": missing}, indent=2),
        encoding="utf-8")
    with open(OUT_USERS, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "raw_user_id", "n_train", "join_ts", "rho_u", "leak_u",
                    "review_group", "join_group", "fine_group", "join_fine_group"])
        for row in per_user:
            w.writerow([row[0], row[1], int(row[2]), int(row[3]), f"{row[4]:.6f}",
                        f"{row[5]:.4f}", *(int(x) for x in row[6:])])

    for ds, r in result.items():
        print(f"\n=== {ds}: {r['users']:,} users, rho {r['rho_total']:.2%}")
        for axis in ("review", "join", "fine", "join_fine"):
            print(f"  {axis}")
            for g in r["axes"][axis]:
                print(f"    {g['group']:>8} {g['range']:>19} users {g['share_users']:6.1%}"
                      f"  rows {g['share_rows']:6.1%}  rho {g['rho']:6.1%}"
                      f"  leak {g['share_leak']:6.1%}")
    print("\ngains (review / join):")
    for g in gains:
        if g["axis"] == "fine" or g.get("diff") is None:
            continue
        print(f"  {g['dataset']:<13}{g['model']:<15}{g['axis']:<7}{g['group']:<8}"
              f"{g['n_users']:>7} {g['diff']:+.5f} [{g['ci_low']:+.5f}, {g['ci_high']:+.5f}]"
              f"  p_user {g['p_holm']:.2g}{'*' if g['significant'] else ' '}"
              f"  p_rows {g['p_rows_holm']:.2g}{'*' if g['significant_rows'] else ' '}")
    print(f"\nwrote {OUT_JSON}\nwrote {OUT_USERS}")


if __name__ == "__main__":
    main()
