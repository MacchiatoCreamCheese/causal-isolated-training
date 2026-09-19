import csv
import json
import math
from statistics import mean, stdev

import numpy as np

from ..lib import user_scores
from ..lib.tuning_config import winner_dir
from ..paths import RESULTS_DIR, REPORTS_DIR

SEEDS = (42, 123, 2026)
MODELS = [("bpr", "BPR"), ("neumf-pretrain", "NeuMF"), ("lightgcn", "LightGCN")]
DATASETS = [("musical", "Musical Instruments"), ("baby", "Baby Products"),
            ("cellphone", "Cell Phones \\& Acc."), ("philadelphia", "Yelp (Philadelphia)"),
            ("movielens", "MovieLens-10M")]
ARMS = [("uniform", "uniform"), ("causal", "past-only")]
METRICS = [("N", "NDCG@20"), ("H", "HitRatio@20")]

CLOSED_FORM = {"musical": 33.18, "baby": 38.42, "cellphone": 44.06,
               "philadelphia": 28.60, "movielens": 46.63}

PCT = "\\%"
SIGNIF = RESULTS_DIR / "diagnostics" / "significance.json"
PREQ_DIR = RESULTS_DIR / "prequential"
OUT_CSV = RESULTS_DIR / "combined_summary.csv"
USER_LEAKAGE = RESULTS_DIR / "diagnostics" / "user_leakage.json"
TABLES = REPORTS_DIR / "tables"


def missing(by_seed):
    return [f"seed{s}" for s in SEEDS if s not in by_seed]


def missing_tex(gone):
    return f"missing [{', '.join(gone)}]"


def complete(by_seed):
    return not missing(by_seed)


def stats_of(by_seed):
    vals = [by_seed[s] for s in SEEDS]
    return mean(vals), stdev(vals)


def fmt(by_seed, digits=4, scale=1.0, unit=""):
    if not complete(by_seed):
        return missing_tex(missing(by_seed))
    m, s = stats_of(by_seed)
    return f"{m * scale:.{digits}f} $\\pm$ {s * scale:.{digits}f}{unit}"


def sci(x):
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = round(x / 10 ** e)
    if m == 10:
        m, e = 1, e + 1
    return f"${m}\\times10^{{{e}}}$"


def p_tex(p):
    if p >= 0.001:
        return f"{p:.3f}"
    e = math.floor(math.log10(p))
    return f"${p / 10 ** e:.1f}{{\\times}}10^{{{e}}}$"


def load_static():
    out = {}
    for model, _ in MODELS:
        for ds, _ in DATASETS:
            for arm, _ in ARMS:
                cell = {k: {} for k in ("NDCG@20", "HitRatio@20", "rho", "collision")}
                for s in SEEDS:
                    d = winner_dir(model, ds, arm, s)
                    npz, wj = d / "winner.users.npz", d / "winner.json"
                    if not (npz.exists() and wj.exists()):
                        continue
                    scores = user_scores.load(npz)
                    for _, key in METRICS:
                        cell[key][s] = float(np.mean(scores[key]))
                    with open(wj, encoding="utf-8-sig") as f:
                        probes = (json.load(f).get("users_run") or {}).get("probes") or {}
                    if "counterfactual_rate" in probes:
                        cell["rho"][s] = float(probes["counterfactual_rate"])
                    if "collision_rate" in probes:
                        cell["collision"][s] = float(probes["collision_rate"])
                out[(model, ds, arm)] = cell
    return out


def load_signif():
    if not SIGNIF.exists():
        return {}, {}
    j = json.loads(SIGNIF.read_text(encoding="utf-8"))
    rows = {(r["model"], r["dataset"]): r for r in j["rows"] if r["kind"] == "sampler"}
    gone = {(m["model"], m["dataset"]): m["missing"]
            for m in j.get("missing", []) if m["kind"] == "sampler"}
    return rows, gone


def load_prequential(tuned_per_arm):
    out = {}
    for path in sorted(PREQ_DIR.glob("*.json")):
        if ("tunedper-arm" in path.name) != tuned_per_arm:
            continue
        with open(path, encoding="utf-8-sig") as f:
            j = json.load(f)
        model = j["model"].lower().split("-m3")[0]
        model = "neumf-pretrain" if model == "neumf" else model
        out.setdefault((model, j["dataset"]), {})[int(j["seed"])] = j["arms"]
    return out


def decay_stats(arms, metric="HitRatio@20", frac=0.10):
    u = np.array([p[metric] for p in arms["uniform"]])
    c = np.array([p[metric] for p in arms["causal"]])
    k = max(1, int(round(len(u) * frac)))
    change = lambda y: float(y[-k:].mean() - y[:k].mean())  # noqa: E731
    return change(u), change(c), float((c > u).mean())


def ladder_tex(static, signif):
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        for mi, (model, m_label) in enumerate(MODELS):
            if mi:
                lines.append("\\cmidrule(lr){2-5}")
            sig = signif.get((model, ds))
            for ri, (tag, key) in enumerate(METRICS):
                u = static[(model, ds, "uniform")][key]
                c = static[(model, ds, "causal")][key]
                cells = [fmt(u), fmt(c)]
                if complete(u) and complete(c):
                    win = 1 if stats_of(c)[0] > stats_of(u)[0] else 0
                    cells[win] = f"\\textbf{{{cells[win]}}}"
                    if tag == "N" and sig and sig["significant"]:
                        cells[win] += "$^{\\dagger}$"
                    elif tag == "N" and sig and sig["significant_rows"]:
                        cells[win] += "$^{\\ddagger}$"
                first = (f"\\multirow{{6}}{{*}}{{{ds_label}}}" if mi == 0 and ri == 0 else "")
                mcol = f"\\multirow{{2}}{{*}}{{{m_label}}}" if ri == 0 else ""
                lines.append(f"{first} & {mcol} & {tag} & {cells[0]} & {cells[1]} \\\\")
    return "\n".join(lines) + "\n"


def rho_tex(static):
    lines = []
    for ds, ds_label in DATASETS:
        cells = [fmt(static[(m, ds, "uniform")]["rho"], digits=2, scale=100, unit=PCT)
                 for m, _ in MODELS]
        coll = [max(v.values()) for m, _ in MODELS
                for v in [static[(m, ds, "causal")]["collision"]] if complete(v)]
        coll_cell = sci(max(coll)) if coll else missing_tex([f"seed{s}" for s in SEEDS])
        measured = (f"\\multicolumn{{3}}{{c}}{{{cells[0]}}}"
                    if cells[0].startswith("missing") and len(set(cells)) == 1
                    else " & ".join(cells))
        lines.append(f"{ds_label} & {measured} & {CLOSED_FORM[ds]:.2f}\\% "
                     f"& {coll_cell} \\\\")
    return "\n".join(lines) + "\n"


def decay_tex(preq):
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        for mi, (model, m_label) in enumerate(MODELS):
            first = f"\\multirow{{3}}{{*}}{{{ds_label}}}" if mi == 0 else ""
            by_seed = preq.get((model, ds), {})
            gone = [f"seed{s}" for s in SEEDS if s not in by_seed]
            if gone:
                body = f"\\multicolumn{{3}}{{c}}{{{missing_tex(gone)}}}"
            else:
                st = {s: decay_stats(by_seed[s]) for s in SEEDS}
                du = {s: st[s][0] for s in SEEDS}
                dp = {s: st[s][1] for s in SEEDS}
                ah = {s: st[s][2] for s in SEEDS}
                ahead = fmt(ah, digits=1, scale=100, unit=PCT)
                body = f"{fmt(du)} & {fmt(dp)} & {ahead}"
            lines.append(f"{first} & {m_label} & {body} \\\\")
    return "\n".join(lines) + "\n"


def bold_p(p, significant):
    s = p_tex(p)
    if not significant:
        return s
    return f"$\\bm{{{s[1:-1]}}}$" if s.startswith("$") else f"\\textbf{{{s}}}"


def signif_tex(signif, gone_map):
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        for mi, (model, m_label) in enumerate(MODELS):
            first = f"\\multirow{{3}}{{*}}{{{ds_label}}}" if mi == 0 else ""
            r = signif.get((model, ds))
            if r is None:
                gone = gone_map.get((model, ds), [f"seed{s}" for s in SEEDS])
                body = f"\\multicolumn{{5}}{{c}}{{{missing_tex(gone)}}}"
            else:
                d = f"{r['diff']:+.4f}"
                if r["significant"]:
                    d = f"\\textbf{{{d}}}"
                p = bold_p(r["p_holm"], r["significant"])
                p_rows = bold_p(r["p_rows_holm"], r["significant_rows"])
                ci = f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                body = f"{d} & {ci} & {p} & {p_rows} & {r['seeds_agree']}/{len(SEEDS)}"
            lines.append(f"{first} & {m_label} & {body} \\\\")
    return "\n".join(lines) + "\n"


def write_csv(static):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "dataset", "arm", "metric", "mean", "sd", "n", "missing_seeds"]
                   + [f"seed{s}" for s in SEEDS])
        for (model, ds, arm), cell in static.items():
            for key, by_seed in cell.items():
                if not by_seed:
                    continue
                vals = list(by_seed.values())
                w.writerow([model, ds, arm, key, f"{mean(vals):.6g}",
                            f"{stdev(vals):.6g}" if len(vals) > 1 else "",
                            len(vals), " ".join(missing(by_seed))]
                           + [f"{by_seed[s]:.6g}" if s in by_seed else "" for s in SEEDS])
    print(f"[combine_seeds] wrote {OUT_CSV}")


AXIS_LABEL = {"review": "reviews", "join": "joined"}


def load_user_leakage():
    if not USER_LEAKAGE.exists():
        return None
    return json.loads(USER_LEAKAGE.read_text(encoding="utf-8"))


def _pct(x, digits=1):
    return f"{100 * x:.{digits}f}\\%"


def user_leakage_tex(ul, axes=("review", "join")):
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        groups = [(axis, g) for axis in axes for g in ul["datasets"][ds]["axes"][axis]]
        for gi, (axis, g) in enumerate(groups):
            if gi and axis != groups[gi - 1][0]:
                lines.append("\\cmidrule(lr){2-8}")
            first = f"\\multirow{{{len(groups)}}}{{*}}{{{ds_label}}}" if gi == 0 else ""
            rng = g["range"].replace("–", "--")
            lines.append(f"{first} & {AXIS_LABEL.get(axis, axis)} & {g['group']} & {rng} & "
                         f"{_pct(g['share_users'])} & {_pct(g['share_rows'])} & "
                         f"{_pct(g['rho'])} & {_pct(g['share_leak'])} \\\\")
    return "\n".join(lines) + "\n"


def user_leakage_fine_tex(ul, axis="fine"):
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        groups = ul["datasets"][ds]["axes"][axis]
        for gi, g in enumerate(groups):
            first = f"\\multirow{{{len(groups)}}}{{*}}{{{ds_label}}}" if gi == 0 else ""
            rng = g["range"].replace("–", "--")
            lines.append(f"{first} & {rng} & {_pct(g['share_users'])} & "
                         f"{_pct(g['share_rows'])} & {_pct(g['rho'])} & "
                         f"{_pct(g['share_leak'])} \\\\")
    return "\n".join(lines) + "\n"


def _gain_cell(g):
    if g is None or g.get("diff") is None:
        return "--"
    s = f"{g['diff']:+.4f}"
    if g["significant"]:
        return f"\\textbf{{{s}}}$^{{\\dagger}}$"
    if g["significant_rows"]:
        return f"{s}$^{{\\ddagger}}$"
    return s


def user_gain_tex(ul, axes=("review", "join")):
    index = {(g["dataset"], g["model"], g["axis"], g["group"]): g for g in ul["gains"]}
    gone = {(m["dataset"], m["model"]): m["missing"] for m in ul["missing"]}
    names = {"review": ("few", "medium", "many"), "join": ("early", "middle", "late")}
    width = sum(len(names[a]) for a in axes)
    lines = []
    for di, (ds, ds_label) in enumerate(DATASETS):
        if di:
            lines.append("\\midrule")
        for mi, (model, m_label) in enumerate(MODELS):
            first = f"\\multirow{{3}}{{*}}{{{ds_label}}}" if mi == 0 else ""
            if (ds, model) in gone:
                body = f"\\multicolumn{{{width}}}{{c}}{{{missing_tex(gone[(ds, model)])}}}"
            else:
                body = " & ".join(_gain_cell(index.get((ds, model, a, n)))
                                  for a in axes for n in names[a])
            lines.append(f"{first} & {m_label} & {body} \\\\")
    return "\n".join(lines) + "\n"


def main():
    static = load_static()
    signif, gone = load_signif()
    if not signif:
        print("[combine_seeds] no significance.json; run research.analysis.significance first")
    write_csv(static)
    TABLES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "ladder.tex": ladder_tex(static, signif),
        "rho.tex": rho_tex(static),
        "decay.tex": decay_tex(load_prequential(tuned_per_arm=False)),
        "decay_perarm.tex": decay_tex(load_prequential(tuned_per_arm=True)),
        "signif.tex": signif_tex(signif, gone),
    }
    ul = load_user_leakage()
    if ul is None:
        print("[combine_seeds] no user_leakage.json; run research.analysis.user_leakage first")
    else:
        outputs["user_leakage.tex"] = user_leakage_tex(ul)
        outputs["user_leakage_fine.tex"] = user_leakage_fine_tex(ul)
        outputs["user_leakage_join_fine.tex"] = user_leakage_fine_tex(ul, "join_fine")
        outputs["user_gain.tex"] = user_gain_tex(ul)
    for name, body in outputs.items():
        (TABLES / name).write_text("% generated by research.analysis.combine_seeds\n"
                                   + body + "\\bottomrule\n",
                                   encoding="utf-8")
        print(f"[combine_seeds] wrote {TABLES / name}")


if __name__ == "__main__":
    main()
