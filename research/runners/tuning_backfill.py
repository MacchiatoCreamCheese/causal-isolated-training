import argparse
import json
import shutil
from pathlib import Path

from ..lib.tuning_config import BASELINE_CONFIG, TUNE_ORDER
from .tuning import SELECT_METRIC, TUNE_DIR, _canon, _value_slug, _write_json

COLS = ("HitRatio@20", "NDCG@20", "Recall@20")


def _table_row(line):
    cols = [c.strip() for c in line.split("|")]
    return cols[0], {k: float(v) for k, v in zip(COLS, cols[1:4])}


def parse_log(path):
    events, mode, val, test, name = [], None, None, None, None
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if s.startswith("VALIDATION:"):
            mode = "val"
        elif s.startswith("TEST:"):
            mode = "test"
        elif "/tune/" in s and "|" in s:
            name, row = _table_row(s)
            if mode == "val":
                val = row
            else:
                test = row
        elif s.startswith("[baseline]") and "cached" not in s and val and test:
            events.append(("baseline", None, None, val, test, name))
            val = test = None
        elif s.startswith("[run]") and val and test:
            knob, value = s.split()[1].split("=", 1)
            events.append(("run", knob, value, val, test, name))
            val = test = None
    return events


def cell_dir(log_path, name):
    parts = name.split("/")
    dataset, recipe, seed = parts[2], parts[3], parts[4][1:]
    stem = Path(log_path).stem
    model_dir = stem[len("tune_"):].split(f"_{dataset}_")[0]
    assert model_dir.split("-")[0] == parts[0].lower(), (log_path, name)
    return model_dir, TUNE_DIR / model_dir / dataset / recipe / f"seed{seed}"


def _matches(stored_test, log_test):
    return round(float(stored_test[SELECT_METRIC]), 4) == log_test[SELECT_METRIC]


def _stamp(payload, val, source):
    test = payload.get("test_metrics", payload["metrics"])
    payload.update(test_metrics=test, val_metrics=val, metrics=val,
                   select_split="validation", val_metrics_source=source)


def backfill(d, events, source, apply):
    loaded = {}
    for kind, knob, value, val, test, _ in events:
        if kind == "baseline":
            paths = [d / "baseline.json"]
        else:
            kd = d / knob
            paths = [p for p in kd.glob("*.json")] if kd.is_dir() else []
        for p in paths:
            payload = loaded.get(p) or json.loads(p.read_text(encoding="utf-8"))
            if kind == "run" and str(payload.get("value")) != value:
                continue
            if "val_metrics" in payload and p not in loaded:
                continue
            if _matches(payload.get("test_metrics", payload["metrics"]), test):
                _stamp(payload, val, source)
                loaded[p] = payload
    if apply:
        for p, payload in loaded.items():
            _write_json(p, payload)
    return loaded


def _validated_winner(d):
    wpath = d / "winner.json"
    if not wpath.exists():
        return False
    w = json.loads(wpath.read_text(encoding="utf-8"))
    return w.get("select_split") == "validation" and "val_metrics_source" not in w


def _carried_value(d, model, knob):
    order = [k for k, _ in TUNE_ORDER[model]]
    for later in order[order.index(knob) + 1:]:
        for p in sorted((d / later).glob("*.json")) if (d / later).is_dir() else []:
            return json.loads(p.read_text(encoding="utf-8"))["config"][knob]
    return None


def reselect(model, d, loaded):
    def read(p):
        if p in loaded:
            return loaded[p]
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    base = read(d / "baseline.json")
    if base is None or "val_metrics" not in base:
        return "incomplete", "baseline", None
    if _canon(base["config"]) != _canon(json.loads(_canon(BASELINE_CONFIG[model]))):
        return "incomplete", "baseline (config changed since)", None
    config = dict(base["config"])
    carry_v = base["val_metrics"][SELECT_METRIC]
    carry_t = base["test_metrics"][SELECT_METRIC]
    for knob, grid in TUNE_ORDER[model]:
        cands = [(config[knob], carry_v, carry_t)]
        for value in grid:
            t = read(d / knob / f"{_value_slug(value)}.json")
            want = dict(config, **{knob: value})
            if (t is None or "val_metrics" not in t
                    or _canon(t["config"]) != _canon(json.loads(_canon(want)))):
                return "incomplete", knob, config
            cands.append((value, t["val_metrics"][SELECT_METRIC],
                          t["test_metrics"][SELECT_METRIC]))
        by_val = max(cands, key=lambda c: c[1])
        by_test = max(cands, key=lambda c: c[2])
        if by_val[0] != by_test[0]:
            taken = _carried_value(d, model, knob)
            if taken is None:
                return "repick", knob, config
            if _canon(taken) != _canon(by_val[0]):
                return "diverge", knob, config
        config[knob] = by_val[0]
        carry_v, carry_t = by_val[1], by_val[2]
    return "agree", (carry_v, carry_t), config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--apply", action="store_true",
                   help="Write and delete. Without it, only report.")
    args = p.parse_args()

    cells = []
    for log in sorted(args.logs):
        events = parse_log(log)
        if not events:
            print(f"{log}: no trials with both tables, skipped")
            continue
        by_name = {}
        for e in events:
            by_name.setdefault(e[5], []).append(e)
        cells += [(log, name, evs) for name, evs in by_name.items()]

    for log, name, events in cells:
        model_dir, d = cell_dir(log, name)
        model = model_dir.split("-")[0]
        loaded = backfill(d, events, f"backfilled from {Path(log).name}, 4 d.p.",
                          args.apply)
        status, where, config = reselect(model, d, loaded)
        rel = d.relative_to(TUNE_DIR)
        if status == "agree":
            v, t = where
            wpath = d / "winner.json"
            if wpath.exists():
                w = json.loads(wpath.read_text(encoding="utf-8"))
                if w.get("select_split") != "validation":
                    assert _canon(w["config"]) == _canon(config), (wpath, w["config"], config)
                    w.update(select_split="validation", metric=v, test_metric=t,
                             val_metrics_source=f"backfilled from {Path(log).name}, 4 d.p.")
                    if args.apply:
                        _write_json(wpath, w)
            print(f"{rel}: KEEP  (validation picks the same values; "
                  f"{len(loaded)} files backfilled)")
        elif status == "repick" and _validated_winner(d):
            print(f"{rel}: KEEP  (winner already selected on validation by the "
                  f"fixed tuner; {len(loaded)} files backfilled)")
        elif status == "repick":
            wpath = d / "winner.json"
            had = wpath.exists()
            if args.apply and had:
                wpath.unlink()
            print(f"{rel}: REPICK at {where}  (validation picks differently; no "
                  f"later trials to delete; {'winner.json removed, ' if had else ''}"
                  f"the tuner re-picks from cache without training)")
        elif status == "diverge":
            order = [k for k, _ in TUNE_ORDER[model]]
            later = [d / k for k in order[order.index(where) + 1:] if (d / k).is_dir()]
            n = sum(len(list(k.glob("*.json"))) for k in later)
            if args.apply:
                for k in later:
                    shutil.rmtree(k)
                (d / "winner.json").unlink(missing_ok=True)
            print(f"{rel}: RERUN after {where}  (validation picks differently; "
                  f"deleted {n} later trials + winner.json; "
                  f"{len(loaded)} files backfilled)")
        else:
            print(f"{rel}: INCOMPLETE at {where}  (no validation score for some "
                  f"trial there; the tuner re-runs from it; "
                  f"{len(loaded)} files backfilled)")
    if not args.apply:
        print("\n(dry run -- re-run with --apply to write and delete)")


if __name__ == "__main__":
    main()
