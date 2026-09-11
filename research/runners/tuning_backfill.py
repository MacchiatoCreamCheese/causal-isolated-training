"""Recover validation scores for tuning runs made before the test-selection fix.

Until 2026-09-11 `runners/tuning.py` selected on the test set. Its trial files
kept test metrics only, but its stdout logs printed cornac's VALIDATION and TEST
tables for every trial, so validation can be recovered without retraining.

For every tuning cell a log covers, this:

  1. writes `val_metrics` into `baseline.json` and each trial file, but only when
     the log's test NDCG@20 matches the file's (4 d.p.) -- proof the two
     describe the same run;
  2. re-runs the coordinate-descent selection on validation, with the tuner's
     own rule (default first, first maximum wins);
  3. if every knob picks the same value as before, stamps `winner.json` as
     validation-selected -- nothing to re-run;
  4. if the picks first differ at knob K, deletes the trial folders of every
     knob *after* K (they were trained around the test-picked value) and
     `winner.json`. Trials up to and including K stay: they were trained at
     configs both selections agree on. Re-running the tuner then trains only
     the knobs after K.

Validation scores recovered this way have 4 decimals (the logs' precision), and
the files say so. Cells no log covers are untouched; the tuner re-runs their
files because they lack `val_metrics`.

Usage (from the repo root, on each machine, over that machine's logs):
    python -m research.runners.tuning_backfill tune_*.log            # dry run
    python -m research.runners.tuning_backfill tune_*.log --apply
"""

import argparse
import json
import shutil
from pathlib import Path

from ..lib.tuning_config import BASELINE_CONFIG, TUNE_ORDER
from .tuning import SELECT_METRIC, TUNE_DIR, _canon, _value_slug, _write_json

COLS = ("HitRatio@20", "NDCG@20", "Recall@20")


def _table_row(line):
    """(model/tune/... name, {metric: value}) from one cornac result row."""
    cols = [c.strip() for c in line.split("|")]
    return cols[0], {k: float(v) for k, v in zip(COLS, cols[1:4])}


def parse_log(path):
    """[(kind, knob, value_str, val, test, name)] for every trial the log ran."""
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
    """results/tuning/<model dir>/<dataset>/<recipe>/seed<n> for one log.

    Dataset, recipe and seed come from the table row (`NeuMF/tune/<dataset>/
    <recipe>/s<seed>`), so any log name works, including one with a seed suffix
    or several cells in one file. Only the model *directory* (e.g.
    `neumf-pretrain`, which the row does not show) comes from the filename:
    whatever sits between `tune_` and `_<dataset>_`.
    """
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
    """Attach validation metrics to the cell's files. Returns {path: payload}."""
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
                continue                                  # already validation
            if _matches(payload.get("test_metrics", payload["metrics"]), test):
                _stamp(payload, val, source)
                loaded[p] = payload
    if apply:
        for p, payload in loaded.items():
            _write_json(p, payload)
    return loaded


def reselect(model, d, loaded):
    """Replay coordinate descent on validation and on test; find the divergence.

    Returns (status, knob, config) where status is one of
    'agree', 'diverge', 'incomplete'.
    """
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
        # One log can hold several seeds (a loop appending to the same file), so
        # group by the cell named in each table row, never by the first one.
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
                assert _canon(w["config"]) == _canon(config), (wpath, w["config"], config)
                w.update(select_split="validation", metric=v, test_metric=t,
                         val_metrics_source=f"backfilled from {Path(log).name}, 4 d.p.")
                if args.apply:
                    _write_json(wpath, w)
            print(f"{rel}: KEEP  (validation picks the same values; "
                  f"{len(loaded)} files backfilled)")
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
