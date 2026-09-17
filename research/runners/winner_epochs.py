"""Recover how many epochs each BPR tuning winner actually trained for.

The ablation's batch-order steps are meant to be a single-variable comparison
against the past-only winner: same recipe, only the order of the batches
changes. They are not, quite. Every step stops on its own early-stopping rule,
and reordering the batches flattens the validation curve sooner, so a step can
quit long before the winner it is compared against did. On musical/causal/s123
the winner trained 198 epochs and both reordered steps stopped at 41, which is
enough to lose on its own.

Fixing that needs the winner's epoch count, and nothing records it: the trial
files keep metrics and wall time, not epochs. The tuning *logs* do. Every
`[run] knob=value ... (Ns)` line is preceded by BPR's own

    - best epoch = B, stopped epoch = S

and the `(Ns)` wall time matches `wall_seconds` in the trial file exactly, which
is what lets a log line be tied to a file without guessing. This walks the logs,
matches on wall time, and stamps `epochs` into each winner's `users_run` block.

`mecha2/runner.py` reads that field: when it is there, the trained steps run at a
fixed budget with early stopping off, so every step gets the same number of
passes over the data. Winners without it keep the old behaviour.

BPR only. NeuMF has no early stopping (a fixed 20 epochs, so the budget is
already equal) and LightGCN's stop epoch is not in its logs.

Usage (repo root):
    python -m research.runners.winner_epochs tune_bpr_*.log           # dry run
    python -m research.runners.winner_epochs tune_bpr_*.log --apply
"""

import argparse
import json
import re
from pathlib import Path

from ..paths import RESULTS_DIR

TUNE_DIR = RESULTS_DIR / "tuning"

#: "- best epoch = 185, stopped epoch = 198"
EPOCHS_RE = re.compile(r"best epoch = (\d+), stopped epoch = (\d+)")
#: "[run]  learning_rate=0.01  val NDCG@20=0.0195  test=0.0161  (3028.4s)"
RUN_RE = re.compile(r"^\[run\].*\(([\d.]+)s\)")
#: "[baseline] val NDCG@20=0.0051  test=0.0043  (5536.1s)"
BASELINE_RE = re.compile(r"^\[baseline\].*\(([\d.]+)s\)")
#: "=== Tuning bpr on musical x causal x seed=123, selecting on ..."
HEADER_RE = re.compile(r"=== Tuning (\S+) on (\S+) x (\S+) x seed=(\d+)")


def parse_log(path):
    """{(model, dataset, recipe, seed): {wall_seconds: stopped_epoch}}.

    A run's epochs are the last `best/stopped epoch` line before its `[run]` or
    `[baseline]` line; a run that never early-stopped has no such line and is
    skipped rather than guessed at.
    """
    out, cell, pending = {}, None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        head = HEADER_RE.search(line)
        if head:
            cell = (head.group(1), head.group(2), head.group(3), head.group(4))
            out.setdefault(cell, {})
            pending = None
            continue
        ep = EPOCHS_RE.search(line)
        if ep:
            pending = int(ep.group(2))
            continue
        run = RUN_RE.match(line) or BASELINE_RE.match(line)
        if run and cell is not None and pending is not None:
            out[cell][round(float(run.group(1)), 1)] = pending
            pending = None
    return out


def winner_paths(model, dataset, recipe, seed):
    d = TUNE_DIR / model / dataset / recipe / f"seed{seed}"
    return d, d / "winner.json"


def _candidate_walls(d, winner):
    """Wall times that could identify the winning run, best evidence first.

    The winner's own `users_run.wall_seconds` is there when the scores came from
    a retrain. When they came from a cached trial instead, that field is missing
    and the trial file that matches the winning config carries the wall time --
    `baseline.json` included, since the baseline wins whenever no knob improves
    on it.
    """
    out = []
    run = winner.get("users_run") or {}
    if run.get("wall_seconds") is not None:
        out.append(float(run["wall_seconds"]))
    config = winner.get("config")
    for path in sorted(d.rglob("*.json")):
        if path.name in ("winner.json",) or path.name.endswith(".tmp"):
            continue
        try:
            trial = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if trial.get("config") == config and trial.get("wall_seconds") is not None:
            out.append(float(trial["wall_seconds"]))
    return out


def stamp(cell, walls, apply):
    """Match the winner's wall time to a log line and write `epochs`."""
    model, dataset, recipe, seed = cell
    d, wpath = winner_paths(model, dataset, recipe, seed)
    if not wpath.exists():
        return "no winner"
    winner = json.loads(wpath.read_text(encoding="utf-8"))
    run = winner.get("users_run") or {}
    if "epochs" in run:
        return f"already stamped ({run['epochs']})"

    candidates = _candidate_walls(d, winner)
    if not candidates:
        return "no wall time to match on"

    epochs = None
    for wall in candidates:
        key = round(wall, 1)
        # Wall time is the run's own total; the log prints it to 0.1s. Allow a
        # second of slack for the trials whose file and log line were rounded
        # from different clocks, but only when exactly one line is that close,
        # so a near-miss never silently picks the wrong run.
        near = [k for k in walls if abs(k - key) <= 1.0]
        if len(near) == 1:
            epochs = walls[near[0]]
            break
    if epochs is None:
        return f"no log line at wall={[round(w, 1) for w in candidates]}s"

    if apply:
        run["epochs"] = epochs
        winner["users_run"] = run
        tmp = wpath.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(winner, indent=2), encoding="utf-8")
        tmp.replace(wpath)
    return f"epochs={epochs}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logs", nargs="+")
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    cells = {}
    for pattern in args.logs:
        for path in sorted(Path(".").glob(pattern)) or [Path(pattern)]:
            if not path.exists():
                continue
            for cell, walls in parse_log(path).items():
                cells.setdefault(cell, {}).update(walls)

    for cell in sorted(cells):
        model, dataset, recipe, seed = cell
        if model != "bpr":
            continue
        status = stamp(cell, cells[cell], args.apply)
        print(f"{dataset:<14}{recipe:<9}seed{seed:<6}{status}")
    if not args.apply:
        print("\ndry run -- pass --apply to write `epochs` into the winners")


if __name__ == "__main__":
    main()
