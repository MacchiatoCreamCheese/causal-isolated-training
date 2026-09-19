import argparse
import json
import re
from pathlib import Path

from ..paths import RESULTS_DIR

TUNE_DIR = RESULTS_DIR / "tuning"

EPOCHS_RE = re.compile(r"best epoch = (\d+), stopped epoch = (\d+)")
RUN_RE = re.compile(r"^\[run\].*\(([\d.]+)s\)")
BASELINE_RE = re.compile(r"^\[baseline\].*\(([\d.]+)s\)")
HEADER_RE = re.compile(r"=== Tuning (\S+) on (\S+) x (\S+) x seed=(\d+)")


def parse_log(path):
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
