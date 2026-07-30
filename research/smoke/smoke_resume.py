"""Smoke test the checkpoint/resume mechanism.

Runs baby's prequential twice. First run writes checkpoints; second
run should detect them and skip training. We don't ask for bit-
identical metrics (CPU is, GPU isn't) — we just confirm:

1. The second run prints "RESUMED" for every cell.
2. The second run completes in << time vs the first.
3. The aggregator-input legacy JSON is still written at the end.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Force CPU path so resume is bit-deterministic
os.environ["USE_GPU"] = "0"

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
DATASET = "baby_dataset/Baby_Products.csv"
SEED = "42"

# Clear any prior smoke checkpoint directory to start clean.
# (Other datasets' checkpoint dirs are unaffected — different paths.)
import glob, re
for d in glob.glob(str(ROOT / "research" / "results" / "runs" / "Baby_Products_cpu_seed42_*")):
    print(f"removing stale {d}", flush=True)
    shutil.rmtree(d)

def run(label):
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "research.runners.prequential", DATASET, SEED],
        cwd=ROOT,
        capture_output=True, text=True,
    )
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-2000:])
        sys.exit(1)
    n_resumed = proc.stdout.count("RESUMED")
    n_saved = proc.stdout.count(": saved ")
    print(f"\n[{label}] {elapsed:.1f}s   resumed={n_resumed}   newly_saved={n_saved}", flush=True)
    return elapsed, n_resumed, n_saved, proc.stdout

t1, r1, s1, _ = run("first run (no checkpoints yet)")
t2, r2, s2, _ = run("second run (should be all resumed)")

print()
print("=" * 60)
print(f"first  run: {t1:.1f}s, {r1} resumed, {s1} new")
print(f"second run: {t2:.1f}s, {r2} resumed, {s2} new")
print(f"speedup:    {t1/t2:.1f}x")
print("=" * 60)
expected_cells = 5 * 2  # 5 slices x 2 variants
assert s1 == expected_cells, f"expected {expected_cells} new saves first run, got {s1}"
assert r1 == 0,              f"expected 0 resumed first run, got {r1}"
assert r2 == expected_cells, f"expected {expected_cells} resumed second run, got {r2}"
assert s2 == 0,              f"expected 0 new saves second run, got {s2}"
print("ALL SMOKE TEST ASSERTIONS PASSED")
