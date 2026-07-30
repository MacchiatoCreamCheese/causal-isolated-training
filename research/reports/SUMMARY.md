# Causally Isolated Training — `research/`

Single source of truth for the paper *Causally Isolated Training for Leakage-Free Recommendation* (`../sample.tex`). Read this end-to-end to onboard from cold; everything else under `research/` is either code, auto-generated tables, or raw run artifacts.

---

## The problem in one paragraph

A deployed recommender at time t can only learn from past interactions, only score against items that exist at t, and only use parameters that existed at t. Standard offline training violates all three: it shuffles the entire interaction log into one bag, samples negatives from the full item catalog regardless of timestamp, and evaluates on a random hold-out drawn from the same shuffled distribution. The resulting accuracy numbers reflect a counterfactual world where every item existed at every time — measurably inflated by leakage the production system cannot exploit. We propose three training-side mechanisms that close this leakage and two probes that quantify it independently of test accuracy.

## The three mechanisms

1. **Causal negative sampling** — negatives for a positive at time t are drawn only from items with first-seen timestamp ≤ t. Implemented via per-item first-seen prefix arrays + binary search. Code: `lib/bpr_cpu.py`, `lib/bpr_gpu.py`, mirrored in `lib/neumf.py` / `lib/lightgcn.py`.
2. **Temporal batching (K=10)** — chronologically partition the training log into K equal-count windows; train each window with warm-started parameters. Code: `lib/temporal_batching.py` (`BPRWindowed`), and per-model windowed variants inside `lib/neumf.py` / `lib/lightgcn.py`.
3. **Prequential evaluation (N=6)** — N equal-count slices over the timeline; for i = 1..N-1 train on cumulative prefix, evaluate on slice i. Code: `runners/prequential.py` with idempotent per-cell JSON+pickle checkpointing.

## Two faithfulness probes

- **Counterfactual-negative rate ρ** — fraction of training negatives whose first-seen timestamp lies *after* their paired positive. Zero by construction under causal sampling. Reported per (model, dataset, recipe) cell in every per-seed JSON.
- **Recommendation-recency distribution** — for each test instance at time t and each of the top-K recommendations i, record (t − τ(i)) in days. Lower median = model is recommending more recently-introduced items. Two-sample KS confirms vanilla vs faithful are different at every prequential slice (Baby, p < 1e-12).

## Models, datasets, protocols

- **Models (personalized recommendation only):** BPR, NeuMF, LightGCN. Sequential models (SASRec etc.) are explicitly out of scope.
- **Datasets:** Amazon Reviews 2023 5-core — Baby Products (1.24M), Cell Phones & Accessories (2.75M), Health & Household (7.18M); plus MovieLens-1M (1.00M, auto-fetched via cornac, cached to `movielens_dataset/MovieLens_1M.csv`). Each dataset records its own `val_ts`/`test_ts` in `lib/ablation_harness.py:DATASETS` — Amazon uses Aug 2021 / Jul 2022 calendar cutoffs; MovieLens uses 80/90-percentile quantiles of its own timestamps (975768738 / 978133376).
- **Protocols:** (i) single-cutoff **TimestampSplit** with `val=1628643414042` (Aug 2021), `test=1658002729837` (Jul 2022); (ii) **prequential** with N=6 slices for BPR only.
- **Seeds:** 42, 123, 2026. Results reported as `mean ± std` (sample std, ddof=1) across the three seeds.
- **2×2 ablation cells per (model, dataset):** `shuffle+uniform` (vanilla), `shuffle+causal`, `win10+uniform`, `win10+causal` (faithful recipe).

## Current state (2026-05-29)

| Phase | Scope | Status |
|---|---|---|
| A | BPR baby × {42, 123, 2026} | done — `research/results/2x2/bpr_baby_seed*.json` |
| B | NeuMF {baby, cellphone, healthcare} × 3 seeds | running (s42 + most of s123 done; s2026 pending) — `bpna9hd3w` |
| C | BPR {cellphone, healthcare} × 3 seeds | pending — runs locally after B |
| D | LightGCN {baby, cellphone, healthcare} × 3 seeds | **running on remote GPU box in parallel** — scp `lightgcn_*.json` back to `research/results/2x2/` when done |
| — | Prequential BPR all 3 datasets seed=42 | done — `results/preq_*.json`, drives §V-C tables/figures |
| — | Aggregation + paper edits | pending — needs A–D before final tables/Fig 3 |

Paper file (`../sample.tex`) state:
- Tables I, II (split comparison), III (BPR 2×2), VI (K ablation), VII (prequential macro), VIII (per-slice HR), IX (ρ), X (recency) — populated for available cells.
- Tables IV (NeuMF 2×2), V (LightGCN 2×2) — Baby cells filled, Cellphone/Healthcare cells `TBD` until Phase B/C land.
- §V-D cross-model paragraph uses Baby-only numbers; rewrites after Phase B/C.
- §VI Discussion, §VII Conclusion, Abstract — `TODO`.
- Mean ± std swap of Tables III/IV/V — pending until all seeds land.

## Directory layout

`research/` is a Python package — every entry point runs as `python -m research.<subpkg>.<module>` from the repo root.

```
research/
  __init__.py
  paths.py                          filesystem anchors (DATA_DIR / RESULTS_DIR / LOGS_DIR / ...); env-overridable
  lib/                              importable building blocks (no CLI main)
    timeaware_data.py               Mechanism 1 as a generic data loader: TimeAwareDataset
                                    (adds neg_sampling="causal" to cornac's Dataset.uij_iter)
                                    + CausalTimestampSplit
    causal_negative_sampler.py      per-fit Numpy/Torch samplers all models share (one definition)
    data.py                         DATASETS registry + build_eval_method() factory (uses cornac Reader)
    causal_sampling.py              Mechanism 1 evaluation-time probes (item_first_seen, future_items_pct, TOP_K)
    temporal_batching.py            Mechanism 2 (BPRWindowed, CPU)
    faithfulness_metrics.py         ρ + recency-distribution helpers
    bpr_cpu.py                      BPR CPU backbone (BPRMiniBatch)
    bpr_gpu.py                      BPR CUDA backbone (BPRMiniBatchGPU / BPRWindowedGPU)
    neumf.py                        NeuMFRecommender — faithful two-tower + three-phase training
    lightgcn.py                     LightGCNRecommender — per-window graph rebuild
    ablation_harness.py             parse_args, load_partial, write_partial (re-exports DATASETS from data.py)
    tuning_config.py                hyperparameter inventory + TUNE_ORDER + BASELINE_CONFIG
  runners/                          CLI entry points (`python -m research.runners.<name>`)
    ablation_bpr.py                 BPR 2×2 runner (multi-seed, GPU)
    ablation_neumf.py               NeuMF 2×2 runner (multi-seed, GPU)
    ablation_lightgcn.py            LightGCN 2×2 runner (multi-seed, GPU+DGL); owns LIGHTGCN_BATCH/EPOCHS/EARLY_STOP
    k_ablation.py                   K sweep — Table VI / Fig 4
    prequential.py                  Mechanism 3, BPR prequential (resumable per-slice)
    tuning.py                       coordinate-descent hyperparameter tuning
  analysis/                         post-hoc tables, stats, figures
    aggregate_2x2.py                per-seed JSONs → reports/RESULTS_2x2.md + results/2x2_summary.csv
    aggregate_prequential.py        prequential JSONs → reports/RESULTS_prequential.md
    stats_paired_tests.py           → reports/STATS_paired_tests.md
    stats_ks_recency.py             → reports/STATS_ks_recency.md (supports --datasets)
    make_figures.py                 generates figures/*.pdf + *.png
  reports/                          auto-generated markdown + this narrative
    SUMMARY.md                      ← this file
    RESULTS_2x2.md
    RESULTS_prequential.md
    STATS_paired_tests.md
    STATS_ks_recency.md
  smoke/                            smoke tests (`python -m research.smoke.<name>`)
  results/
    2x2/<model>_<dataset>_seed<n>.json    per-seed 2×2 metric cells (live, growing)
    preq_<stem>_<backend>_seed<n>.json    per-cell prequential metrics
    runs/<stem>_<backend>_seed<n>/        prequential pickle checkpoints (gitignored)
    tuning/<model>/                       coordinate-descent trials (baseline/<knob>/<value>/winner JSONs)
  figures/                          PDFs/PNGs referenced in sample.tex
  logs/                             cornac Experiment auto-logs (current) + _archive/ (pre-reorg)
```

## Generic causal data loader (Mechanism 1)

Causal negative sampling is a **data-layer** component, not per-model code.
`lib/timeaware_data.TimeAwareDataset` subclasses `cornac.data.Dataset` and adds a
third `neg_sampling` option — `"causal"` — to cornac's own `Dataset.uij_iter`
(joining the built-in `"uniform"`/`"popularity"`). A negative for an interaction
at time *t* is drawn only from items first seen at-or-before *t* (vectorized
`searchsorted` over a per-item first-seen index). `CausalTimestampSplit`
(subclass of `TimestampSplit`) yields a training split that *is* a
`TimeAwareDataset`. `lib/data.build_eval_method(key)` is the one factory every
runner calls — no runner re-writes a `TimestampSplit(...)` block anymore.

All three models (BPR CPU/GPU/windowed, NeuMF, LightGCN) consume the same
definition through `lib/causal_negative_sampler.{NumpyCausalSampler,TorchCausalSampler}`,
which read the first-seen index off the training set — the bespoke
`_build_causal_index`/`_sample_negatives` that used to be copy-pasted into each
model are gone. Any cornac-style model that trains via `train_set.uij_iter(...)`
gets causal sampling for free. (Like vanilla BPR, the causal draw does not reject
an occasionally-observed item; collision <0.1%, and rejecting would break parity.)

## Standalone project

The package depends only on released **cornac ≥ 2.4.0** (where `TimestampSplit`
and the `val_set → fit()` wiring shipped) — the fork made no substantive cornac
edits. A self-contained copy lives at `../causal-isolated-training/` (sibling of
this repo): `research/` + `pyproject.toml` + `requirements.txt` + `paths.py`
anchoring + a README relocation/git-init guide, running against a pip-installed
cornac with no cornac source tree. See that folder's README for how to relocate
it and `git init`.

## Environment

WSL conda env `leakage`:
- Python 3.12
- torch 2.4.1+cu121
- dgl 2.4.0+cu121 (LightGCN only)
- cornac installed editable from repo root
- `pip install powerlaw` (cornac runtime dep not pulled in by `-e .`)

```
conda activate leakage
pip install -e .
pip install powerlaw
```

## Commands (run from repo root)

### 2×2 ablation, multi-seed

```
python -m research.runners.ablation_bpr      --seeds 42,123,2026 --datasets baby,cellphone,healthcare
python -m research.runners.ablation_neumf    --seeds 42,123,2026 --datasets baby,cellphone,healthcare
python -m research.runners.ablation_lightgcn --seeds 42,123,2026 --datasets baby,cellphone,healthcare
```

Each `(model, dataset, seed)` cell is checkpointed per-recipe via `lib.ablation_harness.write_partial`. Killing a running script loses only the in-flight recipe; everything completed is on disk and a re-launch resumes via `load_partial`.

### Prequential BPR

```
python -m research.runners.prequential baby_dataset/Baby_Products.csv 42
USE_GPU=1 python -m research.runners.prequential cellphone_dataset/Cell_Phones_and_Accessories.csv 42
USE_GPU=1 python -m research.runners.prequential healthcare_dataset/Health_and_Household.csv 42
```

Idempotent and resumable at per-slice granularity (own pickle scheme under `results/runs/`).

### Hyperparameter tuning (coordinate descent)

```
python -m research.runners.tuning --model lightgcn
python -m research.runners.tuning --model neumf
python -m research.runners.tuning --model bpr
```

### Aggregate → tables → figures → paper

```
python -m research.analysis.aggregate_2x2          # → reports/RESULTS_2x2.md  + results/2x2_summary.csv
python -m research.analysis.aggregate_prequential  # → reports/RESULTS_prequential.md
python -m research.analysis.stats_paired_tests     # → reports/STATS_paired_tests.md
python -m research.analysis.stats_ks_recency       # → reports/STATS_ks_recency.md
python -m research.analysis.make_figures           # → figures/*.pdf + *.png

cd .. && pdflatex sample && bibtex sample && pdflatex sample && pdflatex sample
```

### Smoke tests

```
python -m research.smoke.smoke_test
python -m research.smoke.smoke_gpu
python -m research.smoke.test_batched_future_items
```

## Headline numbers (Baby, seed=42)

| Recipe | BPR HR@20 | NeuMF HR@20 | LightGCN HR@20 |
|---|---|---|---|
| shuffle+uniform (vanilla) | 0.0208 | 0.0262 | 0.0278 |
| win10+causal (faithful) | **0.0550** | 0.0365 | 0.0357 |
| Δ vs vanilla | **+164%** | +39% | +28% |

Prequential macro HR@20 (BPR):
- Baby: vanilla 0.0745 → faithful **0.1000** (ρ vanilla 35.5%)
- Cellphone: 0.0365 → **0.0825** (ρ 42.8%)
- Healthcare: 0.0477 → **0.0631** (ρ 34.7%)

The faithful recipe wins every (dataset, slice) cell on HR@20 (15/15); paired t-test p ≈ 4 × 10⁻⁸. Multi-seed means/stds replace these once Phases B/C/D land.

## Pitfalls + lessons

- **DGL on Windows is painful.** LightGCN only works in the WSL `leakage` env. cu121 wheels work; cu124 has a torch pin conflict. Never silently fall back to CPU DGL.
- **cornac's `Experiment(save_dir=…)` triggers `model.save(save_dir)` as a side effect** — ~50 MB pickle per cell into `research/logs/<recipe>/`. Storage waste but not breaking; pickles are gitignored. Don't drop `save_dir` or cornac writes `CornacExp-*.log` to cwd instead.
- **`future_items_pct` is identically 0 under `exclude_unknowns=True`** (the candidate pool is restricted to training items, all of which have first-seen ≤ train cutoff < test timestamp). It was eating ~64 min/cell on healthcare; removed from the 2×2 ablation runners. Still meaningful for the split-comparison Table II (`smoke/baseline_bpr_leakage.py`).
- **`val_set` matters for test eval.** cornac's `ranking_eval(test_set=…, val_set=…)` removes val-positive items from the test candidate pool; passing `val_set=None` corrupts the metric. Always go through `cornac.Experiment` rather than calling `ranking_eval` directly.
- **Per-cell checkpointing is the resume granularity** — kill mid-cell loses that cell only.

## Related external files

- `../sample.tex` — paper draft (IEEEtran conference template).
- `../references.bib` — bibliography.
- `../cornac/eval_methods/timestamp_split.py` — `TimestampSplit` (data-side primitive we contributed upstream; commit `fcab86b`).
