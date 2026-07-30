# Causally Isolated Training for Leakage-Free Recommendation

Research code for training-time causal isolation in recommender systems, built
**on top of** [cornac](https://github.com/PreferredAI/cornac) — cornac is
`pip install`ed and imported, never vendored. There is **no cornac source
checkout** in this project; everything runs against a released cornac wheel
(`cornac>=2.4.0`, which is where `TimestampSplit` and the validation-set
plumbing this code depends on first shipped).

## What's here

```
research/
  paths.py            filesystem anchors (data dir + output dirs; env-overridable)
  lib/
    timeaware_data.py    TimeAwareDataset + CausalTimestampSplit  ← the generic
                         causal data loader (Mechanism 1) — see below
    causal_negative_sampler.py   per-fit Numpy/Torch samplers models share
    data.py              DATASETS registry + build_eval_method() factory
    bpr_cpu.py, bpr_gpu.py, temporal_batching.py   BPR (+ windowed)
    neumf.py, lightgcn.py                          NeuMF, LightGCN
    causal_sampling.py   evaluation-time leakage probes
    faithfulness_metrics.py, tuning_config.py, ablation_harness.py
  runners/            ablation_{bpr,neumf,lightgcn}, tuning, prequential, k_ablation
  analysis/           aggregate_*, stats_*, make_figures
  smoke/              quick self-tests
  reports/            HYPERPARAMETERS.md, SUMMARY.md, DEFERRED.md
data/                 dataset CSVs go here (see data/README.md)
```

## The generic causal data loader

The professor's "generic data loader" is the **causal negative sampling**
mechanism, lifted out of the individual models into one reusable component at
cornac's data layer:

- **`TimeAwareDataset(cornac.data.Dataset)`** adds a third `neg_sampling` option
  to cornac's own `Dataset.uij_iter` — `"causal"`, alongside the built-in
  `"uniform"` / `"popularity"`. A negative item for an interaction observed at
  time *t* is drawn only from items whose first appearance is at-or-before *t*,
  so a model is never trained to rank a not-yet-existent item below a positive.
- **`CausalTimestampSplit(TimestampSplit)`** produces a training split that *is*
  a `TimeAwareDataset`, via a `build()` override.

Any cornac-style model that trains by consuming `train_set.uij_iter(...)` gets
causal sampling for free — it is not welded into BPR/NeuMF/LightGCN. Our GPU
models keep an on-device sampler for speed (`causal_negative_sampler.py`) that
reads the *same* per-item first-seen index, so there is a single source of truth.

## Quickstart

```bash
pip install -r requirements.txt          # or: pip install -e .
# LightGCN also needs dgl:  pip install -e ".[lightgcn]"

export RESEARCH_DATA_DIR=./data           # where the *_dataset/ folders live
python -m research.smoke.smoke_counter    # causal cf-rate = 0%, uniform ~38%
python -m research.smoke.smoke_test       # CPU BPR trains 2 epochs
```

Run an ablation / tuning:

```bash
python -m research.runners.ablation_bpr --seeds 42,123,2026 --datasets baby
python -m research.runners.tuning --model bpr
python -m research.analysis.aggregate_2x2
```

Outputs land in `research/results/`, `research/logs/`, `research/figures/`
(override with `RESEARCH_OUTPUT_DIR`).

## Using the loader from your own model

```python
from research.lib.data import build_eval_method

em = build_eval_method("baby")          # a CausalTimestampSplit
train = em.train_set                     # a TimeAwareDataset
for users, pos, neg in train.uij_iter(batch_size=1024, shuffle=True,
                                       neg_sampling="causal"):
    ...                                  # neg never post-dates its positive
```

---

## Relocation & first-time git setup

This folder is self-contained. To make it its own repository:

```bash
# 1. Move it wherever you want it to live (example):
mv causal-isolated-training ~/projects/causal-isolated-training
cd ~/projects/causal-isolated-training

# 2. Create an isolated environment and install deps (pulls cornac from PyPI):
python -m venv .venv && source .venv/bin/activate      # or conda create ...
pip install -r requirements.txt

# 3. Provide the datasets (see data/README.md): copy the three Amazon CSVs into
#    ./data/ (or `export RESEARCH_DATA_DIR=/path/to/existing/data`).
export RESEARCH_DATA_DIR=./data

# 4. Sanity-check it runs standalone (no cornac source tree needed):
python -m research.smoke.smoke_counter

# 5. Initialize git:
git init
git add .
git commit -m "Initial commit: causally isolated training (standalone)"
# git remote add origin <your-repo-url> && git push -u origin main
```

Notes:
- The importable package is `research` (invoke everything as `python -m research.<...>`).
- `.gitignore` already excludes datasets and generated outputs.
- You can now delete the old cornac repo — nothing here points back at it.
