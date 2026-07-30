# Causally Isolated Training for Leakage-Free Recommendation

Research code for **causal negative sampling**, built on top of
[cornac](https://github.com/PreferredAI/cornac) — imported, never vendored.

## The problem

A deployed recommender at time *t* can only learn from what exists at *t*.
Standard offline training breaks this: it samples negatives from the full
catalog regardless of timestamp, so the model is trained to rank items that did
not yet exist below the positives it saw. Those comparisons could never have
happened in production, and the accuracy they buy is not real.

**Causal negative sampling** closes it: a negative for an interaction observed
at time *t* is drawn only from items whose first appearance is at-or-before *t*.

The ablation is single-variable — `uniform` (vanilla) vs `causal` — with a probe
that measures the leakage independently of accuracy:

> **Counterfactual-negative rate ρ** — the fraction of training negatives whose
> first-seen timestamp post-dates their paired positive. Zero by construction
> under causal sampling; 25–45% under uniform.

## The design: the rule lives in the data loader

`TimeAwareDataset` (`lib/timeaware_data.py`) subclasses `cornac.data.Dataset`
and overrides the two iterators cornac's models draw negatives from. Because the
rule lives there, **cornac's own models get it with no model-side code**:

| Model | How it asks for negatives | What we do |
|---|---|---|
| **LightGCN** | `train_set.uij_iter(...)` | nothing — stock `cornac.models.LightGCN` |
| **NeuMF** | `train_set.uir_iter(..., num_zeros=num_neg)` | nothing — stock `cornac.models.NeuMF` |
| **BPR** | compiled Cython over `train_set.matrix` | our own NumPy BPR |

Neither cornac model exposes a sampling option, so the arm is a property on the
training split rather than a model argument:

```python
from research.lib.data import build_eval_method

em = build_eval_method("baby", neg_sampling="causal")
model = cornac.models.LightGCN(...)      # completely unmodified
# ... run it; every negative it draws is causal
print(em.train_set.counterfactual_rate)  # -> 0.0
```

Both arms use the same no-rejection draw, so they differ in exactly one thing:
the item pool.

**BPR is the exception.** cornac's BPR extracts `train_set.matrix` — a CSR
carrying no timestamps — and samples inside a compiled OpenMP loop, so the rule
cannot reach it without editing Cython. `lib/bpr_cpu.py` is a pure-NumPy BPR
reading the *same* first-seen index, so "causal" has one definition. It also
means ρ comes off the **model** for BPR (`model.counterfactual_rate`) but off the
**training split** for the other two.

## Quickstart

```bash
pip install -r requirements.txt          # LightGCN also needs: pip install -e ".[lightgcn]"

# Proves cornac's models really do sample through our loader.
# Synthetic data — needs no dataset CSVs:
python -m research.smoke.test_cornac_causal
```

```bash
export RESEARCH_DATA_DIR=./data                 # Windows: $env:RESEARCH_DATA_DIR = "$PWD\data"

python -m research.smoke.smoke_counter          # causal rho = 0%, uniform ~38%

python -m research.runners.ablation_bpr      --seeds 42,123,2026
python -m research.runners.ablation_neumf    --seeds 42,123,2026
python -m research.runners.ablation_lightgcn --seeds 42,123,2026

python -m research.analysis.aggregate_ablation  # -> reports/RESULTS_ablation.md
python -m research.analysis.make_figures        # -> figures/*.pdf + *.png
```

Cells are checkpointed per-recipe, so killing a run loses only the in-flight cell.

Tuning runs coordinate descent on the vanilla arm, so hyperparameters are not
chosen under the mechanism being evaluated:

```bash
python -m research.runners.tuning --model bpr   # or neumf / lightgcn
```

## Datasets

Amazon Reviews 2023 **5-core** `rating_only` files —
`user_id,parent_asin,rating,timestamp` with a header, read as UIRT. Take the
**timestamp** benchmark variant, not `last_out` (leave-one-out).

| Dataset | Rows | Users / items | |
|---|---|---|---|
| Musical Instruments | 511,836 | 57,439 / 24,587 | default |
| Baby Products | 1,241,083 | 150,777 / 36,013 | default |
| Cell Phones & Acc. | 2,752,785 | 380,999 / 111,480 | default |
| Health & Household | 7,176,552 | 796,054 / 184,346 | `--datasets healthcare` |

Layout under `RESEARCH_DATA_DIR`:

```
musical_dataset/Musical_Instruments.csv
baby_dataset/Baby_Products.csv
cellphone_dataset/Cell_Phones_and_Accessories.csv
healthcare_dataset/Health_and_Household.csv
```

Only the combined CSV is needed — not the `.train/.valid/.test` files.
`CausalTimestampSplit` splits in-code at the cutoffs in `lib/data.py:DATASETS`,
which reproduce the official timestamp partition exactly.

## Layout

`research/` is a Python package — run everything as `python -m research.<...>`.

```
research/
  paths.py                        filesystem anchors; env-overridable
  lib/
    timeaware_data.py             TimeAwareDataset (uij_iter + uir_iter overrides,
                                  rho probe) + CausalTimestampSplit
    causal_negative_sampler.py    NumPy sampler for our BPR (same index)
    data.py                       DATASETS registry + build_eval_method()
    bpr_cpu.py                    BPRMiniBatch — the one hand-written model
    causal_sampling.py            eval-time probes
    faithfulness_metrics.py       recommendation-recency distribution
    ablation_harness.py           RECIPES, set_recipe, checkpointing
    tuning_config.py              every hyperparameter, with its source quote
                                  and a status label — the source of truth
  runners/     ablation_{bpr,neumf,lightgcn}, tuning
  analysis/    aggregate_ablation, make_figures
  smoke/       self-tests
data/          dataset CSVs
```

Generated output lands in `research/{results,reports,figures,logs}/`; override
with `RESEARCH_OUTPUT_DIR`.

## Gotchas

- **`RESEARCH_DATA_DIR` defaults to the project root**, not `./data`. Unset, the
  loader looks for `<repo>/baby_dataset/...` and reports everything missing.
- **cornac's `Dataset.rng` is a legacy `RandomState`** — it has `.randint` but no
  `.integers`. Anything the dataset drives must not assume a modern `Generator`.
- **cornac's NeuMF defaults to `backend="tensorflow"`**, which has no GPU on
  native Windows. The runners pass `backend="pytorch"`; both take the same
  `uir_iter` path.
- **DGL on Windows is painful**, and LightGCN needs it. cu121 wheels work under
  WSL; cu124 conflicts with the torch pin. Never silently fall back to CPU DGL.
- **Reset ρ between cells.** `set_recipe` does it; the counters live on the split,
  which is reused across cells.
- **Always go through `cornac.Experiment`.** Calling `ranking_eval` directly with
  `val_set=None` leaves val-positive items in the test candidate pool and
  corrupts the metric.

`test_cornac_causal` is the test that matters: if cornac ever changes which
iterator a model uses, the ablation would silently become a no-op, and accuracy
numbers alone would not reveal it.

## Open items

- Making cornac's BPR causal would mean threading timestamps into `_fit_sgd` and
  recompiling Cython — then `bpr_cpu.py` could go and everything would be stock.
- `faithfulness_metrics.recommendation_recency_distribution` currently has no
  caller; it should be computed per cell alongside ρ and KS-tested across arms.
