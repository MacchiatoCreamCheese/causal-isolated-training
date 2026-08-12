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
| **NeuMF** | `train_set.uir_iter(..., num_zeros=num_neg)` | nothing that affects sampling — stock `cornac.models.NeuMF` with only `save()` stubbed out (see below) |
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

Within a model both arms draw the same way and reject collisions the same way,
so they differ in exactly one thing: the item pool.

**Collision rejection, per model.** A negative drawn at random is occasionally
an item the user actually interacted with — a positive masquerading as a
negative. Rendle 2009 excludes these by definition (`j ∈ I \ I_u⁺`) and cornac
rejects them too, but *cornac uses a different rule for each entry point*, and
we match it rather than imposing one rule everywhere:

| cornac site | Filter | Why |
|---|---|---|
| `uij_iter` → LightGCN | `dok[u,j] >= pos_rating` | pairwise BPR loss asserts only an ordering, so an item the user rated *below* the positive is a valid — indeed observed rather than assumed — negative |
| `uir_iter` → NeuMF | `dok[u,j] > 0` | pointwise BCE labels the row 0, which is factually wrong for any item the user touched, whatever its rating |
| BPR Cython (`has_non_zero`) | any observed | same as above; matched by our NumPy BPR |

The rule tracks the *loss*, not the dataset. It matters because `load_uirt`
reads raw 1–5 stars with no binarization and 70.6% of musical's ratings are 5.0,
so for most positives cornac admits the user's own 1–4 star items as negatives.
On implicit data the two rules coincide and the distinction vanishes.

Rejection is a *bounded* vectorized retry (`MAX_REJECT_ROUNDS`), not cornac's
unbounded `while`: the earliest interaction in the log has a causal pool of
exactly one item — itself — so no valid negative exists and an unbounded loop
would spin forever on that row. A handful of collisions therefore survive; they
are counted by `collision_rate` and written into every seed JSON. On musical
that is 4 out of 427,957 draws (0.001%).

`smoke/test_sampler_equivalence.py` checks this correspondence rather than
asserting it: every negative we emit is compared against cornac's own predicate
evaluated on cornac's own `dok_matrix`, and the count of illegal ones must equal
the counted residuals exactly. It also compares draw frequencies against
cornac's real per-element loop, calibrated against our own sampler's
run-to-run self-distance (on musical: 0.4040 vs a 0.4037 noise floor).

**BPR is the exception.** cornac's BPR extracts `train_set.matrix` — a CSR
carrying no timestamps — and samples inside a compiled OpenMP loop, so the rule
cannot reach it without editing Cython. `lib/bpr_cpu.py` is a pure-NumPy BPR
reading the *same* first-seen index, so "causal" has one definition. It also
means ρ comes off the **model** for BPR (`model.counterfactual_rate`) but off the
**training split** for the other two.

## Quickstart

Everything runs under **WSL**, in the `leakage24` conda env (Python 3.12,
cornac 2.6.0 from PyPI, torch 2.4.1+cu121 with CUDA, dgl 2.4.0+cu121). LightGCN
needs dgl, which is why the environment is WSL and not native Windows.

cornac comes from PyPI, not a source checkout: the repo uses only cornac's
public API, which is what the `cornac>=2.4.0` pin claims and what a plain
`pip install` proves. (An earlier `leakage` env held cornac 2.3.5 editable from
`../cornac`. It has been retired — 2.3.5 is *below* the declared pin, and being
an editable local fork it could not distinguish "works against cornac" from
"works against my copy of cornac".)

```bash
wsl
conda activate leakage24
cd /mnt/c/Users/nguye/uniyear/causal-isolated-training

# Proves cornac's models really do sample through our loader, and that our
# rejection filter matches cornac's. Synthetic data — needs no dataset CSVs:
python -m research.smoke.test_cornac_causal          # -> ALL CHECKS PASSED (10/10)
python -m research.smoke.test_sampler_equivalence
```

`test_cornac_causal` reports `10/10` only when all ten checks actually ran. Two
of them drive a real `cornac.models.LightGCN` and so need dgl; without it the
test **fails** rather than passing quietly, since those are the checks covering
the `uij_iter` path. On native Windows, where dgl is unavailable, waive them
explicitly and get an honest `PASSED 8/10 - 2 skipped`:

```bash
python -m research.smoke.test_cornac_causal --allow-missing-dgl
# or: RESEARCH_ALLOW_MISSING_DGL=1
```

To rebuild the env from scratch: `conda create -n leakage24 python=3.12`, then
`pip install -r requirements.txt`, then `pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html`.

Version coverage is single-point (2.6.0). The useful second point would be the
pin floor, 2.4.0, pip-installed in CI — not the retired 2.3.5, which is both in
the past and below the floor.

```bash
export RESEARCH_DATA_DIR=./data                 # Windows: $env:RESEARCH_DATA_DIR = "$PWD\data"

python -m research.smoke.smoke_counter          # causal rho = 0%, uniform ~38%
python -m research.smoke.test_sampler_equivalence --dataset musical

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
    fixtures.py                   synthetic UIRT splits (implicit + rated)
    test_cornac_causal.py         cornac's models really sample through us
    test_sampler_equivalence.py   our filter == cornac's, measured
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
- **`NCFBase.save()` raises after writing the pickle.** In cornac 2.6.0 it calls
  `Recommender.save()` first — the `.pkl` and `.meta` land fine — and only then
  hits `raise NotImplementedError()` on the `backend="pytorch"` branch, whose
  TODO is the *separate weight export* (the `.h5` sidecar the TensorFlow branch
  writes), not the pickle. Since `Experiment.run()` calls `save()` whenever
  `save_dir` is set, and does so *after* training and evaluating, a plain NeuMF
  trains for minutes, writes its pickle, then aborts before `write_partial` can
  checkpoint the metrics. `runners/ablation_neumf.py` therefore subclasses
  `save()` to call `Recommender.save()` directly, skipping the raise and nothing
  else — you still get the same `.pkl` + `.meta` as every other cornac model,
  and it round-trips (verified: reloaded model reproduces identical scores).
- **Run in WSL, not native Windows.** LightGCN needs dgl; cu121 wheels work under
  WSL, cu124 conflicts with the torch pin. Never silently fall back to CPU DGL.
  The other two models run fine on Windows if you only need those — but
  `test_cornac_causal` will fail there rather than skip, since without dgl it
  cannot verify the `uij_iter` path. Waive it with `--allow-missing-dgl` and read
  the `8/10 - 2 skipped` summary as what it is: a partial run.
- **rho is counted in both arms, and the draw count is asserted with it.** Under
  causal sampling rho is 0 *by construction* — a positive's own item is always in
  its causal prefix, so the prefix is never empty and no future item is
  reachable. That makes `rho == 0` a weak assertion, and it was weaker still:
  the counter used to be gated on `uniform`, and `counterfactual_rate` returns
  `0.0` when nothing was drawn at all. Both holes are closed, and both were
  verified by sabotage rather than by argument:

  | Sabotage | rho | Old verdict | Now |
  |---|---|---|---|
  | `causal_draw` returns uniform items | 31% | — | 4 causal checks FAIL |
  | override silently degrades to cornac uniform | 0.00% | **PASS** | FAIL: `over 0 draws <-- NO DRAWS: loader was bypassed` |

  That second row is the regression this whole test file exists to catch, and it
  used to slip through. Counting costs ~1s per 1000 epochs (measured: 1.5–2.9 ns
  per draw, a vectorized gather-compare, ~0.5% of the sampling step) — cheap
  enough that gating it to save time is a false economy.
- **Seed the eval method, not just the model.** cornac's models draw negatives
  from the *split*, so a model-side seed does not cover them. Without
  `build_eval_method(..., seed=...)`, `Dataset.rng` silently falls back to
  numpy's global singleton: negative sampling becomes irreproducible and
  multi-seed runs stop being controlled replicates. The runners pass it.
- **`set_recipe` resets the split's RNG.** That is what makes the arms a paired
  comparison — same users, same positives, same batch order, only the negative
  pool differs — and what stops results depending on the order the cells ran in.
- **Always go through `cornac.Experiment`.** Calling `ranking_eval` directly with
  `val_set=None` leaves val-positive items in the test candidate pool and
  corrupts the metric.

`test_cornac_causal` is the test that matters: if cornac ever changes which
iterator a model uses, the ablation would silently become a no-op, and accuracy
numbers alone would not reveal it. `test_sampler_equivalence` guards the other
half — that the uniform arm we compare everything against is cornac's sampler
and not a stricter one of our own invention.

## Open items

- Making cornac's BPR causal would mean threading timestamps into `_fit_sgd` and
  recompiling Cython — then `bpr_cpu.py` could go and everything would be stock.
- `faithfulness_metrics.recommendation_recency_distribution` currently has no
  caller; it should be computed per cell alongside ρ and KS-tested across arms.
- Version coverage is single-point. Add cornac 2.4.0 (the pin floor),
  pip-installed in CI, as the second point.
- **Unequal training budgets across models.** NeuMF runs a fixed 20 epochs with
  no early stopping; BPR and LightGCN run up to 1000 with it. Each is that
  paper's own protocol, so every model sits at its author-intended operating
  point — but a cross-model accuracy gap therefore confounds architecture with up
  to a 50× difference in gradient steps. The ablation claim is unaffected (it is
  within-model, and both arms share a budget); cross-model rows in the results
  table are context, not evidence.
