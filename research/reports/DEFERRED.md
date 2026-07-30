# Deferred work

Things we deliberately did **not** do in past plans but should remember. Promote items to a plan when scheduled; delete the entry once done.

## 1. Refactor `NeuMFRecommender` to subclass cornac's `NeuMF` / `GMF` / `MLP` recommenders

`research/lib/neumf.py` currently inlines all three training phases inside one `fit()` with a shared `self.n_epochs`. Cornac's native pattern is three separate Recommender objects (`cornac.models.GMF`, `cornac.models.MLP`, `cornac.models.NeuMF`), each inheriting from `NCFBase` (`cornac/models/ncf/recom_ncf_base.py`), each with its own `num_epochs` and **built-in early stopping** via an `early_stopping={"min_delta": ..., "patience": ...}` kwarg.

Refactoring to use those three recommenders would:

- Give us per-phase epoch counts for free (matching cornac's intended use).
- Give us early stopping for free on all three phases.
- Eliminate our hand-rolled `_train_module`.

**Cost:** rewiring the research mechanisms (causal sampler, windowed order) into cornac's training loop. Cornac's NCF inner loop is in `recom_ncf_base.py`; we'd need to subclass and override the per-batch sampler hook. Worth scoping when we tackle early stopping for non-LightGCN models.

## 2. BPR adaptive negative sampling

NewBPR §5.3's recipe for the global temporal split includes adaptive negative sampling alongside separated λs (done) and disabled biases (done). Adaptive sampling draws negatives with high current score (hard negatives) — composes with our causal-pool restriction.

Out of scope so far because it adds a *new* mechanism orthogonal to our research axis. Adding it would need its own ablation cell (`adaptive+causal` vs `uniform+causal` etc.) to interpret cleanly. Decide whether to add it as a fourth recipe dimension or as a tuning knob inside the existing cells.

## 3. NeuMF early stopping

He 2017 doesn't pin a patience number; cornac NCF defaults to 20 epochs with optional early stopping. Wiring this requires either:

- (a) the refactor in §1 (free for NeuMF, since cornac's `NCFBase` already has `early_stopping`), or
- (b) implementing `monitor_value` + epoch-loop hooks on our custom recommenders (the path LightGCN took in `research/lib/lightgcn.py:285-298`, and BPR took in `research/lib/bpr_gpu.py`).

BPR's early stopping is already wired (2026-06-06, approach (b), NewBPR §5.2 patience=13 + NDCG@20 monitor + 1000-epoch ceiling). Same pattern would work for NeuMF if we don't want the refactor.

## 4. NeuMF Gaussian init for MLP linear layers

Cornac's `MLP._init_weight` uses `xavier_uniform_` for the linear layers, but He 2017 §4.1 specifies *"randomly initialized model parameters with a Gaussian distribution (with a mean of 0 and standard deviation of 0.01)"* — i.e. Gaussian std=0.01 for *everything*, including the MLP linears. Embeddings already match (cornac uses `nn.init.normal_(std=1e-2)`). The discrepancy is small and arguably cornac's choice is the standard upgrade for deeper MLPs; only revisit if reproducibility-vs-paper becomes a reviewer issue.

## 5. Contribute `neg_sampling="causal"` upstream to cornac

`lib/timeaware_data.TimeAwareDataset` adds `neg_sampling="causal"` to
`Dataset.uij_iter` by subclassing. If it proves broadly useful, propose it
upstream (`cornac/data/dataset.py`, alongside `uniform`/`popularity`) so no
subclass is needed — it would require the base `Dataset` to carry a per-item
first-seen index built from `timestamps`. Optional; the subclass works today and
keeps the research code decoupled from a specific cornac version.

## 6. LightGCN Adam optimizer state across windowed graph rebuilds (note: documenting only — deliberate research choice)

`research/lib/lightgcn.py:214` creates the Adam optimizer once before the window loop. Adam's momentum state persists across window boundaries even as the graph is rebuilt. This is a *deliberate* research-mechanism choice (warm-start, parameters and optimizer state both carry over). Document in the paper rather than fix.
