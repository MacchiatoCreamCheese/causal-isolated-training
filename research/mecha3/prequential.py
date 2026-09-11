"""Mechanism 3: prequential ("test-then-train") evaluation along the timeline.

The question this answers is *how performance moves as the model ages*, which the
single-cutoff ablation cannot see: it trains once and reports one number, so there
is no time axis to plot against.

Test-then-train
---------------

Walk the training rows in chronological order. For each batch::

    1. SCORE it   -- the model has been fitted on strictly earlier rows only
    2. TRAIN on it -- now it, too, is part of the past

Every batch yields one measurement, so a musical-sized run gives ~100 points
rather than the handful a retrain-at-each-cutoff scheme would afford. That is the
difference between a line and a dot-to-dot.

It is also *cheaper* than retraining. A rolling-origin design costs N full
trainings per arm; this costs one chronological pass, because the model is never
rebuilt -- each batch simply becomes training data after it has been scored.

No leakage, by construction: at batch *b* the parameters have seen batches
`0..b-1`, all of which are strictly earlier in time. This is the "timeline scheme"
Ji et al. (TOIS 2023 §5) argue for, and the reason they note it "requires all
recommender models to be incremental in nature".

All three models, two routes
----------------------------

Incrementality is the requirement, and cornac's models do not offer it: NeuMF and
LightGCN run their whole schedule inside one `fit()` with no per-batch seam. The
answer is not to copy them. `adapters.py` imports cornac's *architectures* and
*losses* unchanged -- `NeuMF._build_model_pt()`, LightGCN's `Model`,
`Model.loss_fn`, `construct_graph` -- and writes only the five-line
"forward, loss, backward, step" that cornac keeps inside `fit()`. The part that
must not drift stays cornac's; the part that must be interruptible is ours.

Our NumPy BPR needs less: `lib/bpr_cpu.py` already exposes `train_batch`.

One model-specific caveat each. NeuMF scores an MLP per (user, item) pair rather
than a dot product, so each point is measured on a random subset of its batch --
unbiased, noisier. LightGCN's *graph* is training data, so it is rebuilt from
rows seen so far as the stream advances; handing it the full graph would let
future edges reach current predictions through convolution.

Arms
----

`uniform` vs `causal`. Temporal batching is not a separate arm here: prequential
*requires* chronological batches, so Mechanism 2's ordering is inherent to the
protocol rather than a variable within it.

Caveats worth stating in any writeup
------------------------------------

**One pass is one epoch.** BPR normally runs to a 1000-epoch ceiling with early
stopping; a single chronological pass leaves it heavily undertrained, so absolute
values sit well below the ablation's. Both arms get an identical budget, so the
comparison holds -- but the level does not transfer.

**Early points are the noisiest and least trained.** `warmup_frac` trains normally
on an initial prefix before streaming begins, which is what `main`'s runner
achieved with its "1 warmup + 5 tests" slicing. Points before the warm-up
boundary are not reported.

**Per-batch metrics are jumpy.** ~4k interactions per point is enough to be
meaningful and not enough to be smooth; plot a rolling mean with the raw points
faint behind it.
"""

import numpy as np


def _rank_of_positive(scores, pos_items, seen_mask=None):
    """1-based rank of each row's true item, without sorting.

    `scores` is `(B, num_items)`. The rank is one plus the number of items scoring
    strictly higher, which is a single comparison-and-sum rather than an
    `argsort` -- at 22k items and 4k rows per batch, sorting would dominate the
    run.

    `seen_mask` marks items the user has already interacted with *so far in the
    stream*. They are pushed to -inf so they cannot occupy ranks above the target:
    recommending something the user already has is not a hit, and leaving them in
    would depress every rank uniformly and understate the metric.
    """
    if seen_mask is not None:
        scores = np.where(seen_mask, -np.inf, scores)
    pos_scores = scores[np.arange(len(pos_items)), pos_items]
    # The positive itself may have been masked (it is "seen" only *after* this
    # step, so it should not be) -- restore it before comparing.
    scores[np.arange(len(pos_items)), pos_items] = pos_scores
    return 1 + (scores > pos_scores[:, None]).sum(axis=1)


def score_batch(adapter, users, pos_items, seen, k=20, chunk=512):
    """`(hits, ndcg_sum, n)` for one batch, scored before it is trained on.

    A single relevant item per row, so HR@k is `rank <= k` and NDCG@k is
    `1/log2(rank+1)` when it lands inside k -- the standard leave-one-out form,
    and what makes a per-interaction curve possible at all.

    Scored in chunks: a full `(4096, 22753)` score matrix is ~370 MB, which is
    both wasteful and enough to matter once per batch for a hundred batches.
    """
    hits = 0
    ndcg = 0.0
    for start in range(0, len(users), chunk):
        u = users[start:start + chunk]
        p = pos_items[start:start + chunk]
        scores = adapter.score(u)
        mask = np.zeros_like(scores, dtype=bool)
        for row, uid in enumerate(u):
            already = seen.get(int(uid))
            if already:
                mask[row, list(already)] = True
        ranks = _rank_of_positive(scores, p, mask)
        inside = ranks <= k
        hits += int(inside.sum())
        ndcg += float((1.0 / np.log2(ranks[inside] + 1.0)).sum())
    return hits, ndcg, len(users)


def run_prequential(adapter, train_set, batch_size=4096, warmup_frac=0.2,
                    k=20, warmup_epochs=5, verbose=True, rng=None):
    """Stream `train_set` in time order, scoring each batch before training on it.

    Returns a list of per-batch records: position in the stream, the batch's time
    span, how many interactions were scored, and HR@k / NDCG@k measured on them.

    `adapter` supplies the three model-specific operations -- `prepare`, `score`,
    `observe` -- so this function is the protocol and nothing else. See
    `adapters.py`.
    """
    u_arr, i_arr, _ = train_set.uir_tuple
    u_arr = np.asarray(u_arr, dtype=np.int64)
    i_arr = np.asarray(i_arr, dtype=np.int64)
    ts_arr = np.asarray(train_set.timestamps, dtype=np.int64)
    rng = np.random.default_rng(0) if rng is None else rng

    order = np.argsort(ts_arr, kind="stable")
    n = len(order)
    warmup_end = int(n * warmup_frac)

    adapter.prepare(train_set)

    # --- warm-up: ordinary training on the earliest prefix ---------------------
    # Without it the first reported points describe a model that has seen almost
    # nothing, and the curve opens with a spike of noise rather than a signal.
    warm = order[:warmup_end]
    for _ in range(warmup_epochs):
        for start in range(0, len(warm), batch_size):
            adapter.observe(warm[start:start + batch_size])

    seen = {}
    for row in warm:
        seen.setdefault(int(u_arr[row]), set()).add(int(i_arr[row]))

    # --- stream: score, then absorb -------------------------------------------
    records = []
    stream = order[warmup_end:]
    for b, start in enumerate(range(0, len(stream), batch_size)):
        rows = stream[start:start + batch_size]
        if len(rows) == 0:
            continue

        # NeuMF scores an MLP per (user, item) pair rather than a dot product, so
        # it evaluates a random subset -- an unbiased but noisier estimate. Other
        # adapters score the whole batch.
        eval_rows = rows
        if adapter.eval_sample is not None and len(rows) > adapter.eval_sample:
            eval_rows = rows[rng.choice(len(rows), adapter.eval_sample,
                                        replace=False)]

        hits, ndcg, cnt = score_batch(adapter, u_arr[eval_rows],
                                      i_arr[eval_rows], seen, k=k)
        ts = ts_arr[rows]
        records.append({
            "batch": b,
            "n": int(cnt),
            "n_rows": int(len(rows)),
            "ts_start": int(ts.min()),
            "ts_end": int(ts.max()),
            f"HitRatio@{k}": hits / cnt,
            f"NDCG@{k}": ndcg / cnt,
        })

        adapter.observe(rows)
        for uid, iid in zip(u_arr[rows], i_arr[rows]):
            seen.setdefault(int(uid), set()).add(int(iid))

        if verbose and b % 10 == 0:
            print(f"  [{adapter.sampler_kind}] batch {b}: "
                  f"HR@{k}={hits / cnt:.5f} scored={cnt}", flush=True)

    return records
