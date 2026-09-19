import numpy as np


def _rank_of_positive(scores, pos_items, seen_mask=None):
    if seen_mask is not None:
        scores = np.where(seen_mask, -np.inf, scores)
    pos_scores = scores[np.arange(len(pos_items)), pos_items]
    scores[np.arange(len(pos_items)), pos_items] = pos_scores
    return 1 + (scores > pos_scores[:, None]).sum(axis=1)


def score_batch(adapter, users, pos_items, seen, k=20, chunk=512):
    hits = 0
    ndcg = 0.0
    auc = 0.0
    for start in range(0, len(users), chunk):
        u = users[start:start + chunk]
        p = pos_items[start:start + chunk]
        scores = adapter.score(u)
        mask = np.zeros_like(scores, dtype=bool)
        for row, uid in enumerate(u):
            already = seen.get(int(uid))
            if already:
                mask[row, list(already)] = True
        rows = np.arange(len(p))
        n_cand = scores.shape[1] - mask.sum(axis=1) + mask[rows, p]
        ranks = _rank_of_positive(scores, p, mask)
        inside = ranks <= k
        hits += int(inside.sum())
        ndcg += float((1.0 / np.log2(ranks[inside] + 1.0)).sum())
        many = n_cand > 1
        auc += float(((n_cand[many] - ranks[many]) / (n_cand[many] - 1)).sum())
    return hits, ndcg, auc, len(users)


def run_prequential(adapter, train_set, batch_size=4096, warmup_frac=0.2,
                    k=20, warmup_epochs=5, verbose=True, rng=None):
    u_arr, i_arr, _ = train_set.uir_tuple
    u_arr = np.asarray(u_arr, dtype=np.int64)
    i_arr = np.asarray(i_arr, dtype=np.int64)
    ts_arr = np.asarray(train_set.timestamps, dtype=np.int64)
    rng = np.random.default_rng(0) if rng is None else rng

    order = np.argsort(ts_arr, kind="stable")
    n = len(order)
    warmup_end = int(n * warmup_frac)

    adapter.prepare(train_set)

    warm = order[:warmup_end]
    for _ in range(warmup_epochs):
        for start in range(0, len(warm), batch_size):
            adapter.observe(warm[start:start + batch_size])

    seen = {}
    for row in warm:
        seen.setdefault(int(u_arr[row]), set()).add(int(i_arr[row]))

    records = []
    stream = order[warmup_end:]
    for b, start in enumerate(range(0, len(stream), batch_size)):
        rows = stream[start:start + batch_size]
        if len(rows) == 0:
            continue

        eval_rows = rows
        if adapter.eval_sample is not None and len(rows) > adapter.eval_sample:
            eval_rows = rows[rng.choice(len(rows), adapter.eval_sample,
                                        replace=False)]

        hits, ndcg, auc, cnt = score_batch(adapter, u_arr[eval_rows],
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
            "AUC": auc / cnt,
        })

        adapter.observe(rows)
        for uid, iid in zip(u_arr[rows], i_arr[rows]):
            seen.setdefault(int(uid), set()).add(int(iid))

        if verbose and b % 10 == 0:
            print(f"  [{adapter.sampler_kind}] batch {b}: "
                  f"HR@{k}={hits / cnt:.5f} scored={cnt}", flush=True)

    return records
