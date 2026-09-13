"""The backfill must delete only trials trained around a *test*-selected value.

When validation and test disagree at a knob, what decides validity is which value
the tuner actually carried into the later knobs:

  - a cell tuned before the fix carried the test pick, so its later trials were
    trained at the wrong config and must go;
  - a cell tuned after the fix carried the validation pick, so its later trials
    are valid and must be kept.

Deleting in the second case once wiped valid seed-2026 trials on a machine. The
cells here are hand-built, so no training is needed.
"""

import json

from ..runners import tuning_backfill as bf

ORDER = [("a", [2]), ("b", [20])]
BASE = {"a": 1, "b": 10}


def _write(path, config, val, test):
    path.parent.mkdir(parents=True, exist_ok=True)
    m_val, m_test = {"NDCG@20": val}, {"NDCG@20": test}
    path.write_text(json.dumps({"config": config, "val_metrics": m_val,
                                "test_metrics": m_test, "metrics": m_val,
                                "select_split": "validation"}))


def _cell(tmp_path, carried_a):
    """Knob `a`: validation prefers the default (1), test prefers 2.

    The trials of knob `b` were trained with `a` fixed at `carried_a`.
    """
    d = tmp_path / f"carried{carried_a}"
    _write(d / "baseline.json", BASE, val=0.50, test=0.40)
    _write(d / "a" / "2.json", {**BASE, "a": 2}, val=0.45, test=0.45)
    _write(d / "b" / "20.json", {**BASE, "a": carried_a, "b": 20}, val=0.3, test=0.3)
    return d


def _patch(monkeypatch):
    monkeypatch.setattr(bf, "TUNE_ORDER", {"m": ORDER})
    monkeypatch.setattr(bf, "BASELINE_CONFIG", {"m": BASE})


def test_cell_tuned_after_the_fix_is_kept(tmp_path, monkeypatch):
    _patch(monkeypatch)
    d = _cell(tmp_path, carried_a=1)          # carried the validation pick
    status, where, _config = bf.reselect("m", d, {})
    assert status != "diverge", (status, where)


def test_cell_tuned_on_test_is_cleaned_up(tmp_path, monkeypatch):
    _patch(monkeypatch)
    d = _cell(tmp_path, carried_a=2)          # carried the test pick
    status, where, _config = bf.reselect("m", d, {})
    assert (status, where) == ("diverge", "a")


def test_disagreement_at_the_last_knob_only_repicks(tmp_path, monkeypatch):
    _patch(monkeypatch)
    d = tmp_path / "last"
    _write(d / "baseline.json", BASE, val=0.50, test=0.40)
    _write(d / "a" / "2.json", {**BASE, "a": 2}, val=0.10, test=0.10)
    # b: validation prefers the default (10), test prefers 20; nothing after b.
    _write(d / "b" / "20.json", {**BASE, "b": 20}, val=0.45, test=0.45)
    status, where, _config = bf.reselect("m", d, {})
    assert (status, where) == ("repick", "b")
