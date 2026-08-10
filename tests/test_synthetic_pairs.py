"""Tests for the shared window helper (`_windowed`) and `synthetic_pairs`, which
comparison.pdf renders for the two synthetic exercises (synthetic_fx, synthetic_equity).

THE controlling requirement (see synthetic.py's module docstring): the plot must use the
EXACT SAME real-future window as the grade. `_windowed` is the single source of truth for
that window; `synthetic_correlations` and `synthetic_pairs` both call it, so a test that
pins `synthetic_correlations`'s output also pins the window `synthetic_pairs` shares.
"""
from datetime import date, timedelta

import polars as pl
import pytest

from globalmacro.build import first_valid_date
from globalmacro.validation.synthetic import (
    _windowed,
    synthetic_correlations,
    synthetic_pairs,
)

_N = 1200


def _days(n=_N):
    return [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def _returns(n=_N, scale=1.0):
    return [scale * ((i % 7) - 3) / 1000.0 for i in range(n)]


# ---------------------------------------------------------------------------
# _windowed
# ---------------------------------------------------------------------------


def test_windowed_cutoff_matches_first_valid_date_of_pre_and_window_starts_at_or_after_it():
    d, r = _days(), _returns()
    pre = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})   # real future starts day 400
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})   # synth spans the whole window

    out = _windowed(synth, pre, ship, "X")
    assert out is not None
    cutoff, window = out
    assert cutoff == first_valid_date(pre, "X")
    window_start = window.get_column("date").min()
    assert isinstance(window_start, date)
    assert window_start >= cutoff
    assert set(window.columns) == {"date", "x", "y"}


def test_windowed_none_when_symbol_missing_from_ship_or_pre():
    d, r = _days(), _returns()
    synth = pl.DataFrame({"date": d, "X": r})
    ship_missing_x = pl.DataFrame({"date": d, "Y": r})
    pre = pl.DataFrame({"date": d, "X": r})
    assert _windowed(synth, pre, ship_missing_x, "X") is None

    ship = pl.DataFrame({"date": d, "X": r})
    pre_missing_x = pl.DataFrame({"date": d, "Y": r})
    assert _windowed(synth, pre_missing_x, ship, "X") is None


def test_windowed_none_when_the_real_future_never_has_a_valid_observation():
    d, r = _days(), _returns()
    synth = pl.DataFrame({"date": d, "X": r})
    ship = pl.DataFrame({"date": d, "X": r})
    pre = pl.DataFrame({"date": d, "X": [None] * len(d)})   # cutoff is None
    assert _windowed(synth, pre, ship, "X") is None


# ---------------------------------------------------------------------------
# synthetic_correlations regression: output must be UNCHANGED by the _windowed refactor
# ---------------------------------------------------------------------------


def test_synthetic_correlations_output_unchanged_by_windowed_refactor():
    # Pinned against the pre-refactor implementation (inline cutoff+join+filter). If the
    # _windowed extraction ever derives the window differently, these numbers move.
    d, r = _days(), _returns()
    pre = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})

    out = synthetic_correlations(synth, pre, ship)
    row = out.filter(pl.col("instrument") == "X").row(0, named=True)

    assert row["correlation"] == pytest.approx(0.9999999819399173)
    assert row["n_obs"] == 26
    assert row["corr_daily"] == pytest.approx(0.9999999999999999)
    assert row["corr_daily_alt"] is None
    assert row["mean_gap_bp"] == pytest.approx(-0.003750000000000053)
    assert row["n_backfilled"] == 400
    assert row["used"] is True


# ---------------------------------------------------------------------------
# synthetic_pairs: windowing (load-bearing), schema, empty input
# ---------------------------------------------------------------------------


def test_synthetic_pairs_excludes_the_pre_cutoff_backfill_region():
    d, r = _days(), _returns()
    cutoff_idx = 400
    pre = pl.DataFrame({"date": d, "X": [None] * cutoff_idx + r[cutoff_idx:]})
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})   # exists BEFORE the cutoff too

    cutoff = first_valid_date(pre, "X")
    pairs = synthetic_pairs(synth, pre, ship)

    assert pairs.height > 0
    first_month = pairs.get_column("month").min()
    assert first_month >= cutoff.replace(day=1)


def test_synthetic_pairs_schema_and_dtypes():
    d, r = _days(), _returns()
    pre = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})

    out = synthetic_pairs(synth, pre, ship, name_of={"X": "Ecks"})
    assert set(out.columns) == {"instrument", "name", "month", "ours", "theirs"}
    assert out.schema["instrument"] == pl.Utf8
    assert out.schema["name"] == pl.Utf8
    assert out.schema["month"] == pl.Date
    assert out.schema["ours"] == pl.Float64
    assert out.schema["theirs"] == pl.Float64
    assert out.height > 0
    assert out.filter(pl.col("instrument") == "X").get_column("name").to_list()[0] == "Ecks"


def test_synthetic_pairs_defaults_name_to_the_symbol_when_name_of_is_absent():
    d, r = _days(), _returns()
    pre = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})

    out = synthetic_pairs(synth, pre, ship)
    assert out.get_column("name").to_list()[0] == "X"


def test_synthetic_pairs_empty_input_returns_empty_frame_with_columns():
    empty = pl.DataFrame(schema={"date": pl.Date})
    out = synthetic_pairs(empty, empty, empty)
    assert out.height == 0
    assert set(out.columns) == {"instrument", "name", "month", "ours", "theirs"}


# ---------------------------------------------------------------------------
# Wiring: both Checks now advertise pairs() + series_labels
# ---------------------------------------------------------------------------


def test_synthetic_fx_and_equity_checks_have_pairs_wired():
    from globalmacro.validation.synthetic_equity import synthetic_equity_check
    from globalmacro.validation.synthetic_fx import synthetic_fx_check

    assert synthetic_fx_check.pairs is not None
    assert synthetic_fx_check.series_labels == ("CIP synthetic", "real future")

    assert synthetic_equity_check.pairs is not None
    assert synthetic_equity_check.series_labels == ("spot synthetic", "real future")
