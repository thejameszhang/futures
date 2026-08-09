from datetime import date, timedelta

import polars as pl

from globalmacro.build import compute_monthly_returns
from globalmacro.validation.synthetic import synthetic_correlations

# Fixtures must clear TWO thresholds or synthetic_correlations silently drops the symbol
# and the frame comes back EMPTY:
#   * >= 24 joined months (_MIN_MONTHS), and
#   * >= 15 observations in a month, else compute_monthly_returns nulls that month.
# 1,200 consecutive days is ~39 months of ~30 days each, which clears both with room.
_N = 1200


def _days(n=_N):
    return [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]


def _returns(n=_N, scale=1.0):
    # Must VARY. A constant series has zero variance, so pl.corr returns null, the row is
    # dropped, and the frame is empty -- the same failure as too-few-months, different cause.
    return [scale * ((i % 7) - 3) / 1000.0 for i in range(n)]


def test_used_is_false_when_synthetic_starts_after_the_cutoff():
    # The 6E case: the synthetic begins AFTER the future does, so it backfills nothing.
    d, r = _days(), _returns()
    ship = pl.DataFrame({"date": d, "X": r})
    pre = ship.clone()                                  # real future exists from day 0
    synth = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    out = synthetic_correlations(synth, pre, ship)
    row = out.filter(pl.col("instrument") == "X").row(0, named=True)
    assert row["n_backfilled"] == 0
    assert row["used"] is False


def test_used_is_true_when_synthetic_precedes_the_cutoff():
    # Real future starts on day 400; the synthetic spans everything -> 400 backfilled.
    d, r = _days(), _returns()
    pre = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    ship = pl.DataFrame({"date": d, "X": r})
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})
    out = synthetic_correlations(synth, pre, ship)
    row = out.filter(pl.col("instrument") == "X").row(0, named=True)
    assert row["n_backfilled"] == 400
    assert row["used"] is True


def test_used_column_is_boolean_dtype():
    # run.py does correlations.filter(pl.col("used")); a Utf8 "true"/"false" would raise.
    d, r = _days(), _returns()
    ship = pl.DataFrame({"date": d, "X": r})
    pre = ship.clone()
    synth = pl.DataFrame({"date": d, "X": _returns(scale=1.1)})
    out = synthetic_correlations(synth, pre, ship)
    assert out.height == 1, "fixture too short: the symbol was dropped"
    assert out.schema["used"] == pl.Boolean


def test_a_month_with_no_synthetic_data_is_null_not_zero():
    # THE cardinal rule. If this regresses, correlations silently collapse: a hand-rolled
    # (x+1).product()-1 would return 0.0 here, and years of fake zeros would follow.
    d, r = _days(), _returns()
    synth = pl.DataFrame({"date": d, "X": [None] * 400 + r[400:]})
    m = compute_monthly_returns(synth.select("date", pl.col("X"))).sort("date")
    first = m.get_column("X").to_list()[0]
    assert first is None, "a month with no observations must be null, never 0.0"


def test_all_null_synthetic_month_inside_window_is_excluded_not_compounded_to_zero():
    # Same cardinal rule as above, but through synthetic_correlations end-to-end, so a
    # regression in this module's own _monthly_corr (e.g. swapped for a hand-rolled
    # (x+1).product()-1) is actually caught. The test above only pins build.py.
    #
    # x (synthetic) and y (shipped) are identical every day except one calendar month,
    # where x has zero observations and y carries a large outlier return. If the null
    # month were correctly excluded, x == y everywhere that is compared and correlation
    # is exactly 1.0. If the null month were instead compounded to a fake 0.0 for x, that
    # fake 0.0 gets paired against y's real outlier return and drags the correlation down.
    d, r = _days(), _returns()

    ship_returns = list(r)
    null_month_idx = [i for i, day in enumerate(d) if day.year == 2021 and day.month == 1]
    for i in null_month_idx:
        ship_returns[i] = 0.0
    ship_returns[null_month_idx[0]] = 0.5  # outlier: real future has a large return this month

    synth_returns: list[float | None] = list(r)
    for i in null_month_idx:
        synth_returns[i] = None  # synthetic has NO observation at all this month

    ship = pl.DataFrame({"date": d, "X": ship_returns})
    pre = ship.clone()  # real future exists for the whole span -> cutoff is day 0
    synth = pl.DataFrame({"date": d, "X": synth_returns})

    out = synthetic_correlations(synth, pre, ship)
    row = out.filter(pl.col("instrument") == "X").row(0, named=True)
    assert row["correlation"] > 0.99, (
        "the all-null month must be excluded from the correlation, not compounded to a "
        "fake 0.0 and paired against the real return"
    )
