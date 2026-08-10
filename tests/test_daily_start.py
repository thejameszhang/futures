"""A series must start when its index began being published DAILY.

An index that trades a handful of times a month is not a daily price series, and its early
history must not be published as one. Datastream carries such series on a continuous grid
regardless, so the leading edge has to be cut.

This also fixes a second defect. Z (FTSE 100) is MONTH-END ONLY until 1984, and
load_synthetic_returns' coalesce(Z, FTALLSH) PREFERS Z -- so a month-end row drops a
MONTH-LONG return into a daily cell, and the month then compounds FTALLSH's ~21 daily returns
AND Z's monthly one. December 1983 ships +21.50% against a true +1.61%. Z's month-end era is
not "daily" by this rule, so it is cut and the coalesce falls through to FTALLSH -- which is
what the splice intends.

The DENOMINATOR is each index's OWN exchange calendar. A panel-wide calendar makes any
exchange with a holiday cluster the rest of the world does not share -- Chinese New Year --
look sub-daily, and wrongly truncates it.
"""
import math
from datetime import date, timedelta

import pandas as pd
import pandas_market_calendars as pmc
import polars as pl
import pytest

from globalmacro.pipeline.equities import (
    _build_equity_indices,
    first_daily_date,
    get_schedule_dates,
)
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import EQUITIES_PATH, PROJECT_ROOT

# Derived by the rule, then pinned. A change here means the DATA changed -- investigate
# before updating the pin.
EXPECTED_DAILY_START = {
    "FTALLSH": date(1969, 1, 1),    # 26 obs in 1968, 253 in 1969. Feeds Z.
    "HSI":     date(1969, 12, 1),   # 14 obs/yr before
    "AUSTOLD": date(1980, 1, 1),    # 12 obs/yr before. Feeds AP.
    "Z":       date(1984, 1, 1),    # MONTH-END ONLY until 1984 -- see the double-count note
    "SGP":     date(1987, 1, 1),    # 12 obs/yr before
    "SWSEALI": date(1987, 1, 1),    # 12 obs/yr before. Feeds FXS30.
    "FW20":    date(1994, 5, 1),    # Warsaw traded a few sessions/week until 1994 (tier 2)
    # CN is dense from day one -- this pin is NOT about truncation. It exists because the
    # per-index calendar denominator is load-bearing: an earlier draft judged every index
    # against a panel-wide union calendar, and Chinese New Year (a Shanghai holiday cluster
    # the rest of the world does not share) made CN look sub-daily for a month at a time. That
    # moved CN's cutoff from 1999-01 to 2000-03 and destroyed 270 rows of genuinely daily data.
    # Nothing else in this file would catch a union calendar sneaking back in -- the two-sided
    # bound below (expected <= first < expected + 1 month) is what stops it.
    "CN":      date(1999, 1, 1),
}


def _sessions(calendar, start, end):
    # schedule() is annotated -> pd.DataFrame, so its .index is typed as the generic
    # pandas Index, not DatetimeIndex -- it is one at runtime (see schedule_from_days,
    # which builds it from a pd.DatetimeIndex of valid trading days). pandas' own
    # stubs don't expose DatetimeIndex.date either (it's mixin-delegated), so go
    # through Series.dt, which is fully typed.
    idx = pmc.get_calendar(calendar).schedule(str(start), str(end)).index
    return set(pd.Series(idx).dt.date)


def _frame(dates):
    return pl.DataFrame({"date": list(dates), "pi_ret": [0.001] * len(list(dates))}).sort("date")


def test_a_weekly_series_is_never_daily():
    # One observation a week against a 5-day exchange calendar -> ~20% density.
    weekly = [date(2000, 1, 3) + timedelta(days=7 * i) for i in range(60)]
    cal = _sessions("NYSE", date(2000, 1, 1), date(2001, 6, 30))
    assert first_daily_date(_frame(weekly), cal) is None


def test_a_dense_series_keeps_its_first_day():
    cal = sorted(_sessions("NYSE", date(2000, 1, 3), date(2003, 12, 31)))
    assert first_daily_date(_frame(cal), set(cal)) == cal[0]


def test_a_partial_first_month_is_not_cut():
    # Starting on the 20th must not cost the series its first fortnight.
    cal = sorted(_sessions("NYSE", date(2000, 1, 20), date(2003, 12, 31)))
    assert first_daily_date(_frame(cal), set(cal)) == cal[0]


def test_a_short_tail_cannot_establish_daily():
    # The window must be FULL. A sparse series with two dense months at the very END must NOT
    # be declared daily -- all() of a truncated slice would wrongly say yes.
    monthly = [date(2010 + (m // 12), (m % 12) + 1, 15) for m in range(24)]
    dense = sorted(_sessions("NYSE", date(2012, 11, 1), date(2012, 12, 31)))
    cal = _sessions("NYSE", date(2010, 1, 1), date(2012, 12, 31))
    assert first_daily_date(_frame(monthly + dense), cal) is None


# -- Bracket tests: pin DAILY_DENSITY and DAILY_WINDOW by OBSERVABLE EFFECT, not by asserting
# the constants' values. Asserting `DAILY_DENSITY == 0.60` is a tautology -- a maintainer who
# relaxes the rule just updates the assert alongside it. These instead build synthetic series
# that must land on opposite sides of the policy and check first_daily_date's output. A
# mutation to either constant (DAILY_DENSITY 0.60 -> 0.30, or DAILY_WINDOW 12 -> 3) must make
# at least one of these fail.

def _by_month(exchange, start, end):
    cal = sorted(_sessions(exchange, start, end))
    by_month = {}
    for d in cal:
        by_month.setdefault((d.year, d.month), []).append(d)
    return cal, by_month


def test_a_50_percent_dense_leading_year_is_cut():
    # First 12 months at 50% of their exchange's sessions (below the 0.60 floor), followed by
    # 12 fully-dense months. The sparse year must be cut: the series starts at the dense year.
    cal, by_month = _by_month("NYSE", date(2000, 1, 1), date(2001, 12, 31))
    month_keys = sorted(by_month.keys())
    sparse_year, dense_year = month_keys[:12], month_keys[12:24]
    observed = []
    for key in sparse_year:
        sessions = by_month[key]
        observed.extend(sessions[: math.floor(0.50 * len(sessions))])
    for key in dense_year:
        observed.extend(by_month[key])
    result = first_daily_date(_frame(observed), set(cal))
    expected_cutoff = date(dense_year[0][0], dense_year[0][1], 1)
    assert result == expected_cutoff, "a 50%-dense leading year must be cut, not published as daily"


def test_a_70_percent_dense_leading_year_is_not_cut():
    # Same shape, but the leading year is 70% dense (above the 0.60 floor) -- it must NOT be
    # cut: the series keeps its very first observed day.
    cal, by_month = _by_month("NYSE", date(2000, 1, 1), date(2001, 12, 31))
    month_keys = sorted(by_month.keys())
    dense_leading_year, dense_year = month_keys[:12], month_keys[12:24]
    observed = []
    for key in dense_leading_year:
        sessions = by_month[key]
        observed.extend(sessions[: math.floor(0.70 * len(sessions))])
    for key in dense_year:
        observed.extend(by_month[key])
    result = first_daily_date(_frame(observed), set(cal))
    assert result == min(observed), "a 70%-dense leading year must NOT be cut"


def test_eleven_dense_trailing_months_cannot_establish_daily():
    # 24 sparse leading months (1 obs each -- nowhere near dense) then exactly 11 fully-dense
    # trailing months. 11 < DAILY_WINDOW, so no full window is ever dense: None.
    cal, by_month = _by_month("NYSE", date(2010, 1, 1), date(2013, 12, 31))
    month_keys = sorted(by_month.keys())
    leading, trailing = month_keys[:24], month_keys[24:35]
    assert len(trailing) == 11
    observed = [by_month[k][0] for k in leading]
    for k in trailing:
        observed.extend(by_month[k])
    assert first_daily_date(_frame(observed), set(cal)) is None


def test_twelve_dense_trailing_months_establishes_daily():
    # Identical shape, but exactly 12 fully-dense trailing months -- a full DAILY_WINDOW -- so
    # a cutoff must be returned.
    cal, by_month = _by_month("NYSE", date(2010, 1, 1), date(2013, 12, 31))
    month_keys = sorted(by_month.keys())
    leading, trailing = month_keys[:24], month_keys[24:36]
    assert len(trailing) == 12
    observed = [by_month[k][0] for k in leading]
    for k in trailing:
        observed.extend(by_month[k])
    assert first_daily_date(_frame(observed), set(cal)) is not None


def _spot():
    df = pl.read_csv(str(EQUITIES_PATH / "spot_equity_returns.csv"), infer_schema_length=0)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c != "date"]
    )


def _exchange_by_symbol():
    # The same construction main() uses to pair each shipped column with ITS OWN exchange.
    equities = _build_equity_indices(load_config(PROJECT_ROOT / "tier1.yaml"))
    equities.extend(_build_equity_indices(load_config(PROJECT_ROOT / "tier2.yaml")))
    return {e.symbol: e.exchange_pmc_name for e in equities}


def _add_months(d, months):
    total = (d.month - 1) + months
    return date(d.year + total // 12, total % 12 + 1, 1)


@pytest.mark.parametrize("symbol,expected", sorted(EXPECTED_DAILY_START.items()))
def test_the_shipped_series_starts_where_expected(symbol, expected):
    # Passes only after spot_equity_returns.csv is regenerated.
    spot = _spot()
    if symbol not in spot.columns:
        pytest.skip(f"{symbol} not in the shipped frame")
    first = spot.select(pl.col("date").filter(pl.col(symbol).is_not_null()).min()).item()
    # Two-sided: >= expected catches UNDER-truncation (a sub-daily leading edge published as
    # daily). < expected + 1 month catches OVER-truncation -- the failure mode the design most
    # fears. An earlier draft used a panel-wide union calendar instead of each index's own,
    # and it destroyed 14 months of genuinely daily CN data (Chinese New Year makes Shanghai
    # look sparse against a world calendar): CN's cutoff moved 1999-01 -> 2000-03, deleting 270
    # real rows. A one-sided ">=" assert would never have caught that.
    assert expected <= first < _add_months(expected, 1), (
        f"{symbol} starts {first}, expected exactly {expected} (its first genuinely daily "
        f"month) -- either a sub-daily leading edge is being published as daily, or the "
        f"truncation cut into real daily data."
    )


def test_no_unpinned_series_has_a_sub_daily_leading_edge():
    # Anything NOT pinned must be dense from its first year. A new entry means a sub-daily
    # edge appeared and must be understood, not silently cut.
    #
    # CN is now pinned (see EXPECTED_DAILY_START), so it is skipped here regardless of its
    # density -- that is fine. This scan re-derives density from CN's OWN calendar directly
    # from the shipped CSV, so a union-calendar bug inside first_daily_date would never show up
    # as a surprise here anyway (the shipped CN dates it measures are already post-truncation).
    # The two-sided pin in test_the_shipped_series_starts_where_expected is what actually
    # catches that regression.
    #
    # The denominator is SESSIONS ACTUALLY AVAILABLE over the series' own first 12-month
    # window on ITS OWN exchange calendar -- not an extrapolation from a flat "month" count.
    # A series starting mid-month (FXC25: 2016-12-20) has a genuinely short first month; an
    # extrapolated rate (observed * 12 / months_available) treats that short month as if it
    # were full-length and wrongly reports the series as sub-daily even when it is 100% dense
    # against the sessions its exchange actually held. See test_a_partial_first_month_is_not_cut,
    # which guards the real rule against the identical trap.
    spot = _spot()
    exchange_by_symbol = _exchange_by_symbol()
    surprises = {}
    for symbol in (c for c in spot.columns if c != "date"):
        s = spot.select("date", pl.col(symbol).alias("v")).drop_nulls().sort("date")
        if s.height < 250:
            continue
        first = s.get_column("date").min()
        # .min() is typed over every polars dtype's Python literal; the "date" column is
        # pl.Date (see _spot's strptime), so this is a real date at runtime -- assert it
        # rather than assume, so a dtype regression fails loudly here instead of on the
        # .year/.month access below.
        assert isinstance(first, date), f"{symbol}: expected a date, got {type(first)!r}"
        window_start = first
        window_end = _add_months(date(first.year, first.month, 1), 12)  # exclusive
        in_window = s.filter(
            (pl.col("date") >= window_start) & (pl.col("date") < window_end)
        ).height
        exchange = exchange_by_symbol[symbol]
        schedule = set(get_schedule_dates(exchange).explode().to_list())
        sessions_available = sum(1 for d in schedule if window_start <= d < window_end)
        rate = in_window / sessions_available if sessions_available else 0.0
        # Hardcoded, NOT imported from production. The whole point of this scan is to catch a
        # relaxed DAILY_DENSITY -- importing the constant would make the scan's own threshold
        # slide with the rule it exists to police, so it could never detect a relaxation.
        if rate < 0.60 and symbol not in EXPECTED_DAILY_START:
            surprises[symbol] = (first, in_window, sessions_available)
    assert not surprises, f"sub-daily leading edge on unpinned series: {surprises}"
