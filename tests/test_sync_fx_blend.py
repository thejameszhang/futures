from datetime import date, timedelta

import polars as pl
import pytest


def _weekday_grid(start: date, n: int) -> list[date]:
    """n consecutive dates from `start` (calendar-dense, incl. weekends)."""
    return [start + timedelta(days=i) for i in range(n)]


def _toy_fx_sync(dates, **cols) -> pl.DataFrame:
    return pl.DataFrame({"date": dates, **cols})


G10 = {"6A": "AUD", "6C": "CAD", "6J": "JPY", "6B": "GBP", "6E": "EUR",
       "6N": "NZD", "6S": "CHF", "NOK": "NOK", "SEK": "SEK"}


def test_g10_futures_set_matches_validation_g10_majors():
    from globalmacro.utils.sync_fx import G10_FUTURES
    from globalmacro.validation.fx_futures import G10_MAJORS
    assert set(G10_FUTURES) == set(G10_MAJORS)


def test_covered_day_uses_the_futures_return():
    # Mon-Fri 2020-01-06..10; cutoff = first finite future = 01-06. Compustat drifts +0.001/day;
    # the future says +0.02 on Wed. On Wed (both endpoints observed) the blend return must be
    # the FUTURE's +0.02, so the Tue->Wed level ratio == 1.02, not the Compustat 1.001.
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 5)  # Mon..Fri
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002, 1.003, 1.004])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.005, 0.02, 0.005, 0.005]})
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR"})
    lvl = out.get_column("EUR").to_list()
    assert lvl[2] / lvl[1] == pytest.approx(1.02, abs=1e-9)   # Wed used the future


def test_futures_holiday_falls_to_compustat_no_double_count():
    # 5 weekdays; future is null (holiday) on Wed. The Tue->Thu compounded blend move must
    # equal the Compustat Tue->Thu move exactly (the future's gap-spanning return is NOT
    # applied on top of the Compustat holiday move).
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 5)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.010, 1.020, 1.033, 1.040])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.01, None, 0.02, 0.006]})
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR"})
    lvl = out.get_column("EUR").to_list()
    comp_tue_thu = 1.033 / 1.010
    assert lvl[3] / lvl[1] == pytest.approx(comp_tue_thu, abs=1e-9)


def test_verbatim_compustat_below_cutoff_and_continuous_at_splice():
    # Future starts (first finite) on Wed. Mon/Tue must be Compustat verbatim; Wed level
    # equals Compustat Wed (cutoff day routes to Compustat); Thu uses the future.
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 5)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002, 1.003, 1.004])
    pre = pl.DataFrame({"date": dates, "6E": [None, None, 0.5, 0.03, 0.03]})
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR"})
    lvl = out.get_column("EUR").to_list()
    assert lvl[0] == pytest.approx(1.0)      # Mon verbatim
    assert lvl[1] == pytest.approx(1.001)    # Tue verbatim
    assert lvl[2] == pytest.approx(1.002)    # Wed == Compustat (cutoff day)
    assert lvl[3] == pytest.approx(1.002 * 1.03)   # Thu uses the future


def test_weekend_is_a_single_step_and_reindexed_back():
    # Fri 2020-01-10 .. Mon 2020-01-13 with a Sat/Sun gap. Output must carry the weekend
    # rows (ffilled from Friday) AND the Fri->Mon future move must be a single step.
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 10), 4)  # Fri, Sat, Sun, Mon
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.0, 1.0, 1.05])   # Compustat ffills the weekend
    pre = pl.DataFrame({"date": dates, "6E": [0.0, None, None, 0.02]})  # future only on weekdays
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR"})
    assert out.height == 4                                 # weekend rows preserved
    lvl = out.get_column("EUR").to_list()
    assert lvl[1] == pytest.approx(lvl[0])                 # Sat ffilled from Fri
    assert lvl[3] / lvl[0] == pytest.approx(1.02, abs=1e-9)  # Fri->Mon single future step


def test_non_g10_currency_passes_through_unchanged():
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 3)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002], KRW=[1000.0, 1001.0, 1002.0])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.01, 0.01], "KRW": [0.0, 0.5, 0.5]})
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR", "KRW": "KRW"})
    assert out.get_column("KRW").to_list() == [1000.0, 1001.0, 1002.0]   # EM untouched


def test_c1_guard_raises_on_all_usd_map():
    # build_currency_map maps every future to 'USD'; that must override zero currencies and RAISE.
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 3)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.01, 0.01]})
    with pytest.raises(ValueError, match="zero currencies"):
        build_sync_fx_panel(fx, pre, {"6E": "USD"})


def test_symbol_absent_from_pre_splice_is_pure_compustat():
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 3)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002], JPY=[100.0, 100.1, 100.2])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.01, 0.01]})  # no 6J column
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR", "6J": "JPY"})
    assert out.get_column("JPY").to_list() == [100.0, 100.1, 100.2]  # untouched (no future)


def test_below_cutoff_weekend_quote_is_verbatim_not_friday_ffill():
    # Below the cutoff the panel must be pure Compustat VERBATIM, including a weekend row
    # whose real quote differs from the preceding Friday (Compustat sometimes carries a
    # distinct Sat/Sun print, e.g. GBP-cross 1994-02-19/20). The blend must not clobber it
    # with a Friday-forward-fill: Sat's output must equal Sat's raw fx_sync level exactly.
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 3), 7)  # Fri, Sat, Sun, Mon, Tue, Wed, Thu
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.005, 1.006, 1.01, 1.011, 1.012, 1.013])
    pre = pl.DataFrame({"date": dates, "6E": [None, None, None, None, None, None, 0.01]})  # cutoff = Thu
    out = build_sync_fx_panel(fx, pre, {"6E": "EUR"})
    lvl = out.get_column("EUR").to_list()
    assert lvl[1] == pytest.approx(1.005)   # Sat verbatim, NOT ffilled from Friday's 1.0
    assert lvl[2] == pytest.approx(1.006)   # Sun verbatim too


def test_integration_g10_differs_em_equals_compustat():
    # End-to-end shape of the hook: a covered G10 currency's blended level differs from
    # Compustat; an EM currency's is identical. (Same inputs the hook feeds build_sync_fx_panel.)
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    dates = _weekday_grid(date(2020, 1, 6), 4)
    fx = _toy_fx_sync(dates, EUR=[1.0, 1.001, 1.002, 1.003], KRW=[1000.0, 1000.1, 1000.2, 1000.3])
    pre = pl.DataFrame({"date": dates, "6E": [0.0, 0.05, 0.05, 0.05], "KRW": [0.0, 0.9, 0.9, 0.9]})
    m = {"6E": "EUR", "KRW": "KRW"}
    out = build_sync_fx_panel(fx, pre, m)
    assert out.get_column("EUR").to_list() != fx.get_column("EUR").to_list()   # blended
    assert out.get_column("KRW").to_list() == fx.get_column("KRW").to_list()   # EM untouched
    assert set(out.columns) == set(fx.columns)                                 # same column set


def _has_raw_sync_data() -> bool:
    """True iff the raw CSVs pre_splice_panel('sync') + save_compustat_fx_rates() need are
    on disk. The data-coupled tests skip (not error) where they are absent."""
    from globalmacro.utils.paths import COMPUSTAT_PATH, DATASETS_ROOT
    return (
        (COMPUSTAT_PATH / "exrt_dly.csv").exists()
        and (DATASETS_ROOT / "tier1" / "sync" / "currency_daily_returns.csv").exists()
    )


def test_never_synthetic_below_cutoff_on_real_panels():
    # REAL-DATA regression: for a late-listing G10 currency
    # (NOK, real future 2002), (1) the cutoff from the synthetic-blind pre-splice panel is the
    # REAL 2002 start, NOT the 1996 CIP-synthetic floor; and (2) the blended sync level BELOW
    # that cutoff equals RAW Compustat spot verbatim -- never the synthetic backfill. Uses the
    # SAME inputs build.main feeds the blend (build_sync_fx_panel + pre_splice_panel('sync')).
    if not _has_raw_sync_data():
        pytest.skip("raw sync CSVs absent")
    from globalmacro.pipeline.fx import (
        SYMBOL_TO_CURCDD_MAPPING,
        save_compustat_fx_rates,
    )
    from globalmacro.utils.panels import first_finite_date
    from globalmacro.utils.sync_fx import build_sync_fx_panel
    from globalmacro.validation.synthetic import pre_splice_panel
    fx_sync = save_compustat_fx_rates()
    pre = pre_splice_panel("sync")
    cutoff = first_finite_date(pre, "NOK")
    assert cutoff > date(1996, 6, 1)   # real 2002 start, NOT the 1996 synthetic floor
    out = build_sync_fx_panel(fx_sync, pre, SYMBOL_TO_CURCDD_MAPPING)
    below = (
        out.filter(pl.col("date") < cutoff)
        .select("date", pl.col("NOK").alias("blend"))
        .join(fx_sync.select("date", pl.col("NOK").cast(pl.Float64).alias("raw")), on="date", how="inner")
        .drop_nulls()
    )
    assert below.height > 0
    max_abs_diff = below.select((pl.col("blend") - pl.col("raw")).abs().max()).item()
    assert max_abs_diff < 1e-9   # blended NOK below its cutoff == raw Compustat, never synthetic
