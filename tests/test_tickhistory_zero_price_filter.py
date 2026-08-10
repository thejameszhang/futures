"""A literal `0.0` Datastream price must never reach the fallback price series.

`open_N`/`settlement_N` == 0.0 is Datastream's "no trade yet" marker, not a real price --
same family as the FKLI zero-settlement-denominator bug fixed in `5468d2c`. Left unfiltered
it can reach `lasttrdprice_cN` via `coalesce(open_N, ...)` (`:490`/`:507` after this fix) and
from there `compute_settlement_price` may accept it outright whenever a stale bid/ask spread
happens to straddle zero. See `.superpowers/sdd/zero-price-fallback-report.md` for the traced
before/after on the two real occurrences the census found (GC clscode 1508 2020-12-18, ES
clscode 1035 2016-11-23) and why neither ever reached a shipped panel cell.

`load_open_prices`/`load_settlement_prices` null the zero PER COLUMN, not per row: open_1..4
(resp. settlement_1..4) are four independent contract-order prices sharing one row per
(clscode, date), so a row-level filter (like the one `load_quotes_data` uses at `:94`, where
bid/ask genuinely are one paired quote) would also discard the other three orders' valid
prices on a date where only one order printed a zero.
"""
from datetime import date

import polars as pl
import pytest

import globalmacro.pipeline.tickhistory as th
from globalmacro.utils.models import AssetClass
from globalmacro.utils.paths import FUTURES_PATH as REAL_FUTURES_PATH

_OPEN_SCHEMA = {"clscode": pl.Int64, "date": pl.Date, "open_1": pl.Float64, "open_2": pl.Float64, "open_3": pl.Float64, "open_4": pl.Float64}
_SETTLEMENT_SCHEMA = {"clscode": pl.Int64, "date": pl.Date, "settlement_1": pl.Float64, "settlement_2": pl.Float64, "settlement_3": pl.Float64, "settlement_4": pl.Float64}


def test_load_open_prices_nulls_zero_but_keeps_nonzero_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "FUTURES_PATH", tmp_path)
    monkeypatch.setattr(th, "ASSET_CLASS", AssetClass.COMMODITY, raising=False)  # -> CT parquet
    rows = [
        # Order 4 prints the zero marker; orders 1-3 are real prices on the SAME row/date --
        # the fix must null only order 4, not the whole row.
        {"clscode": 9001, "date": date(2020, 6, 15), "open_1": 100.0, "open_2": 101.5, "open_3": 102.25, "open_4": 0.0},
        # An ordinary all-nonzero row must pass through byte-for-byte untouched.
        {"clscode": 9001, "date": date(2020, 6, 16), "open_1": 103.0, "open_2": 103.5, "open_3": 104.0, "open_4": 104.5},
    ]
    pl.DataFrame(rows, schema=_OPEN_SCHEMA).write_parquet(tmp_path / "datastream_futures_open_CT.parquet")

    got = th.load_open_prices().sort("date")

    zero_day = got.filter(pl.col("date") == date(2020, 6, 15)).row(0, named=True)
    assert zero_day["open_1"] == 100.0
    assert zero_day["open_2"] == 101.5
    assert zero_day["open_3"] == 102.25
    assert zero_day["open_4"] is None, "a literal 0.0 open price reached the price series instead of being nulled"

    ordinary_day = got.filter(pl.col("date") == date(2020, 6, 16)).row(0, named=True)
    assert ordinary_day == {"clscode": 9001, "date": date(2020, 6, 16), "open_1": 103.0, "open_2": 103.5, "open_3": 104.0, "open_4": 104.5}


def test_load_settlement_prices_nulls_zero_but_keeps_nonzero_prices(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "FUTURES_PATH", tmp_path)
    monkeypatch.setattr(th, "ASSET_CLASS", AssetClass.COMMODITY, raising=False)  # -> CT parquet
    rows = [
        {"clscode": 9079, "date": date(1999, 3, 2), "settlement_1": 1800.0, "settlement_2": 0.0, "settlement_3": 1802.5, "settlement_4": 1803.0},
        {"clscode": 9079, "date": date(1999, 3, 3), "settlement_1": 1804.0, "settlement_2": 1804.5, "settlement_3": 1805.0, "settlement_4": 1805.5},
    ]
    pl.DataFrame(rows, schema=_SETTLEMENT_SCHEMA).write_parquet(tmp_path / "datastream_futures_settlement_CT.parquet")

    got = th.load_settlement_prices().sort("date")

    zero_day = got.filter(pl.col("date") == date(1999, 3, 2)).row(0, named=True)
    assert zero_day["settlement_1"] == 1800.0
    assert zero_day["settlement_2"] is None, "a literal 0.0 settlement price reached the price series instead of being nulled"
    assert zero_day["settlement_3"] == 1802.5
    assert zero_day["settlement_4"] == 1803.0

    ordinary_day = got.filter(pl.col("date") == date(1999, 3, 3)).row(0, named=True)
    assert ordinary_day == {"clscode": 9079, "date": date(1999, 3, 3), "settlement_1": 1804.0, "settlement_2": 1804.5, "settlement_3": 1805.0, "settlement_4": 1805.5}


def test_load_open_prices_real_gc_and_es_cells_are_null_not_zero(monkeypatch):
    """Regression pin for the two real occurrences the full-history census found (all 63
    fallback symbols swept; these are the only two): GC clscode 1508 open_4 on 2020-12-18,
    ES clscode 1035 open_4 on 2016-11-23. Both must load as null, never as 0.0."""
    parquet = REAL_FUTURES_PATH / "datastream_futures_open_CT.parquet"
    if not parquet.exists():
        pytest.skip("real datastream parquet absent")
    monkeypatch.setattr(th, "ASSET_CLASS", AssetClass.COMMODITY, raising=False)  # -> CT parquet; FUTURES_PATH left real

    got = th.load_open_prices()
    for clscode, d, label in [(1508, date(2020, 12, 18), "GC"), (1035, date(2016, 11, 23), "ES")]:
        row = got.filter((pl.col("clscode") == clscode) & (pl.col("date") == d))
        assert row.height == 1, f"{label} {d} row missing from load_open_prices"
        assert row["open_4"].item() is None, f"{label} {d} open_4 loaded as {row['open_4'].item()!r}, expected null"
