"""A null price is NOT an observation (Problem B).

Datastream emits rows with no price at all (pi_ = null; 103 of them). Since a return is
pi_(t)/pi_(t-1) - 1, ONE null price destroys TWO returns: its own day (null numerator)
AND the next day (null denominator) -- so the real move across the gap is never
computed at all. equities.main() must drop null-price rows before differencing, so the
next return spans back to the last REAL price instead of being lost.
"""
from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

import globalmacro.pipeline.equities as equities_mod


def _fake_equity(symbol, exchange="NYSE"):
    return SimpleNamespace(symbol=symbol, exchange_pmc_name=exchange)


def test_main_drops_null_price_rows_before_differencing(monkeypatch, tmp_path):
    # Mon 1/6, Tue 1/7, Wed 1/8 (null price), Thu 1/9, Fri 1/10 -- an ordinary NYSE
    # trading week, no holidays. Wed has NO price at all.
    dates = [date(2020, 1, 6), date(2020, 1, 7), date(2020, 1, 8), date(2020, 1, 9), date(2020, 1, 10)]
    prices = [100.0, 105.0, None, 110.0, 112.0]

    # IND/SET50 are unconditionally dropped by main() after the pivot; they must be
    # present (even trivially) or that .drop(...) raises.
    frame = pl.DataFrame({
        "symbol": ["TEST"] * 5 + ["IND", "SET50"],
        "date": dates + [dates[0], dates[0]],
        "pi_": prices + [1.0, 1.0],
    })

    # DS2INDEXDATA/equities are normally assigned only inside the `if __name__ ==
    # "__main__":` guard, so they do not exist as module attributes at import time;
    # raising=False lets monkeypatch create them for the duration of this test.
    monkeypatch.setattr(equities_mod, "DS2INDEXDATA", frame, raising=False)
    monkeypatch.setattr(equities_mod, "equities", [
        _fake_equity("TEST"), _fake_equity("IND"), _fake_equity("SET50"),
    ], raising=False)
    monkeypatch.setattr(equities_mod, "EQUITIES_PATH", tmp_path)

    equities_mod.main()

    out = pl.read_csv(tmp_path / "spot_equity_returns.csv", try_parse_dates=True)

    # The null-price row must be dropped entirely -- not kept around as a null/zero return.
    assert date(2020, 1, 8) not in out["date"].to_list(), (
        "Wednesday's null-price row survived into the output; it must be dropped before "
        "differencing, not merely produce a null return."
    )

    # Monday is still the first observation -> still null (unchanged behavior).
    mon = out.filter(pl.col("date") == date(2020, 1, 6))["TEST"].item()
    assert mon is None

    # Tuesday is an ordinary consecutive-day return, unaffected by the fix.
    tue = out.filter(pl.col("date") == date(2020, 1, 7))["TEST"].item()
    assert tue == pytest.approx(105.0 / 100.0 - 1)

    # Thursday's return must span back across the gap to Tuesday's REAL price (105),
    # not be lost to a null numerator/denominator, and must NEVER be fabricated as 0.0.
    thu = out.filter(pl.col("date") == date(2020, 1, 9))["TEST"].item()
    assert thu is not None, "the real move across the null-price gap was never computed"
    assert thu != 0.0, "a null return must NEVER become 0.0"
    assert thu == pytest.approx(110.0 / 105.0 - 1)

    # Friday is an ordinary consecutive-day return following the recovered Thursday row.
    fri = out.filter(pl.col("date") == date(2020, 1, 10))["TEST"].item()
    assert fri == pytest.approx(112.0 / 110.0 - 1)
