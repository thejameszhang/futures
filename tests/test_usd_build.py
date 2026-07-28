from datetime import date

import polars as pl
import pytest

from globalmacro.build import (
    build_currency_map,
    load_symbols,
    load_symbols_to_save,
    load_synthetic_returns,
    save_usd_datasets,
)


def test_load_synthetic_returns_uses_source_specific_fx():
    from globalmacro.build import load_rf
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    equities = [f for f in (t1f + t2f) if f.dsindexcode is not None]
    a, s = load_synthetic_returns(load_rf(), equities)
    # async and sync must NOT be identical clones now (different FX source)
    for col in ["NOK", "SEK", "6N", "6A"]:
        if col in a.columns and col in s.columns:
            m = a.select(["date", pl.col(col).alias("x")]).join(
                s.select(["date", pl.col(col).alias("y")]), on="date", how="inner").drop_nulls()
            # same rate, different snapshot -> highly but not perfectly correlated
            if m.height > 500:
                c = m.select(pl.corr("x", "y")).item()
                assert 0.5 < c < 0.9999, f"{col}: async/sync synthetic corr={c} (expected same-rate-diff-time)"
                return
    raise AssertionError("no comparable currency column found")


def test_currency_map_covers_every_published_symbol():
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    ccy = build_currency_map(t1f + t2f)
    published = set(load_symbols_to_save(t1f)) | set(load_symbols_to_save(t2f))
    assert sorted(published - set(ccy)) == []
    assert ccy["6E"] == "USD" and ccy["XAE"] == "USD" and ccy["FDAX"] == "EUR"


def test_save_usd_datasets_uses_async_vs_sync_fx(tmp_path):
    d = [date(2020, 1, 1), date(2020, 1, 2)]
    def panel():
        return pl.DataFrame({"date": d, "ES": [None, 0.01], "FGBL": [None, 0.01]})
    fx_async = pl.DataFrame({"date": d, "EUR": [1.10, 1.20]})   # +9.1%
    fx_sync = pl.DataFrame({"date": d, "EUR": [1.10, 1.15]})    # +4.5%
    ccy = {"ES": "USD", "FGBL": "EUR"}
    save_usd_datasets(panel(), panel(), panel(), panel(), ccy, fx_async, fx_sync, out_root=tmp_path)
    for rel in ["tier1/async/async_daily_usd.csv", "tier1/sync/sync_daily_usd.csv",
                "tier1/async/async_monthly_usd.csv", "tier2/async/async_daily_usd.csv",
                "tier2/sync/sync_daily_usd.csv", "tier2/async/async_monthly_usd.csv"]:
        assert (tmp_path / rel).is_file(), rel
    a = pl.read_csv(tmp_path / "tier1/async/async_daily_usd.csv")["FGBL"].to_list()[1]
    s = pl.read_csv(tmp_path / "tier1/sync/sync_daily_usd.csv")["FGBL"].to_list()[1]
    assert a == pytest.approx((1 + 0.01) * (1.20 / 1.10) - 1)   # async used fx_async
    assert s == pytest.approx((1 + 0.01) * (1.15 / 1.10) - 1)   # sync used fx_sync
