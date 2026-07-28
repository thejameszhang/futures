import polars as pl
import pytest

from globalmacro.build import GICS_SECTOR_TICKERS, rename_gics_to_tickers


def test_gics_map_covers_the_11_sectors():
    assert set(GICS_SECTOR_TICKERS) == {"10", "15", "20", "25", "30", "35", "40", "45", "50", "55", "60"}
    assert GICS_SECTOR_TICKERS["10"] == "XAE"   # Energy
    assert GICS_SECTOR_TICKERS["50"] == "XAZ"   # Communication Services


def test_rename_renames_and_preserves_values():
    df = pl.DataFrame({"date": [1, 2], "10": [0.1, 0.2], "45": [0.3, 0.4]})
    out = rename_gics_to_tickers(df)
    assert out.columns == ["date", "XAE", "XAK"]
    assert out["XAE"].to_list() == [0.1, 0.2]


def test_rename_rejects_unknown_code():
    df = pl.DataFrame({"date": [1], "99": [0.1]})
    with pytest.raises(ValueError):
        rename_gics_to_tickers(df)
