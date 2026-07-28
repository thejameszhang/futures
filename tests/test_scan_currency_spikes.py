from datetime import date

import polars as pl

from scripts.scan_currency_spikes import scan_prod_currency_spikes


def test_flags_only_currency_spikes_over_threshold():
    df = pl.DataFrame({
        "date": [date(2014, 2, 13), date(2014, 2, 14), date(2014, 2, 18)],
        "KRW": [0.01, 47.75, -0.98],   # two spike rows
        "6E":  [0.01, 0.02, -0.01],    # clean
        "bond_X": [99.0, 99.0, 99.0],  # not a currency column -> ignored
    })
    out = scan_prod_currency_spikes(df, currency_symbols=["KRW", "6E"], threshold=0.30)
    assert out["symbol"].to_list() == ["KRW", "KRW"]
    assert out["date"].to_list() == [date(2014, 2, 14), date(2014, 2, 18)]
