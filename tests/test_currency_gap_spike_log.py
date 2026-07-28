from datetime import date

import polars as pl

from globalmacro.build import currency_health


def test_reports_spikes_and_long_gaps():
    df = pl.DataFrame({
        "date": [date(2014, 2, d) for d in (10, 11, 12, 13, 14, 17, 18, 19)],
        "KRW":  [0.0, 0.0, 0.0, 0.0, 47.75, None, None, 0.0],   # spike
        "6E":   [0.0, None, None, None, None, None, None, 0.0],  # 6-day gap
    })
    lines = currency_health(df, ["KRW", "6E"], threshold=0.30)
    assert "currency KRW: 1 spike(s) |ret|>30% (SHIPPED)" in lines
    assert "currency 6E: max null gap 6 days" in lines


def test_leading_trailing_null_pad_is_not_a_gap():
    # Listing-boundary pad: currency lists mid-panel, so the column starts
    # with a multi-day leading null run followed by contiguous data (no
    # interior gap). This must NOT be reported as a gap.
    df = pl.DataFrame({
        "date": [date(2014, 2, d) for d in (10, 11, 12, 13, 14, 17, 18, 19)],
        "PLN":  [None, None, None, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    lines = currency_health(df, ["PLN"], threshold=0.30)
    assert lines == []
