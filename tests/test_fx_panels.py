from datetime import date
from pathlib import Path

import polars as pl

from globalmacro.pipeline.fx import build_datastream_direct_panel, build_synthetic

# USD-per-X anchor bands on 2019-06-28 to catch inversion errors.
ANCHOR = {
    "EUR": (1.10, 1.16), "GBP": (1.24, 1.30), "JPY": (0.0090, 0.0095),
    "AUD": (0.68, 0.72), "CAD": (0.74, 0.78), "CHF": (0.99, 1.04),
    "NZD": (0.65, 0.69), "NOK": (0.114, 0.120), "SEK": (0.104, 0.110),
}


def test_datastream_direct_panel_orientation():
    p = build_datastream_direct_panel()
    assert "USD" in p.columns and (p["USD"] == 1.0).all()
    row = p.filter(pl.col("date") == date(2019, 6, 28))
    for ccy, (lo, hi) in ANCHOR.items():
        v = row.select(pl.col(ccy).cast(pl.Float64)).item()
        assert lo <= v <= hi, f"{ccy}={v} outside [{lo},{hi}] — likely inverted"


FIXTURES = Path(__file__).parent / "fixtures"


def test_build_synthetic_reproduces_cip_fixture():
    """CIP regression guard: build_synthetic on a small committed spot-panel
    fixture (sliced from the shipped fx_sync.csv, 1990-2000, EUR/JPY/USD) must
    exactly reproduce a committed expected ret1 snapshot. The snapshot was
    generated from code already proven byte-identical to the pre-refactor CIP,
    so it inherits that validation. No skip guard: fixtures are committed."""
    spot = pl.read_csv(FIXTURES / "cip_spot_fixture.csv", try_parse_dates=True, infer_schema_length=None)
    expected = pl.read_csv(FIXTURES / "cip_expected_ret1.csv", try_parse_dates=True, infer_schema_length=None)
    out = build_synthetic(spot)

    compared = 0
    for s in ["6E", "6J"]:
        m = (out.select(["date", pl.col(s).alias("mine")])
             .join(expected.select(["date", pl.col(s).alias("expected")]),
                   on="date", how="inner").drop_nulls())
        assert m.height > 100, f"{s}: too few overlapping rows to be a meaningful check"
        compared += m.height
        assert (m["mine"] - m["expected"]).abs().max() < 1e-12, f"{s}: CIP drifted after refactor"
    assert compared > 100
