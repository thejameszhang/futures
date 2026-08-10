from datetime import date

import polars as pl

from globalmacro.pipeline.tickhistory import apply_unit_transforms, manual_corrections
from globalmacro.utils.models import Future


class _F(Future):  # minimal Future stand-in; manual_corrections only reads .symbol
    def __init__(self, symbol):
        self.symbol = symbol


def test_krw_substitutes_lasttrdprice_into_front_settlement():
    # KRW: settlement is vwap-derived and corrupt on the fix dates; the TickHistory last
    # trade (lasttrdprice) is clean. c1 front on 2006-09-20 & 2007-03-27; c2 front on 2014-02-14.
    dates = [date(2006, 9, 20), date(2007, 3, 27), date(2014, 2, 14), date(2020, 1, 2)]
    df = pl.DataFrame({
        "date_": dates,
        "settlement_c1":   [0.000105, 0.000538, 1.0,       1.0],   # first two bad
        "lasttrdprice_c1": [0.00105,  0.0010632, 1.0,      1.0],   # clean
        "settlement_c2":   [1.0,      1.0,       0.0466648, 1.0],   # third bad
        "lasttrdprice_c2": [1.0,      1.0,       0.0009428, 1.0],   # clean
        "open_1":          [1.0, 1.0, 1.0, 1.0],
    })
    out = manual_corrections(_F("KRW"), df)
    assert out["settlement_c1"].to_list() == [0.00105, 0.0010632, 1.0, 1.0]
    assert out["settlement_c2"].to_list() == [1.0, 1.0, 0.0009428, 1.0]


def test_pln_substitutes_open_into_front_settlement():
    # PLN: settlement AND lasttrdprice carry the SAME bad print on the fix dates; only the
    # Datastream open_1 is clean (same pattern as the existing HE/ZB/TF branches).
    dates = [date(2008, 7, 16), date(2008, 11, 24), date(2020, 1, 2)]
    df = pl.DataFrame({
        "date_": dates,
        "settlement_c1":   [0.3098, 0.2594, 0.49],
        "lasttrdprice_c1": [0.3098, 0.2594, 0.49],   # also bad
        "open_1":          [0.4890, 0.3327, 0.49],   # clean (Datastream)
    })
    out = manual_corrections(_F("PLN"), df)
    assert out["settlement_c1"].to_list() == [0.4890, 0.3327, 0.49]


def test_manual_corrections_noop_for_symbol_without_fix():
    df = pl.DataFrame({"date_": [date(2020, 1, 2)], "settlement_c1": [1.0]})
    assert manual_corrections(_F("XYZ"), df).equals(df)


def test_6j_6z_match_legacy_inline_transform():
    cols = ["front_month_settlement", "settlement_c1", "settlement_c2", "settlement_c3", "settlement_c4"]
    dates = [date(2000, 1, d) for d in (3, 4, 5)]
    vals = [0.05, 500.0, 0.5]   # spans the 6J (<0.1) and 6Z (>10) predicates
    df = pl.DataFrame({"date_": dates, **{c: list(vals) for c in cols}})
    out_6j = apply_unit_transforms(_F("6J"), df)
    out_6z = apply_unit_transforms(_F("6Z"), df)
    exp_6j = [v * 100 if v < 0.1 else v for v in vals]
    exp_6z = [100 - v if v > 10 else v for v in vals]
    for c in cols:
        assert out_6j[c].to_list() == exp_6j
        assert out_6z[c].to_list() == exp_6z
    assert apply_unit_transforms(_F("6E"), df).equals(df)   # no-op otherwise
