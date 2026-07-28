from datetime import date

import polars as pl
import pytest

from globalmacro.pipeline.to_usd import usd_panel


def test_identity_no_gaps():
    d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    local = pl.DataFrame({"date": d, "FGBL": [None, 0.01, -0.02]})
    fx = pl.DataFrame({"date": d, "EUR": [1.10, 1.11, 1.09]})
    v = usd_panel(local, {"FGBL": "EUR"}, fx)["FGBL"].to_list()
    assert v[0] is None
    assert v[1] == pytest.approx((1 + 0.01) * (1.11 / 1.10) - 1)
    assert v[2] == pytest.approx((1 - 0.02) * (1.09 / 1.11) - 1)


def test_usd_symbol_passthrough():
    d = [date(2020, 1, 1), date(2020, 1, 2)]
    local = pl.DataFrame({"date": d, "ES": [None, 0.01]})
    fx = pl.DataFrame({"date": d, "EUR": [1.1, 1.2]})
    assert usd_panel(local, {"ES": "USD"}, fx)["ES"].to_list() == [None, 0.01]


def test_gap_uses_previous_observation():
    d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]
    local = pl.DataFrame({"date": d, "FGBL": [None, 0.01, None, 0.02]})
    fx = pl.DataFrame({"date": d, "EUR": [1.10, 1.11, 1.12, 1.13]})
    v = usd_panel(local, {"FGBL": "EUR"}, fx)["FGBL"].to_list()
    assert v[2] is None
    assert v[3] == pytest.approx((1 + 0.02) * (1.13 / 1.11) - 1)


def test_first_observation_uses_prev_grid_fx():
    d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]
    local = pl.DataFrame({"date": d, "FGBL": [None, 0.01, 0.02]})
    fx = pl.DataFrame({"date": d, "EUR": [1.10, 1.11, 1.12]})
    v = usd_panel(local, {"FGBL": "EUR"}, fx)["FGBL"].to_list()
    assert v[1] == pytest.approx((1 + 0.01) * (1.11 / 1.10) - 1)


def test_fx_holiday_yields_zero_fx_return():
    # FX does not quote on d3 (FX holiday) while the asset trades every day.
    d = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]
    local = pl.DataFrame({"date": d, "FGBL": [None, 0.01, 0.02, 0.03]})
    fx = pl.DataFrame({"date": d, "EUR": [1.10, 1.11, None, 1.13]})  # d3 FX holiday
    v = usd_panel(local, {"FGBL": "EUR"}, fx)["FGBL"].to_list()
    # d3: FX forward-fills to 1.11 -> r_fx = 0 -> USD return == local return
    assert v[2] == pytest.approx(0.02)
    # d4: FX move measured over d2->d4 via the ffilled prev observation (1.11)
    assert v[3] == pytest.approx((1 + 0.03) * (1.13 / 1.11) - 1)


def test_unmapped_symbol_raises():
    d = [date(2020, 1, 1)]
    with pytest.raises(ValueError, match="No currency mapping"):
        usd_panel(pl.DataFrame({"date": d, "FGBL": [0.0]}), {}, pl.DataFrame({"date": d, "EUR": [1.1]}))


def test_missing_fx_currency_raises():
    d = [date(2020, 1, 1), date(2020, 1, 2)]
    with pytest.raises(ValueError, match="missing currency columns"):
        usd_panel(pl.DataFrame({"date": d, "FGBL": [None, 0.01]}), {"FGBL": "EUR"},
                  pl.DataFrame({"date": d, "JPY": [100.0, 101.0]}))


def test_sync_panel_first_observation_recovered_from_earlier_fx():
    """Sync panels begin exactly at the symbol's first observation — the panel's
    first row has no prior grid row, so level.shift(1) would otherwise be null on
    that row and the whole first observation would be dropped. usd_panel must seed
    the FX grid with the most recent fx_levels row strictly before the panel's
    first date so the first USD return is recoverable."""
    d = [date(2020, 1, 2), date(2020, 1, 3)]  # panel's FIRST row IS the first observation
    local = pl.DataFrame({"date": d, "FGBL": [0.01, -0.02]})
    fx = pl.DataFrame({
        "date": [date(2019, 12, 31)] + d,  # earlier fx history available, unlike local
        "EUR": [1.05, 1.10, 1.11],
    })
    v = usd_panel(local, {"FGBL": "EUR"}, fx)["FGBL"].to_list()
    assert v[0] is not None
    assert v[0] == pytest.approx((1 + 0.01) * (1.10 / 1.05) - 1)
    assert v[1] == pytest.approx((1 - 0.02) * (1.11 / 1.10) - 1)
