from datetime import date

import polars as pl

from globalmacro.utils.characteristics import calc_returns_until_expiry


def _apply(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(calc_returns_until_expiry(1, "settlement"))


def test_a_same_contract_return_is_computed_normally():
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "daystomaturity_1": [19, 18],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 5, 15)],
        "settlement_1": [100.0, 110.0],
        "settlement_2": [101.0, 111.0],
    })
    got = _apply(df).get_column("ret_temp_1").to_list()
    assert got[1] is not None and abs(got[1] - 0.1) < 1e-12


def test_the_documented_roll_case_still_uses_yesterdays_slot_two():
    """When the front drops out and everything shifts up, today's slot 1 IS yesterday's
    slot 2 -- the same contract. This return is correct and must survive."""
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "daystomaturity_1": [19, 73],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 5, 15)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 8, 20)],
        "settlement_1": [100.0, 220.0],
        "settlement_2": [200.0, 300.0],
    })
    got = _apply(df).get_column("ret_temp_1").to_list()
    assert got[1] is not None and abs(got[1] - 0.1) < 1e-12


def test_a_cross_contract_return_is_nulled():
    """The defect: slot 1 holds a far-dated contract today and the near one yesterday, and
    yesterday's slot 2 is not it either. Neither comparison refers to one contract."""
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "daystomaturity_1": [19, 1000],
        "lasttrddate_1": [date(2020, 3, 21), date(2022, 12, 1)],
        "lasttrddate_2": [date(2020, 5, 15), date(2023, 3, 1)],
        "settlement_1": [100.0, 500.0],
        "settlement_2": [101.0, 505.0],
    })
    assert _apply(df).get_column("ret_temp_1").to_list()[1] is None


def test_the_expiry_day_roll_is_preserved():
    """daystomaturity_1.shift(1) == 0: yesterday was the front's last trading day, so
    today's slot 1 is yesterday's slot 2. Same contract, must survive."""
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 20), date(2020, 3, 23)],
        "daystomaturity_1": [0, 53],
        "lasttrddate_1": [date(2020, 3, 20), date(2020, 5, 15)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 8, 20)],
        "settlement_1": [100.0, 220.0],
        "settlement_2": [200.0, 300.0],
    })
    got = _apply(df).get_column("ret_temp_1").to_list()
    assert got[1] is not None and abs(got[1] - 0.1) < 1e-12


def test_the_guard_is_scoped_per_class():
    """A shift must never reach across clscodes."""
    df = pl.DataFrame({
        "clscode": [1, 2],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "daystomaturity_1": [19, 18],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 5, 15)],
        "settlement_1": [100.0, 110.0],
        "settlement_2": [101.0, 111.0],
    })
    assert _apply(df).get_column("ret_temp_1").to_list() == [None, None]
