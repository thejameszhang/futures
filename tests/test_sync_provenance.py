import polars as pl

from globalmacro.pipeline.tickhistory import attach_front_slot


def test_front_slot_records_which_slot_supplied_the_price():
    df = pl.DataFrame({
        "expiring_this_month": [0, 1, 0],
        "settlement_c1": [100.0, 200.0, None],
        "settlement_c2": [101.0, 201.0, 301.0],
    })
    out = attach_front_slot(df)
    assert out["front_month_settlement"].to_list() == [100.0, 201.0, 301.0]
    assert out["front_slot"].to_list() == [1, 2, 2]


def test_front_slot_is_not_recoverable_from_value_when_slots_are_equal():
    df = pl.DataFrame({
        "expiring_this_month": [0, 1],
        "settlement_c1": [100.0, 100.0],
        "settlement_c2": [100.0, 100.0],
    })
    assert attach_front_slot(df)["front_slot"].to_list() == [1, 2]
