import polars as pl

from globalmacro.pipeline.tickhistory import (
    attach_front_slot,
    attach_traditional_front_slot,
)


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


def test_no_c1_fallback_in_the_expiry_month():
    # In the expiry month (expiring_this_month == 1) the front price is always c2 --
    # a missing c2 leaves the price null rather than falling back to c1, and the slot
    # still records 2 (the slot the selection targeted, not proof a price exists).
    df = pl.DataFrame({"expiring_this_month": [1], "settlement_c1": [100.0], "settlement_c2": [None]})
    out = attach_front_slot(df)
    assert out["front_month_settlement"].to_list() == [None]
    assert out["front_slot"].to_list() == [2]


# ---------------------------------------------------------------------------
# attach_traditional_front_slot: exhaustive over the reachable front_month domain.
# front_month is expected to land in 1..4, but nothing upstream guarantees it, so the
# fallback branch (null, 0, negatives, and out-of-range values) is exercised alongside
# the four real slots -- both a mislabeled boundary (e.g. rejecting front_month == 4)
# and an always-1 slot rule would still pass a test that only checked 1..4.
# ---------------------------------------------------------------------------

_FRONT_MONTH_DOMAIN = [None, -2, -1, 0, 1, 2, 3, 4, 5, 6, 100]


def test_traditional_front_slot_matches_the_selected_price_over_the_whole_domain():
    n = len(_FRONT_MONTH_DOMAIN)
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [1.0] * n,
        "settlement_c2": [2.0] * n,
        "settlement_c3": [3.0] * n,
        "settlement_c4": [4.0] * n,
    })
    out = attach_traditional_front_slot(df)
    for front_month, price, slot in zip(
        out["front_month"].to_list(), out["front_month_settlement"].to_list(), out["front_slot"].to_list(),
        strict=True,
    ):
        expected_slot = front_month if front_month in (1, 2, 3, 4) else 1
        assert slot == expected_slot, (front_month, price, slot)
        assert price == float(expected_slot), (front_month, price, slot)


def test_traditional_front_slot_matches_the_selected_price_when_some_settlements_are_null():
    n = len(_FRONT_MONTH_DOMAIN)
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [None] * n,
        "settlement_c2": [2.0] * n,
        "settlement_c3": [None] * n,
        "settlement_c4": [4.0] * n,
    })
    out = attach_traditional_front_slot(df)
    slot_to_col = {1: "settlement_c1", 2: "settlement_c2", 3: "settlement_c3", 4: "settlement_c4"}
    for row in out.iter_rows(named=True):
        expected_slot = row["front_month"] if row["front_month"] in (1, 2, 3, 4) else 1
        assert row["front_slot"] == expected_slot, row
        assert row["front_month_settlement"] == row[slot_to_col[expected_slot]], row


def test_traditional_front_slot_domain_is_exactly_one_through_four():
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [1.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c2": [2.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c3": [3.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c4": [4.0] * len(_FRONT_MONTH_DOMAIN),
    })
    out = attach_traditional_front_slot(df)
    assert set(out["front_slot"].to_list()) == {1, 2, 3, 4}
    assert out["front_slot"].null_count() == 0
