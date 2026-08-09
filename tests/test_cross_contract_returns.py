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
    """daystomaturity_1.shift(1) == 0 fires here, but so does the second disjunct --
    lasttrddate_1 (2020-5-15) == lasttrddate_2.shift(1) (2020-5-15). Today's slot 1 is
    yesterday's slot 2, the same contract, and the return must survive.

    Because both disjuncts hold together, this fixture does NOT discriminate the
    days-to-maturity disjunct alone: deleting `(pl.col('daystomaturity_1').shift(1) == 0) | `
    from `roll` leaves lasttrddate_1 == lasttrddate_2.shift(1) still true, so this test keeps
    passing regardless. See test_the_days_to_maturity_disjunct_can_only_be_proven_by_nulling
    for a fixture that isolates the first disjunct -- which, by construction, can only be
    asserted via a null, not a survival.
    """
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


def test_the_days_to_maturity_disjunct_can_only_be_proven_by_nulling():
    """Isolates the first disjunct: daystomaturity_1.shift(1) == 0 fires (yesterday's slot 1
    reached zero days to maturity), while the second disjunct does not -- lasttrddate_1 today
    (2020-3-20) != lasttrddate_2.shift(1) (2020-5-15).

    This branch cannot be pinned down with a survival assertion. Whenever the days-to-maturity
    disjunct fires, `roll` is True, which forces the denominator to lasttrddate_2.shift(1). For
    the second disjunct to be false (so the fixture isolates the first), lasttrddate_1 must
    differ from that same lasttrddate_2.shift(1) -- but that is exactly `same_contract`'s
    condition. So `roll` fired via this disjunct plus the second disjunct being false together
    force same_contract == False, and the guard nulls the cell. There is no fixture where this
    disjunct fires alone and the result survives -- the only honest assertion is the null.

    lasttrddate_1 is held equal to yesterday's lasttrddate_1 (2020-3-20 both days) so that
    deleting the disjunct under test flips `roll` to False and same_contract to True via the
    *other* branch of the `denominator` when/otherwise, which produces a non-null return --
    proving this fixture actually discriminates the mutant.
    """
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 20), date(2020, 3, 23)],
        "daystomaturity_1": [0, 17],
        "lasttrddate_1": [date(2020, 3, 20), date(2020, 3, 20)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 6, 19)],
        "settlement_1": [100.0, 105.0],
        "settlement_2": [200.0, 210.0],
    })
    assert _apply(df).get_column("ret_temp_1").to_list()[1] is None


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


def test_cross_contract_cells_finds_the_defect_and_nothing_else():
    from globalmacro.utils.characteristics import cross_contract_cells
    df = pl.DataFrame({
        "clscode": [1, 1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3), date(2020, 3, 4)],
        "exp_1": [0, 0, 0],
        "daystomaturity_1": [19, 1000, 17],
        "lasttrddate_1": [date(2020, 3, 21), date(2022, 12, 1), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), date(2023, 3, 1), date(2020, 5, 15)],
        "lasttrddate_3": [date(2020, 8, 20), date(2023, 6, 1), date(2020, 8, 20)],
        "ret_1": [None, 4.0, -0.8],
    })
    got = cross_contract_cells(df)
    # 03-03 jumps to a far contract; 03-04 comes back. Both ratios span two contracts.
    assert got.get_column("date").to_list() == [date(2020, 3, 3), date(2020, 3, 4)]


def test_unadjudicable_cells_reports_an_unknown_contract():
    """A non-null return whose denominator contract is unknown cannot be certified either
    way. It must be counted, not silently excluded by the null-propagating `num != den`."""
    from globalmacro.utils.characteristics import unadjudicable_cells
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "exp_1": [0, 0],
        "daystomaturity_1": [19, 18],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 5, 15)],
        "lasttrddate_3": [date(2020, 8, 20), date(2020, 8, 20)],
        "lasttrddate_4": [date(2020, 11, 20), date(2020, 11, 20)],
        "ret_1": [0.01, 0.02],
    })
    # row 0 has no previous row within its class, so its denominator contract is unknown
    assert unadjudicable_cells(df).get_column("date").to_list() == [date(2020, 3, 2)]


def test_the_exp_1_roll_branch_reaches_lasttrddate_3():
    """When exp_1 == 1 the finalised ret_1 is ret_temp_2 (Task 1's shift), so a roll under
    exp_1 == 1 reaches one slot further than under exp_1 == 0: the denominator is
    lasttrddate_3.shift(1), not lasttrddate_2.shift(1). Row 0 forces roll for row 1 via the
    days-to-maturity disjunct (daystomaturity_1.shift(1) == 0). Row 1's lasttrddate_2 -- the
    exp_1 == 1 numerator -- is set equal to row 0's lasttrddate_3 -- the correct denominator
    -- so the pair is provably the same contract: neither cross-contract nor unadjudicable.

    This single fixture discriminates three mutants at once. Each makes row 1's denominator
    resolve to row 0's lasttrddate_2 or lasttrddate_1 instead of lasttrddate_3, so it no
    longer matches the numerator and the return is wrongly flagged cross-contract:
    - the roll branch under exp_1 == 1 using lasttrddate_2.shift(1) (i+1) instead of
      lasttrddate_3.shift(1) (i+2);
    - the numerator ignoring exp_1 (falling back to lasttrddate_1);
    - the denominator ignoring exp_1 (falling back to the exp_1 == 0 branch).
    """
    from globalmacro.utils.characteristics import (
        cross_contract_cells,
        unadjudicable_cells,
    )
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "exp_1": [0, 1],
        "daystomaturity_1": [0, 53],
        "lasttrddate_1": [date(2020, 3, 2), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 8, 20)],
        "lasttrddate_3": [date(2020, 8, 20), date(2020, 11, 20)],
        "ret_1": [None, 0.05],
    })
    assert cross_contract_cells(df).height == 0
    assert unadjudicable_cells(df).height == 0


def test_roll_can_fire_via_the_lasttrddate_disjunct_alone():
    """Isolates the second disjunct of `roll`: row 0's daystomaturity_1 is nonzero (19), so
    the days-to-maturity disjunct is false for row 1. Row 1's lasttrddate_1 is set equal to
    row 0's lasttrddate_2, so only `lasttrddate_1 == lasttrddate_2.shift(1)` sets roll True.
    Under roll the denominator is lasttrddate_2.shift(1), which equals row 1's own
    lasttrddate_1 -- numerator and denominator agree: a preserved same-contract return.

    Dropping this disjunct from `roll` (leaving only the days-to-maturity check) flips roll
    to False for row 1, so the denominator falls back to lasttrddate_1.shift(1) -- row 0's
    lasttrddate_1, a different date -- and the return is wrongly flagged cross-contract.
    """
    from globalmacro.utils.characteristics import (
        cross_contract_cells,
        unadjudicable_cells,
    )
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "exp_1": [0, 0],
        "daystomaturity_1": [19, 17],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 5, 15)],
        "lasttrddate_2": [date(2020, 5, 15), date(2020, 6, 19)],
        "lasttrddate_3": [date(2020, 8, 20), date(2020, 9, 20)],
        "ret_1": [None, 0.03],
    })
    assert cross_contract_cells(df).height == 0
    assert unadjudicable_cells(df).height == 0


def test_unadjudicable_cells_needs_the_or_not_just_a_null_denominator():
    """Guards against narrowing unadjudicable_cells' `num.is_null() | den.is_null()` down to
    `den.is_null()` alone -- the narrowing the docstring warns about, which drops the real
    count 1,082 -> 1,048. Row 1 has exp_1 == 1 with lasttrddate_2 null, so the numerator
    (lasttrddate_2) is null while the denominator -- lasttrddate_2.shift(1), row 0's
    lasttrddate_2 -- is known. A null numerator with a known denominator is only caught by
    the `|` version; `den.is_null()` alone would miss it.
    """
    from globalmacro.utils.characteristics import unadjudicable_cells
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "exp_1": [0, 1],
        "daystomaturity_1": [19, 17],
        "lasttrddate_1": [date(2020, 3, 21), date(2020, 3, 21)],
        "lasttrddate_2": [date(2020, 5, 15), None],
        "lasttrddate_3": [date(2020, 8, 20), date(2020, 11, 20)],
        "ret_1": [None, 0.02],
    })
    assert unadjudicable_cells(df).get_column("date").to_list() == [date(2020, 3, 3)]


def test_cross_contract_cells_ignores_null_returns():
    from globalmacro.utils.characteristics import cross_contract_cells
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date": [date(2020, 3, 2), date(2020, 3, 3)],
        "exp_1": [0, 0],
        "daystomaturity_1": [19, 1000],
        "lasttrddate_1": [date(2020, 3, 21), date(2022, 12, 1)],
        "lasttrddate_2": [date(2020, 5, 15), date(2023, 3, 1)],
        "lasttrddate_3": [date(2020, 8, 20), date(2023, 6, 1)],
        "ret_1": [None, None],
    })
    assert cross_contract_cells(df).height == 0
