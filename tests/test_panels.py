from datetime import date

import polars as pl


def test_first_valid_date_ignores_leading_null_and_nan():
    from globalmacro.utils.panels import first_valid_date
    df = pl.DataFrame({
        "date": [date(2000, 1, 3), date(2000, 1, 4), date(2000, 1, 5)],
        "x": [None, float("nan"), 1.0],
    })
    assert first_valid_date(df, "x") == date(2000, 1, 5)


def test_first_finite_date_also_screens_inf():
    from globalmacro.utils.panels import first_finite_date, first_valid_date
    df = pl.DataFrame({
        "date": [date(2000, 1, 3), date(2000, 1, 4)],
        "x": [float("inf"), 2.0],
    })
    # first_valid_date treats inf as present (not null, not NaN) -> 01-03;
    # first_finite_date screens it -> 01-04.
    assert first_valid_date(df, "x") == date(2000, 1, 3)
    assert first_finite_date(df, "x") == date(2000, 1, 4)


def test_first_finite_date_none_when_no_finite_value():
    from globalmacro.utils.panels import first_finite_date
    df = pl.DataFrame({"date": [date(2000, 1, 3)], "x": [None]})
    assert first_finite_date(df, "x") is None


def test_build_still_reexports_the_primitives():
    # validation/synthetic*.py import first_valid_date FROM build; keep that working.
    from globalmacro.build import first_valid_date, is_present_expr  # noqa: F401
