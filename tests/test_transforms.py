import polars as pl
from datetime import date
from globalmacro.build import coerce_numeric_data, keep_after_date, set_null_on_date


def test_coerce_casts_numeric_and_nulls_nonnumeric():
    df = pl.DataFrame({"date": [date(2020, 1, 1)], "a": ["1.5"], "b": ["x"]})
    out = coerce_numeric_data(df)
    assert out["a"].to_list() == [1.5]
    assert out["b"].to_list() == [None]


def test_keep_after_date_nulls_values_before_cutoff():
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "a": [1.0, 2.0]})
    out = keep_after_date(df, "a", date(2020, 1, 2), inclusive=True)
    assert out["a"].to_list() == [None, 2.0]


def test_set_null_on_date_blanks_a_single_day():
    df = pl.DataFrame({"date": [date(2020, 1, 1), date(2020, 1, 2)], "a": [1.0, 2.0]})
    out = set_null_on_date(df, "a", date(2020, 1, 1))
    assert out["a"].to_list() == [None, 2.0]
