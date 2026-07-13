import polars as pl
import pytest
from datetime import date

from globalmacro.build import compute_monthly_returns, filter_dataset_by_monthly_returns


def _daily(values: list[float | None], start: date = date(2020, 1, 1)) -> pl.DataFrame:
    """One business-day-ish column 'X'; calendar days are fine, only the count matters."""
    dates = [start.replace(day=1 + i) for i in range(len(values))]
    return pl.DataFrame({"date": dates, "X": values}, schema={"date": pl.Date, "X": pl.Float64})


def test_default_is_fifteen_observations():
    """The default MUST stay 15 -- every validation call site depends on it."""
    m = compute_monthly_returns(_daily([0.01] * 14))
    assert m.get_column("X").to_list() == [None]

    m = compute_monthly_returns(_daily([0.01] * 15))
    assert m.get_column("X").item() == pytest.approx(1.01**15 - 1)


def test_min_observations_one_keeps_a_fourteen_observation_month():
    """This is BB 1997-12: 14 real returns, discarded by the 15-threshold."""
    m = compute_monthly_returns(_daily([0.01] * 14), min_observations=1)
    assert m.get_column("X").item() == pytest.approx(1.01**14 - 1)


def test_a_month_with_no_observations_is_null_at_every_threshold():
    """The cardinal rule: an absent month is null, NEVER 0.0."""
    for k in (1, 15):
        m = compute_monthly_returns(_daily([None] * 20), min_observations=k)
        assert m.get_column("X").to_list() == [None], f"min_observations={k} fabricated a value"


def test_the_single_observation_exception_is_gone():
    """The `!= 1` clause let a lone daily return pose as a month (CL 1983-03, GC 1977-08)."""
    m = compute_monthly_returns(_daily([-0.00442]))
    assert m.get_column("X").to_list() == [None]


def test_filter_dataset_recomputes_the_monthly_product_on_the_clipped_panel():
    """filter_dataset_by_monthly_returns' subtlety: the monthly product (min_observations=1)
    must be computed on the CLIPPED daily panel, not the original -- otherwise a thin leading
    month that was too sparse to count as the series' START still gets compounded and
    published as if it were a real month.

    A two-column frame: X's January holds only 3 real observations (too thin to establish a
    first FULL month at the default min_observations=15 threshold), so X's series starts in
    February. Y is fully dense throughout, as a control -- it must be untouched.

    This must FAIL if the min_observations=1 call is hoisted above the clipping loop so each
    monthly frame is computed once from the ORIGINAL panel: Jan's 3-observation fragment would
    then compound to a real (non-null) number and be republished in the monthly product, even
    though the daily panel itself was correctly clipped.
    """
    jan_dates = [date(2020, 1, d) for d in range(1, 24)]  # 23 calendar days
    feb_dates = [date(2020, 2, d) for d in range(1, 24)]  # 23 calendar days
    dates = jan_dates + feb_dates

    # X: only 3 real observations in January (the rest null) -- too thin to start the series.
    # Fully observed in February.
    x_values = [0.033, 0.033, 0.033] + [None] * (len(jan_dates) - 3) + [0.01] * len(feb_dates)
    # Y: fully observed throughout -- a dense control that clipping must leave untouched.
    y_values = [0.01] * len(jan_dates) + [0.01] * len(feb_dates)

    daily = pl.DataFrame(
        {"date": dates, "X": x_values, "Y": y_values},
        schema={"date": pl.Date, "X": pl.Float64, "Y": pl.Float64},
    ).sort("date")

    clipped, monthly = filter_dataset_by_monthly_returns(daily)

    # The daily panel: January's X observations are clipped to null (X starts in February).
    jan_clipped = clipped.filter(pl.col("date").dt.month() == 1).get_column("X").to_list()
    assert all(v is None for v in jan_clipped), "X's thin January must be null in the clipped daily panel"

    # The monthly product: January's X cell must ALSO be null -- not the 3-observation
    # fragment's compounded product. Recomputing on the ORIGINAL panel instead of the clipped
    # one is exactly the bug this test exists to catch.
    jan_monthly_x = monthly.filter(pl.col("date").dt.month() == 1).get_column("X").to_list()
    assert jan_monthly_x == [None] or jan_monthly_x == [], (
        f"X's thin leading January was republished in the monthly product: {jan_monthly_x} "
        f"-- the monthly product must be recomputed on the CLIPPED panel, not the original."
    )

    # Y is dense throughout and untouched by clipping; it must still have a real January cell.
    jan_monthly_y = monthly.filter(pl.col("date").dt.month() == 1).get_column("Y").to_list()
    assert jan_monthly_y and jan_monthly_y[0] is not None
