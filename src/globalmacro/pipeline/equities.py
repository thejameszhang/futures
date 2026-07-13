import copy
from datetime import date

import pandas_market_calendars as pmc
import polars as pl
from tqdm import tqdm
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import DATASTREAM_PATH, EQUITIES_PATH, PROJECT_ROOT
        
CACHE = {}

def get_schedule_dates(exchange: str, start_date: str = "1950-01-01", end_date: str = "2025-12-31") -> pl.Series:
    if exchange in CACHE:
        return CACHE[exchange]
    calendar = pmc.get_calendar(exchange)
    schedule = calendar.schedule(start_date=start_date, end_date=end_date)
    schedule_dates = pl.Series(
        "date",
        schedule.index.to_numpy(dtype="datetime64[D]"),
    ).cast(pl.Date).implode()
    CACHE[exchange] = schedule_dates
    return schedule_dates

DAILY_DENSITY = 0.60              # a month is "daily" if it holds >= 60% of its exchange's sessions
DAILY_WINDOW = 12                 # ...and the series starts at the first OBSERVED month whose next 12
                                   # consecutive OBSERVED months all are
MIN_OBSERVATIONS_TO_JUDGE = 50    # below this many real observations, we cannot judge the series at
                                   # all -- see first_daily_date's docstring


def first_daily_date(returns: pl.DataFrame, schedule_dates) -> date | None:
    """The date from which this index began being published as a genuine DAILY series.

    An index that trades a handful of times a month is not a daily price series. Datastream
    carries such series on a continuous grid regardless, so the leading edge must be cut --
    otherwise a 12-observations-a-year era is published as if it were daily.

    A month counts as daily if it holds at least DAILY_DENSITY of the sessions ITS OWN
    EXCHANGE was open that month. The exchange's own calendar is essential: judged against a
    panel-wide calendar, an exchange with a holiday cluster the rest of the world does not
    share -- Chinese New Year, say -- looks sub-daily for a month and is wrongly truncated.

    The window must be FULL: all() of a truncated slice would let a sparse series with a
    couple of dense months at the very end pass. The window counts consecutive OBSERVED
    months, not consecutive calendar months: a month with zero observations is simply absent
    from the sequence the window walks, so it is stepped over rather than breaking the run.
    Harmless for today's universe (no index has a month with literally zero observations
    embedded inside an otherwise-daily era), but a hypothetical series with a calendar gap --
    say 12 dense Januaries, one per year for 12 years, and nothing else -- would satisfy this
    rule even though it is not daily by any ordinary reading of the word.

    This also fixes Z's double-count. Z (FTSE 100) is MONTH-END ONLY until 1984, and
    load_synthetic_returns' coalesce(Z, FTALLSH) PREFERS Z -- so a month-end row drops a
    MONTH-LONG return into a daily cell, and the month then compounds FTALLSH's ~21 daily
    returns AND Z's monthly one. Z's month-end era is not daily by this rule, so it is cut and
    the coalesce falls through to FTALLSH -- which is what the splice intends.

    `returns` holds [date, pi_ret] for ONE index; `schedule_dates` is that index's own
    trading calendar. Returns None when no daily start can be established -- fewer than
    MIN_OBSERVATIONS_TO_JUDGE observations, or no DAILY_WINDOW consecutive daily months. The
    caller then leaves the series UNTRUNCATED: we do not delete a series we cannot judge.
    """
    observed = returns.filter(pl.col("pi_ret").is_not_null()).sort("date")
    if observed.height < MIN_OBSERVATIONS_TO_JUDGE:
        return None
    start = observed.get_column("date").min()

    sessions = sorted(d for d in schedule_dates if d >= start)
    if not sessions:
        return None
    available = (
        pl.DataFrame({"date": sessions})
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("sessions"))
    )
    by_month = (
        observed.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(pl.len().alias("observed"))
        .join(available, on="month", how="left")
        .sort("month")
        .with_columns(
            (pl.col("observed") >= DAILY_DENSITY * pl.col("sessions")).alias("is_daily")
        )
    )
    months = by_month.get_column("month").to_list()
    is_daily = by_month.get_column("is_daily").to_list()
    for i in range(len(months)):
        window = is_daily[i:i + DAILY_WINDOW]
        if len(window) < DAILY_WINDOW:
            break                      # a partial window can never establish "daily"
        if all(window):
            return start if i == 0 else months[i]
    return None


def main():
    spot_equity_returns = []
    for equity in tqdm(equities):
        schedule_dates = get_schedule_dates(equity.exchange_pmc_name)
        series = (
            DS2INDEXDATA
            .filter((pl.col("symbol") == equity.symbol) & (pl.col("date").is_in(schedule_dates)))
            # A null price is NOT an observation. Datastream emits rows with no price at all
            # (pi_ = null). Since a return is pi_(t)/pi_(t-1) - 1, ONE null price destroys TWO
            # returns -- its own day (null numerator) AND the next (null denominator) -- so the
            # real move across the gap is never computed.
            # Dropping the row makes the next return span to the last REAL price.
            .filter(pl.col("pi_").is_not_null())
            .sort("date")
            .with_columns((pl.col("pi_") / pl.col("pi_").shift(1) - 1).alias("pi_ret"))
        )
        # Start the series where the index began being published DAILY. Judged against THIS
        # index's own exchange calendar -- a panel-wide one would make Chinese New Year look
        # like sparsity and wrongly truncate CN. Also fixes Z's month-end double-count.
        sessions = set(schedule_dates.explode().to_list())
        cutoff = first_daily_date(series.select("date", "pi_ret"), sessions)
        if cutoff is not None:
            series = series.filter(pl.col("date") >= cutoff)
        spot_equity_returns.append(series)

    spot_equity_returns = (
        pl.concat(spot_equity_returns)
        .pivot(index="date", on="symbol", values="pi_ret")
        .drop("IND", "SET50")
        .sort("date")
    )
    spot_equity_returns.write_csv(EQUITIES_PATH / "spot_equity_returns.csv")

def _build_equity_indices(futures):
    equities = []
    for future in futures:
        if future.dsindexcode is not None:
            if len(future.dsindexcode) > 1:
                historical_future = copy.deepcopy(future)
                historical_future.symbol = f"{future.dsindexmnem[0]}"
                historical_future.dsindexcode = [future.dsindexcode[0]]
                equities.append(historical_future)

                active_future = copy.deepcopy(future)
                active_future.dsindexcode = [future.dsindexcode[1]]
                equities.append(active_future)
            else:
                equities.append(future)
    return equities

if __name__ == "__main__":
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    tier2 = load_config(PROJECT_ROOT / "tier2.yaml")
    equities = _build_equity_indices(futures)
    equities.extend(_build_equity_indices(tier2))

    dsindexcodes = [f.dsindexcode[0] for f in equities]

    key = pl.DataFrame({
        "dsindexcode": dsindexcodes,
        "symbol": [f.symbol for f in equities],
    })

    # Load datastream equity index data
    DS2INDEXDATA = (
        pl.scan_csv(EQUITIES_PATH / "ds2indexdata.csv", schema_overrides={"valuedate": pl.Date})
        .filter(pl.col("dsindexcode").is_in(dsindexcodes))
        .with_columns(pl.col("valuedate").dt.date().alias("date"))
        .drop("valuedate")
        .filter(pl.col("date").dt.weekday().is_in([1, 2, 3, 4, 5]))
        .sort("date")
        .collect()
    )

    # Join key
    DS2INDEXDATA = (
        DS2INDEXDATA
        .join(key, on="dsindexcode", how="left")
        .sort(["dsindexcode", "date"])
    )

    main()
