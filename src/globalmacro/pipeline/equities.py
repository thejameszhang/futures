import copy
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

def main():
    spot_equity_returns = []
    for equity in tqdm(equities):
        schedule_dates = get_schedule_dates(equity.exchange_pmc_name)
        spot_equity_returns.append(
            DS2INDEXDATA
            .filter((pl.col("symbol") == equity.symbol) & (pl.col("date").is_in(schedule_dates)))
            .sort("date")
            .with_columns((pl.col("pi_") / pl.col("pi_").shift(1) - 1).alias("pi_ret"))
        )

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
