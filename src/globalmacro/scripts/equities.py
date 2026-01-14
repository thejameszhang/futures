import copy
import pandas_market_calendars as pmc
import polars as pl
from tqdm import tqdm
from utils.config import load_config
from utils.paths import DATASTREAM_PATH, EQUITIES_PATH, PROJECT_ROOT
        
# .filter(~(pl.col("symbol").is_in(["FTALLSH", "HSI"])) | (pl.col("date") >= pl.date(1970, 1, 1)))
# .filter(~(pl.col("symbol").is_in(["AUSTOLD", "AP"])) | (pl.col("date") >= pl.date(1980, 1, 1)))

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
            .with_columns((pl.col("pi_usd") / pl.col("pi_usd").shift(1) - 1).alias("pi_ret"))
        )

    spot_equity_returns = (
        pl.concat(spot_equity_returns)
        .pivot(index="date", on="symbol", values="pi_ret")
        .sort("date")
    )
    print(spot_equity_returns.head())
    spot_equity_returns.write_csv(EQUITIES_PATH / "spot_equity_returns.csv")

if __name__ == "__main__":
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
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

    dsindexcodes = [float(f.dsindexcode[0]) for f in equities]

    key = pl.DataFrame({
        "dsindexcode": dsindexcodes,
        "symbol": [f.symbol for f in equities],
    })

    # Load FX rates from Compustat
    fx_long = (
        pl.read_csv(DATASTREAM_PATH / "comp" / "merged_tousd.csv", schema_overrides={"date": pl.Date})
        .fill_null(strategy="forward")
        .unpivot(index="date", variable_name="isocurrcode", value_name="fx")
        .with_columns(pl.col("fx").cast(pl.Float64).alias("fx"))
    )
    
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

    # Information about the indices
    DS2EQUITYINDEX = (
        pl.scan_csv(EQUITIES_PATH / "ds2equityindex.csv")
        .select(["dsindexcode", "isocurrcode"])
        .collect()
    )

    # Join key and FX rates
    DS2INDEXDATA = (
        DS2INDEXDATA
        .join(DS2EQUITYINDEX, on="dsindexcode", how="left")
        .join(fx_long, on=["date", "isocurrcode"], how="left")
        .sort(["dsindexcode", "date"])
    )

    # Convert to USD and calculate returns
    DS2INDEXDATA = (
        DS2INDEXDATA
        .with_columns((pl.col("pi_") * pl.col("fx")).alias("pi_usd"))
        .with_columns([
            (pl.col("pi_") / pl.col("pi_").shift(1) - 1).over("dsindexcode").alias("pi_ret_local"),
            (pl.col("pi_usd") / pl.col("pi_usd").shift(1) - 1).over("dsindexcode").alias("pi_ret_usd"),
        ])
        .with_columns(pl.col("pi_ret_usd").alias("pi_ret"))
        .join(key, on="dsindexcode", how="left")
    )

    main()
