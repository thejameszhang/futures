import argparse
import copy
from datetime import time, date
import polars as pl
import os
os.chdir("../")
from utils.config import load_config
from utils.models import AssetClass
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--asset_class', type=str, default='currency', help='Asset class to process')
args = parser.parse_args()
ASSET_CLASS = AssetClass(args.asset_class)
if ASSET_CLASS is None:
    raise ValueError(f"Invalid asset class: {args.asset_class}")

TICKHISTORY_PATH = "data/tickhistory"
DATASTREAM_PATH = "data/datastream/futures"
CONFIG = []
for future in load_config():
    if future.ric is not None and ASSET_CLASS in future.asset_class:
        if len(future.ric) > 1:
            historical_future = copy.deepcopy(future)
            historical_future.symbol = f"{future.ric[0]}"
            historical_future.ric = [future.ric[0]]
            CONFIG.append(historical_future)

            active_future = copy.deepcopy(future)
            active_future.ric = [future.ric[1]]
            CONFIG.append(active_future)
        else:
            CONFIG.append(future)

if ASSET_CLASS == AssetClass.TRADITIONAL:
    CONFIG = list(filter(lambda f: f.ct is not None, CONFIG))

CLSCODES = [f.clscode for f in CONFIG]
CALCSERIESNAMES = [f.calcseriesname for f in CONFIG]
DATASET_START_DATE = pl.date(1996, 1, 1)
DATASET_END_DATE = pl.date(2025, 6, 11)
TIME_SYNCED = True 
ALL_RICS = []
for future in CONFIG:
    ALL_RICS.extend([f"{future.ric[0]}c1", f"{future.ric[0]}c2", f"{future.ric[0]}c3", f"{future.ric[0]}c4"])

# Full sample daily dataset, series doesnt have price adjustment
CONTINUOUS_SERIES_DATA = (
    pl.scan_csv(f'{DATASTREAM_PATH}/datastream_continuous_series.csv')
    .filter(pl.col('RollMethodCode') == 0)
    .filter(pl.col('PositionFwdCode') == 0)
    .filter(pl.col('ClsCode').is_in(CLSCODES))
    .filter(pl.col('CalcSeriesName').is_in(CALCSERIESNAMES))
    .filter(pl.col('Date_').str.to_date("%Y-%m-%d") >= DATASET_START_DATE)
    .filter(pl.col('Date_').str.to_date("%Y-%m-%d") <= DATASET_END_DATE)
    .collect()
).select(['ClsCode', 'Date_', 'Settlement']).sort(['Date_', 'ClsCode']).with_columns([
    pl.col('Date_').str.to_date("%Y-%m-%d").alias("date_")
]).unique(keep='first').sort(['date_'])

# Load expiry dates for proper rolling 
LSEG_DATA = pl.scan_csv(f"{DATASTREAM_PATH}/lseg/tr_ds_fut_dsfutcontrinfo.csv", ignore_errors=True).with_columns([
    pl.col("startdate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").dt.date().alias("startdate"),
    pl.col("lasttrddate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").dt.date().alias("lasttrddate"),
]).select(['clscode', 'lasttrddate', 'startdate']).collect()

def load_trades_data(path: str) -> pl.DataFrame:
    return pl.scan_csv(f"{TICKHISTORY_PATH}/trades/{path}", dtypes={"Price": pl.Float64, "Volume": pl.Int64}).filter(
        pl.col("#RIC").is_in(ALL_RICS)).select(["#RIC", "Date-Time", "Price", "Volume", "GMT Offset"]).with_columns([
        # Parse datetime including nanoseconds and timezone
        pl.col("Date-Time").str.replace("Z$", "+00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z").alias("datetime"),
        # Ordering contracts
        pl.col("#RIC").str.extract(r"c(\d)$").cast(pl.UInt8).alias("order"),
        # Ensure GMT Offset is numeric for comparisons
        pl.col("GMT Offset").cast(pl.Int8).alias("GMT Offset"),
        # Volume assumption
        pl.when((pl.col("Volume") <= 0.0) | pl.col("Volume").is_null() | pl.col("Volume").is_nan()).then(1.0).otherwise(pl.col("Volume")).alias("Volume"),
        # Trade price adjustment because 0.0 price doesnt make sense
        pl.when(pl.col("Price") == 0.0).then(pl.lit(None)).otherwise(pl.col("Price")).alias("Price")
    ]).unique(keep='first').with_columns([
        (pl.col("datetime") + pl.duration(hours=pl.col("GMT Offset"))).alias("local_datetime"),
        pl.col("datetime").dt.convert_time_zone("America/New_York").alias("datetime_et")
    ]).filter(pl.col("Price").is_not_null()).collect(streaming=True).sort("local_datetime")

def load_quotes_data(path: str) -> pl.DataFrame:
    return pl.scan_csv(f"{TICKHISTORY_PATH}/quotes/{path}", dtypes={"Close Bid": pl.Float64, "Close Ask": pl.Float64, "GMT Offset": pl.Int8}
    ).filter((pl.col("#RIC").is_in(ALL_RICS)) & (pl.col("Type") == "Intraday 1Min")).select([
        "#RIC", "Date-Time", "Close Bid", "Close Ask", "GMT Offset"
    ]).with_columns([
        # Parse datetime including nanoseconds and timezone
        pl.col("Date-Time").str.replace("Z$", "+00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z").alias("datetime"),
        # Ordering contracts
        pl.col("#RIC").str.extract(r"c(\d)$").cast(pl.UInt8).alias("order"),
    ]).unique(keep='first').with_columns([
        (pl.col("datetime") + pl.duration(hours=pl.col("GMT Offset"))).alias("local_datetime"),
        pl.col("datetime").dt.convert_time_zone("America/New_York").alias("datetime_et")
    ]).filter((pl.col("Close Bid").is_not_null()) & (pl.col("Close Ask").is_not_null())).collect(streaming=True).sort("local_datetime")

def count_expiries_between(row_date, row_trad_last):
    """Helper function used to deterine the front month contract for CT futures"""
    if row_date is None or row_trad_last is None:
        return None
    cnt = 0
    for d in expiry_list:
        if d >= row_date and d <= row_trad_last:
            cnt += 1
    return cnt

filename = ASSET_CLASS.value if "equity" not in ASSET_CLASS.value else "equity"
TRADES_DATA = load_trades_data(f"tier1_{filename}_trades.csv")
QUOTES_DATA = load_quotes_data(f"tier1_{filename}_quotes.csv")

returns_data = []
for FUTURE in CONFIG:
    print(f"\n{'='*50}")
    print(f"Processing {FUTURE.symbol}")
    print(f"{'='*50}")
    DATETIME_COLUMN = "datetime_et" if TIME_SYNCED else "local_datetime"
    SETTLEMENT_START = time(9, 30) if TIME_SYNCED else FUTURE.settlement_start
    SETTLEMENT_END = time(9, 31) if TIME_SYNCED else FUTURE.settlement_end

    FUTURE_RICS = [f"{FUTURE.ric[0]}c1", f"{FUTURE.ric[0]}c2", f"{FUTURE.ric[0]}c3", f"{FUTURE.ric[0]}c4"]
    trades_data = TRADES_DATA.filter(pl.col("#RIC").is_in(FUTURE_RICS)).with_columns(
        pl.col(DATETIME_COLUMN).dt.date().alias("date_")
    ).sort([DATETIME_COLUMN])
    quotes_data = QUOTES_DATA.filter(pl.col("#RIC").is_in(FUTURE_RICS)).with_columns(
        pl.col(DATETIME_COLUMN).dt.date().alias("date_")
    ).sort([DATETIME_COLUMN])
    
    # Take the union of dates in the trades and quotes data
    dates_trades = pl.DataFrame({"date_": trades_data["date_"].unique().sort()})
    dates_quotes = pl.DataFrame({"date_": quotes_data["date_"].unique().sort()})
    dates = pl.concat([dates_quotes, dates_trades]).unique(subset=["date_"]).drop_nans().sort("date_")

    print(trades_data.select(["GMT Offset"]).unique())

    time_filtered_data = trades_data.filter(
        (pl.col(DATETIME_COLUMN).dt.time() >= SETTLEMENT_START) & 
        (pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_END)
    )

    # Compute VWAP in the designated settlement interval 
    daily_vwaps = time_filtered_data.filter(
        pl.col("Price").is_not_null() & pl.col("Volume").is_not_null()
    ).group_by(["#RIC", "date_", "order"]).agg([
        (pl.col("Price") * pl.col("Volume")).sum().alias("price_volume_sum"),
        pl.col("Volume").sum().alias("total_volume"),
    ]).with_columns([
        (pl.col("price_volume_sum") / pl.col("total_volume")).alias("vwap")
    ]).sort(["#RIC", "date_", "order"]).with_columns([
        (pl.col("vwap").pct_change().over(["#RIC", "order"])).alias("daily_return")
    ]).select(["#RIC", "date_", "order", "vwap", "daily_return"]).sort(["date_", "order"])
    daily_vwaps_c1 = daily_vwaps.filter(pl.col("order") == 1).select(["date_", "vwap"]).rename({"vwap": "vwap_c1"})
    daily_vwaps_c2 = daily_vwaps.filter(pl.col("order") == 2).select(["date_", "vwap"]).rename({"vwap": "vwap_c2"})
    daily_vwaps_c3 = daily_vwaps.filter(pl.col("order") == 3).select(["date_", "vwap"]).rename({"vwap": "vwap_c3"})
    daily_vwaps_c4 = daily_vwaps.filter(pl.col("order") == 4).select(["date_", "vwap"]).rename({"vwap": "vwap_c4"})
    daily_vwaps = (dates.join(daily_vwaps_c1, on=["date_"], how="left").join(
        daily_vwaps_c2, on=["date_"], how="left").join(
        daily_vwaps_c3, on=["date_"], how="left").join(
        daily_vwaps_c4, on=["date_"], how="left")
    .sort(["date_"]))
    
    # Compute TODAYS last traded prices prior to the settlement start time
    trades_data = trades_data.filter((pl.col("Price").is_not_null()) & (pl.col("Volume").is_not_null()))
    today_lasttrdprices = trades_data.filter(
        pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_START
    ).group_by(["#RIC", "date_", "order"]).agg([
        pl.col("Price").sort_by(pl.col(DATETIME_COLUMN)).last().alias("lasttrdprice_today")
    ]).sort(["date_"])

    # Compute last traded price for ALL days (regardless of time); does not include days where there is no trade data
    all_lasttrdprices = trades_data.group_by(["#RIC", "date_", "order"]).agg([
        pl.col("Price").sort_by(pl.col(DATETIME_COLUMN)).last().alias("lasttrdprice")
    ]).sort(["#RIC", "date_", "order"])

    # Shift dates by +1 day so backward join only matches previous days, and not the current day (lookahead bias)
    all_lasttrdprices_shifted = all_lasttrdprices.with_columns([
        (pl.col("date_") + pl.duration(days=1)).alias("fallback_date")
    ]).select([
        "#RIC",
        "fallback_date",
        "order",
        "lasttrdprice"
    ])

    # Get all unique dates and RIC/order combinations to create full grid; all dates × all RIC/order combinations
    ric_order_combos = trades_data.select(["#RIC", "order"]).unique()
    full_grid = dates.join(ric_order_combos, how="cross").sort(["date_", "#RIC", "order"])
    lasttrdprices = full_grid.join(
        today_lasttrdprices,
        on=["#RIC", "date_", "order"],
        how="left"
    ).join_asof(
        all_lasttrdprices_shifted,
        left_on="date_",
        right_on="fallback_date",
        by=["#RIC", "order"],
        strategy="backward"
    )
    
    # Create mapping of each date to its previous trading day
    prev_trading_day = dates.with_columns([
        pl.col("date_").shift(1).alias("prev_trading_day")
    ]).filter(pl.col("prev_trading_day").is_not_null())
    
    # Join to get previous trading day for each date, then check if fallback matches
    lasttrdprices = lasttrdprices.join(
        prev_trading_day,
        on="date_",
        how="left"
    ).with_columns([
        # Only use fallback if it's from the previous trading day
        # If today's price exists, use it; otherwise use fallback only if it's from prev trading day
        pl.when(pl.col("lasttrdprice_today").is_not_null())
        .then(pl.col("lasttrdprice_today"))
        # IMPORTANT: Reverse the previous +1 day shift to fallback_date 
        .when(pl.col("fallback_date") - pl.duration(days=1) == pl.col("prev_trading_day"))
        .then(pl.col("lasttrdprice"))
        .otherwise(None)
        .alias("lasttrdprice")
    ]).select(["#RIC", "date_", "order", "lasttrdprice"]).sort(["date_", "order"])    
    
    # Renaming for c1, c2, c3, and c4
    lasttrdprice_c1 = lasttrdprices.filter(pl.col("order") == 1).select(["date_", pl.col("lasttrdprice").alias("lasttrdprice_c1")])
    lasttrdprice_c2 = lasttrdprices.filter(pl.col("order") == 2).select(["date_", pl.col("lasttrdprice").alias("lasttrdprice_c2")])
    lasttrdprice_c3 = lasttrdprices.filter(pl.col("order") == 3).select(["date_", pl.col("lasttrdprice").alias("lasttrdprice_c3")])
    lasttrdprice_c4 = lasttrdprices.filter(pl.col("order") == 4).select(["date_", pl.col("lasttrdprice").alias("lasttrdprice_c4")])
    fallback_prices = (dates.join(lasttrdprice_c1, on=["date_"], how="left").join(
        lasttrdprice_c2, on=["date_"], how="left").join(
        lasttrdprice_c3, on=["date_"], how="left").join(
        lasttrdprice_c4, on=["date_"], how="left")
    .sort(["date_"]))
    daily_vwaps = daily_vwaps.join(fallback_prices, on=["date_"], how="left")

    # Check the last trade price against the bid-ask spread; if outside, use the midpoint
    last_bid_ask_spread = quotes_data.filter(pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_END).with_columns([
        pl.col(DATETIME_COLUMN).dt.date().alias("date_")
    ]).group_by(["#RIC", "date_", "order"]).agg([
        pl.col("Close Bid").sort_by(pl.col(DATETIME_COLUMN)).last().alias("last_bid"),
        pl.col("Close Ask").sort_by(pl.col(DATETIME_COLUMN)).last().alias("last_ask"),
    ]).sort(["date_"])
    last_bid_ask_spread_c1 = last_bid_ask_spread.filter(pl.col("order") == 1).select(["date_", pl.col("last_bid").alias("last_bid_c1"), pl.col("last_ask").alias("last_ask_c1")])
    last_bid_ask_spread_c2 = last_bid_ask_spread.filter(pl.col("order") == 2).select(["date_", pl.col("last_bid").alias("last_bid_c2"), pl.col("last_ask").alias("last_ask_c2")])
    last_bid_ask_spread_c3 = last_bid_ask_spread.filter(pl.col("order") == 3).select(["date_", pl.col("last_bid").alias("last_bid_c3"), pl.col("last_ask").alias("last_ask_c3")])
    last_bid_ask_spread_c4 = last_bid_ask_spread.filter(pl.col("order") == 4).select(["date_", pl.col("last_bid").alias("last_bid_c4"), pl.col("last_ask").alias("last_ask_c4")])
    bid_ask_spread = (dates.join(last_bid_ask_spread_c1, on=["date_"], how="left").join(
        last_bid_ask_spread_c2, on=["date_"], how="left").join(
        last_bid_ask_spread_c3, on=["date_"], how="left").join(
        last_bid_ask_spread_c4, on=["date_"], how="left")
    .sort(["date_"]))
    daily_vwaps = daily_vwaps.join(bid_ask_spread, on=["date_"], how="left").with_columns([
        ((pl.col("last_bid_c1") + pl.col("last_ask_c1")) / 2.0).alias("midpoint_c1"),
        ((pl.col("last_bid_c2") + pl.col("last_ask_c2")) / 2.0).alias("midpoint_c2"),
        ((pl.col("last_bid_c3") + pl.col("last_ask_c3")) / 2.0).alias("midpoint_c3"),
        ((pl.col("last_bid_c4") + pl.col("last_ask_c4")) / 2.0).alias("midpoint_c4"),
    ])

    def compute_settlement_price(i: int):
        return pl.when(pl.col(f"vwap_c{i}").is_not_null()).then(pl.col(f"vwap_c{i}")).otherwise(
            pl.when(
                (pl.col(f"lasttrdprice_c{i}") >= pl.col(f"last_bid_c{i}")) & (pl.col(f"lasttrdprice_c{i}") <= pl.col(f"last_ask_c{i}"))
            ).then(
                pl.col(f"lasttrdprice_c{i}")
            ).otherwise(pl.coalesce(pl.col(f"midpoint_c{i}"), pl.col(f"lasttrdprice_c{i}")))
        ).alias(f"settlement_c{i}")
    daily_vwaps = daily_vwaps.with_columns([
        compute_settlement_price(1),
        compute_settlement_price(2),
        compute_settlement_price(3),
        compute_settlement_price(4),
    ])

    daily_vwaps = daily_vwaps.filter(
        (pl.col("settlement_c1").is_not_null()) | 
        (pl.col("settlement_c2").is_not_null()) | 
        (pl.col("settlement_c3").is_not_null()) | 
        (pl.col("settlement_c4").is_not_null())
    )

    # For each date_ in daily_vwaps, assign the lasttrddate that is closest to the date_ but in the future
    expiry_dates = LSEG_DATA.filter(pl.col('clscode') == FUTURE.clscode).sort(['lasttrddate']).select([pl.col('startdate'), pl.col('lasttrddate')])
    expiry_list = expiry_dates["lasttrddate"].to_list()
    daily_vwaps = daily_vwaps.join_asof(expiry_dates, left_on="date_", right_on="lasttrddate", strategy="forward").with_columns([
        pl.when(
            (pl.col("date_").dt.year() == pl.col("lasttrddate").dt.year()) & (pl.col("date_").dt.month() == pl.col("lasttrddate").dt.month())
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("expiring_this_month"),
        pl.col("date_").dt.month().alias("current_month"),
    ])
    
    # Determine the front month contract
    if ASSET_CLASS == AssetClass.TRADITIONAL:
        # Use this column trad_lasttrddate for knowing when an active contract expires
        trad_expiry_dates = expiry_dates.filter(pl.col("lasttrddate").dt.month().is_in(FUTURE.ct)).with_columns([
            pl.col("lasttrddate").alias("trad_lasttrddate")
        ]).select([pl.col("trad_lasttrddate")]).sort(["trad_lasttrddate"])
        daily_vwaps = daily_vwaps.join_asof(trad_expiry_dates, left_on="date_", right_on="trad_lasttrddate", strategy="forward")

        daily_vwaps = daily_vwaps.with_columns([
            # month_diff really represents the number of contracts valid between date_ and trad_lasttrddate
            pl.struct(["date_", "trad_lasttrddate"]).map_elements(
                lambda s: count_expiries_between(s["date_"], s["trad_lasttrddate"]),
                return_dtype=pl.Int32,
            ).alias("month_diff"),
        ]).with_columns([
            # Special logic for SI and HG futures with trading cycle [3, 5, 7, 9, 12]
            pl.when((FUTURE.symbol == "SI") | (FUTURE.symbol == "HG")).then(
                pl.when(pl.col("month_diff") == 1).then(
                    pl.when(pl.col("expiring_this_month") == 1).then(
                        pl.when(pl.col("current_month").is_in([3, 5, 7])).then(pl.lit(3))
                        .when(pl.col("current_month").is_in([9, 12])).then(pl.lit(4))
                        .otherwise(pl.lit(1))  # fallback, we should never hit here
                    ).otherwise(pl.lit(1))
                )
                .when(pl.col("month_diff") == 2).then(pl.lit(2))
                .when(pl.col("month_diff") == 3).then(pl.lit(3))
                # Default fallback for other month_diff values; should never hit here
                .otherwise(
                    pl.when(pl.col("expiring_this_month") == 1).then(pl.lit(2)).otherwise(pl.lit(1))
                ))    
            # Otherwise, they have a traditional cycle of [3, 6, 9, 12]
            .otherwise(
                pl.when(pl.col("month_diff") == 1).then(
                    pl.when(pl.col("expiring_this_month") == 1).then(pl.lit(4)).otherwise(pl.lit(1))
                )
                .when(pl.col("month_diff") == 2).then(pl.lit(2))
                .when(pl.col("month_diff") == 3).then(pl.lit(3))
                .otherwise(pl.lit(1))
            ).alias("front_month")
        ]).with_columns([
            # Now use front_month to select the appropriate settlement price
            pl.when(pl.col("front_month") == 1).then(pl.col("settlement_c1"))
            .when(pl.col("front_month") == 2).then(pl.col("settlement_c2"))
            .when(pl.col("front_month") == 3).then(pl.col("settlement_c3"))
            .when(pl.col("front_month") == 4).then(pl.col("settlement_c4"))
            .otherwise(pl.col("settlement_c1"))  # fallback
            .alias("front_month_settlement")
        ])
    else:
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("expiring_this_month") == 1).then(pl.col("settlement_c2")).otherwise(pl.col("settlement_c1")).alias("front_month_settlement")
        ])
        # If we should use the current month's data, but we can't determine a settlement price for it, 
        # then use the back month's data 
        daily_vwaps = daily_vwaps.with_columns([
            pl.when((pl.col("expiring_this_month") == 0) & (pl.col("front_month_settlement").is_null())).then(
                pl.col("settlement_c2")
            ).otherwise(
                pl.col("front_month_settlement")
            ).alias("front_month_settlement")
        ])

    # Join official data from LSEG to see if they round their settlement prices
    continuous_series_data = CONTINUOUS_SERIES_DATA.filter(pl.col("ClsCode") == FUTURE.clscode).sort(["date_"])
    prices = daily_vwaps.join(continuous_series_data, on=["date_"], how="left")

    # No weekends in our final returns dataset
    prices = prices.filter((pl.col("date_").dt.weekday() != 6) & (pl.col("date_").dt.weekday() != 7))

    # Historical price adjustment when the exchange changes contract details -> scales the price appropriately
    if FUTURE.adjustments is not None:
        for adjustment in FUTURE.adjustments:
            prices = prices.with_columns([
                pl.when(pl.col("front_month_settlement") > adjustment.get("threshold")).then(pl.col("front_month_settlement") / adjustment.get("divisor")).otherwise(pl.col("front_month_settlement")).alias("front_month_settlement")
            ])
    # This is intentional; this is a bad way to do it but it's okay
    if FUTURE.symbol == "6J":
        prices = prices.with_columns([
            pl.when(pl.col("front_month_settlement") < 0.1).then(pl.col("front_month_settlement") * 100).otherwise(pl.col("front_month_settlement")).alias("front_month_settlement")
        ])

    # Returns ret1 without any price adjustment + other returns ret_c1 and ret_c2 which are just the returns for c1 and c2 series
    symbol_data = prices.with_columns([
        (pl.col("front_month_settlement")/pl.col("front_month_settlement").shift(1) - 1).alias("ret1"),
        (pl.col("settlement_c1")/pl.col("settlement_c2").shift(1) - 1).alias("rollback_c2_to_c1"),
        (pl.col("settlement_c2")/pl.col("settlement_c3").shift(1) - 1).alias("rollback_c3_to_c2"),
        (pl.col("settlement_c3")/pl.col("settlement_c4").shift(1) - 1).alias("rollback_c4_to_c3"),
        (pl.col("settlement_c1")/pl.col("settlement_c1").shift(1) - 1).alias("ret_c1"),
        (pl.col("settlement_c2")/pl.col("settlement_c2").shift(1) - 1).alias("ret_c2"),
        (pl.col("settlement_c3")/pl.col("settlement_c3").shift(1) - 1).alias("ret_c3"),
        (pl.col("settlement_c4")/pl.col("settlement_c4").shift(1) - 1).alias("ret_c4"),
        (pl.col("expiring_this_month").diff()).alias("shift")
    ])

    # Compute the front month return with price adjustment
    if ASSET_CLASS == AssetClass.TRADITIONAL:
        # Normal cases: if lasttrddate is somewhere in the middle of the month
        symbol_data = symbol_data.with_columns([
            pl.when(pl.col("shift") == 1).then(
                pl.when(pl.col("front_month") == 1).then(pl.col("ret_c1"))
                .when(pl.col("front_month") == 2).then(pl.col("ret_c2"))
                .when(pl.col("front_month") == 3).then(pl.col("ret_c3"))
                .when(pl.col("front_month") == 4).then(pl.col("ret_c4"))
                .otherwise(pl.col("ret1"))
            ).when(pl.col("shift") == -1).then(
                pl.when(pl.col("front_month") == 1).then(pl.col("rollback_c2_to_c1"))
                .when(pl.col("front_month") == 2).then(pl.col("rollback_c3_to_c2"))
                .when(pl.col("front_month") == 3).then(pl.col("rollback_c4_to_c3"))
                .otherwise(pl.col("ret1"))
            ).otherwise(
                pl.when(pl.col("expiring_this_month") == 1).then(
                    pl.coalesce(pl.col("ret1"), pl.col("ret_c2"), pl.col("ret_c1"))
                ).otherwise(pl.coalesce(pl.col("ret1"), pl.col("ret_c1"), pl.col("ret_c2")))
            )
            .alias("ret1_adjusted")
        ])
    else:
        # Normal cases: if lasttrddate is somewhere in the middle of the month
        # When shift = 1, we switched from 0 to 1 ie. from c1 to c2 -> should use ret_c2
        # When shift = -1, we switched from 1 to 0 ie. from c2 to c1 -> we should use the return from settlement_c2 to settlement_c1 (actually the same contract)
        # When shift = 0, we didn't switch contracts -> we should use the daily return
        symbol_data = symbol_data.with_columns([
            pl.when(pl.col("shift") == 1).then(pl.col("ret_c2")).when(
                pl.col("shift") == -1).then(pl.col("rollback_c2_to_c1")
            ).otherwise(
                pl.when(pl.col("expiring_this_month") == 1).then(
                    pl.coalesce(pl.col("ret1"), pl.col("ret_c2"))
                ).otherwise(pl.coalesce(pl.col("ret1"), pl.col("ret_c1"), pl.col("ret_c2")))
            ).alias("ret1_adjusted"),
        ]).sort("date_")
        
        # Edge case: if lasttrddate is on the last trading day of the month
        # Adjustment for futures (RB, and more) who have expirations on the last trading day of the month, so shift is always set to 1. 
        symbol_data = symbol_data.with_columns([
            pl.when(
                (pl.col("expiring_this_month") == 1) & 
                (pl.col("expiring_this_month").shift(1) == 1) & 
                (pl.col("date_").shift(1) == pl.col("lasttrddate").shift(1))
            ).then(pl.coalesce(pl.col("rollback_c2_to_c1"), pl.col("ret1_adjusted"))).otherwise(pl.col("ret1_adjusted")).alias("ret1_adjusted"),
        ]).sort("date_")

    returns_series = symbol_data.select(["date_", "ret1_adjusted"]).with_columns([
        pl.lit(FUTURE.symbol).alias("symbol")
    ]).sort("date_")
    returns_data.append(returns_series)

all_returns = pl.concat(returns_data, how="vertical_relaxed")
returns_wide = all_returns.pivot(on="symbol", values="ret1_adjusted", index="date_").sort("date_").with_columns(
    pl.col("date_").alias("date")
)
returns_wide.write_csv(f"datasets/tier1/sync/{ASSET_CLASS.name.lower()}_daily_returns.csv")
