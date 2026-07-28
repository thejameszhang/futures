import argparse
import copy
import os
from datetime import date, time

import polars as pl
from tqdm import tqdm

from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass, Future
from globalmacro.utils.paths import (
    DATASETS_ROOT,
    FUTURES_PATH,
    PROJECT_ROOT,
    TICKHISTORY_PATH,
)


def load_trades_data(path: str) -> pl.DataFrame:
    """
    Loads trades data from LSEG TickHistory.
    Args:
        path: Path to the trades data file.
    Returns:
        pl.DataFrame: Trades data.
    """
    return (
        pl.scan_csv(TICKHISTORY_PATH / "trades" / path, schema_overrides={"Price": pl.Float64, "Volume": pl.Int64})
        .filter(pl.col("#RIC").is_in(ALL_RICS))
        .select(["#RIC", "Date-Time", "Price", "Volume", "GMT Offset"])
        .with_columns([
            # Parse datetime including nanoseconds and timezone
            pl.col("Date-Time").str.replace("Z$", "+00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z").alias("datetime"),
            # Ordering contracts
            pl.col("#RIC").str.extract(r"c(\d)$").cast(pl.UInt8).alias("order"),
            # Ensure GMT Offset is numeric for comparisons
            pl.col("GMT Offset").cast(pl.Float32).alias("GMT Offset"),
            # Volume assumption
            pl.when((pl.col("Volume") <= 0.0) | pl.col("Volume").is_null() | pl.col("Volume").is_nan()).then(1.0).otherwise(pl.col("Volume")).alias("Volume"),
            # Trade price adjustment because 0.0 price doesnt make sense
            pl.when(pl.col("Price") == 0.0).then(pl.lit(None)).otherwise(pl.col("Price")).alias("Price")
        ])
        .unique(keep='first')
        .with_columns([
            (pl.col("datetime") + pl.duration(hours=pl.col("GMT Offset"))).alias("local_datetime"),
            pl.col("datetime").dt.convert_time_zone("America/New_York").alias("datetime_et"),
            pl.col("datetime").dt.convert_time_zone("Europe/London").alias("datetime_london")
        ])
        .filter(pl.col("Price").is_not_null())
        .collect(engine="streaming")
        .sort("local_datetime")
    )


def load_quotes_data(path: str) -> pl.DataFrame:
    """
    Load quotes data from LSEG TickHistory.
    Args:
        path: Path to the quotes data file.
    Returns:
        pl.DataFrame: Quotes data.
    """
    return (
        pl.scan_csv(TICKHISTORY_PATH / "quotes" / path, schema_overrides={"Close Bid": pl.Float64, "Close Ask": pl.Float64, "GMT Offset": pl.Float32})
        .filter((pl.col("#RIC").is_in(ALL_RICS)) & (pl.col("Type") == "Intraday 1Min"))
        .select(["#RIC", "Date-Time", "Close Bid", "Close Ask", "GMT Offset"])
        .with_columns([
            # Parse datetime including nanoseconds and timezone
            pl.col("Date-Time").str.replace("Z$", "+00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z").alias("datetime"),
            # Ordering contracts
            pl.col("#RIC").str.extract(r"c(\d)$").cast(pl.UInt8).alias("order"),
        ])
        .unique(keep='first')
        .with_columns([
            (pl.col("datetime") + pl.duration(hours=pl.col("GMT Offset"))).alias("local_datetime"),
            pl.col("datetime").dt.convert_time_zone("America/New_York").alias("datetime_et"),
            pl.col("datetime").dt.convert_time_zone("Europe/London").alias("datetime_london")
        ])
        .filter((pl.col("Close Bid").is_not_null()) & (pl.col("Close Ask").is_not_null()))
        .filter((pl.col("Close Bid") != 0.0) & (pl.col("Close Ask") != 0.0))
        .collect(engine="streaming")
        .sort("local_datetime")
    )


def load_open_prices() -> pl.DataFrame:
    """Load open prices from the LSEG Datastream."""
    cs_or_ct = "CS" if ASSET_CLASS == AssetClass.TRADITIONAL else "CT"
    return (
        pl.scan_parquet(FUTURES_PATH / f"datastream_futures_open_{cs_or_ct}.parquet")
        .filter(pl.col("date") >= pl.date(1996, 1, 4))
        .select(["clscode", "date", "open_1", "open_2", "open_3", "open_4"])
        .filter((pl.col("clscode") != 290) | (pl.col("date") < date(2003, 12, 22)))
        .collect()
    )


def load_settlement_prices() -> pl.DataFrame:
    """Load settlement prices from the LSEG Datastream; only used for LME Metals."""
    cs_or_ct = "CS" if ASSET_CLASS == AssetClass.TRADITIONAL else "CT"
    return (
        pl.scan_parquet(FUTURES_PATH / f"datastream_futures_settlement_{cs_or_ct}.parquet")
        .filter(pl.col("date") >= pl.date(1996, 1, 4))
        .select(["clscode", "date", "settlement_1", "settlement_2", "settlement_3", "settlement_4"])
        .filter((pl.col("clscode") != 290) | (pl.col("date") < date(2003, 12, 22)))
        .collect()
    )

def load_expiry_dates() -> pl.DataFrame:
    """Load expiry dates from the LSEG Datastream."""
    return (
        pl.scan_csv(FUTURES_PATH / "dsfutcontrinfo.csv", ignore_errors=True, schema_overrides={"lasttrddate": pl.Date, "startdate": pl.Date})
        .filter((pl.col("clscode") != 290) | (pl.col("lasttrddate") < date(2003, 12, 22)))
        .filter(pl.col("lasttrddate").is_not_null())
        .select(['clscode', 'lasttrddate', 'startdate'])
        .collect()
    )


def count_expiries_between(row_date: date, row_trad_last: date, expiry_list: list[date]) -> int | None:
    """Helper function used to deterine the front month contract for CT futures"""
    if row_date is None or row_trad_last is None:
        return None
    cnt = 0
    for d in expiry_list:
        if d >= row_date and d <= row_trad_last:
            cnt += 1
    return cnt


def compute_settlement_price(i: int):
    """Helper function used to compute the settlement price for each contract."""
    return (
        pl.when(pl.col(f"vwap_c{i}").is_not_null()).then(pl.col(f"vwap_c{i}"))
        .otherwise(
            pl.when((pl.col(f"lasttrdprice_c{i}") >= pl.col(f"last_bid_c{i}")) & (pl.col(f"lasttrdprice_c{i}") <= pl.col(f"last_ask_c{i}")))
            .then(pl.col(f"lasttrdprice_c{i}"))
            .otherwise(pl.coalesce(pl.col(f"midpoint_c{i}"), pl.col(f"lasttrdprice_c{i}")))
        ).alias(f"settlement_c{i}")
    )


def manual_corrections(FUTURE: Future, daily_vwaps: pl.DataFrame) -> pl.DataFrame:
    """Manual corrections for prices due to erroneous data points"""
    # Strange VWAP of ~6 on 199-04-23; I manually set it to the lasttrdprice = open price.
    if FUTURE.symbol == "KC":
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(1999, 4, 23)).then(pl.col("lasttrdprice_c1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1")
        ])
    elif FUTURE.symbol == "SB":
        daily_vwaps = daily_vwaps.with_columns([
            # Weird midpoint messes up this first one
            pl.when(pl.col("date_") == date(1996, 6, 13)).then(pl.col("lasttrdprice_c2")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
            # Odd VWAP calculation of 8 when the price is 5
            pl.when(pl.col("date_") == date(2000, 3, 14)).then(pl.col("lasttrdprice_c1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
        ])
    elif FUTURE.symbol == "HE":
        # All tick data seems wrong on this day; trading at 60 but everything is close to 5
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(2009, 6, 22)).then(pl.col("open_1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
            pl.when(pl.col("date_") == date(2009, 6, 22)).then(pl.col("open_2")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    # Spike of 20% followed by -17.5% on 1999-12-21 and 1999-12-22 immediately after front month expires
    elif FUTURE.symbol == "ZB":
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(1999, 12, 21)).then(pl.col("open_1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
        ])
    elif FUTURE.symbol == "FIE":
        # None of the data on 1998-03-24 makes sense
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(1998, 3, 24)).then(pl.col("lasttrdprice_c2")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
            pl.when(pl.col("date_") == date(1998, 3, 24)).then(pl.col("lasttrdprice_c2")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "MFS":
        # Erroneous bids in June 2015 that mess up the midprice calculation and thus prices
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_").is_between(date(2015, 6, 1), date(2015, 6, 5), closed='both')).then(pl.coalesce(pl.col("lasttrdprice_c2"), pl.col("last_ask_c2"))).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "EOE":
        # Erroneous VWAP of 0.7 when the price is trading around 548
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(1996, 4, 16)).then(pl.col("lasttrdprice_c2")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "DJ":
        # No quotes data, VWAP, or open prices, and last traded prices are inflated 10x??
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_") == date(2001, 1, 15)).then(pl.col("settlement_c1") / 10.0).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
            pl.when(pl.col("date_") == date(2001, 1, 15)).then(pl.col("settlement_c2") / 10.0).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "TF":
        daily_vwaps = daily_vwaps.with_columns([
            pl.col("open_1").alias("settlement_c1"),
            pl.col("open_2").alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "FTI":
        daily_vwaps = daily_vwaps.with_columns([
            # Erroneous VWAP calculation
            pl.when(pl.col("date_") == date(1997, 6, 26)).then(pl.col("lasttrdprice_c1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
            # Contract changed causing price to go from 1200 to 500; TODO: if implementing features, fix this better
            # pl.when((pl.col("date_") == date(1999, 1, 4))).then(None).otherwise(pl.col("ret1_adjusted")).alias("ret1_adjusted"),
        ])
    elif FUTURE.symbol == "WDC":
        daily_vwaps = daily_vwaps.filter(pl.col("settlement_c1") > 1).filter(pl.col("settlement_c2") > 1)
    elif FUTURE.symbol == "KRW":
        # Bad settlement prints (the vwap is corrupt; the TickHistory last trade is clean).
        # Front is c1 on 2006-09-20 & 2007-03-27, c2 on 2014-02-14. Values verified against
        # the ET debug table and WM/Reuters 4pm spot (match <0.5%); the raw spikes have no
        # counterpart in spot -> data errors, not macro. See em-fx-scrubbing design.
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_").is_in([date(2006, 9, 20), date(2007, 3, 27)]))
              .then(pl.col("lasttrdprice_c1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
            pl.when(pl.col("date_") == date(2014, 2, 14))
              .then(pl.col("lasttrdprice_c2")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
        ])
    elif FUTURE.symbol == "PLN":
        # Front-contract (c1) bad print in BOTH settlement and lasttrdprice; the independent
        # Datastream open_1 is the clean value (same tactic as HE "all tick data wrong"/ZB/TF).
        # The 2008-11-24 print also causes the flagged +32% rebound on 2008-11-25. Verified vs
        # WM/Reuters 4pm spot (open_1 within <0.5%; spot moved -0.06% / +3.8%, not -36% / -19%).
        daily_vwaps = daily_vwaps.with_columns([
            pl.when(pl.col("date_").is_in([date(2008, 7, 16), date(2008, 11, 24)]))
              .then(pl.col("open_1")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
        ])
    return daily_vwaps


def apply_unit_transforms(FUTURE: Future, daily_vwaps: pl.DataFrame) -> pl.DataFrame:
    """Post-adjustment units/orientation transforms for specific FX futures. MUST run
    after FUTURE.adjustments (6J is non-commutative with the rescale). Migrated verbatim
    from the former inline block: 6J units (<0.1 -> x100), 6Z orientation (>10 -> 100 - x)."""
    cols = ["front_month_settlement", "settlement_c1", "settlement_c2", "settlement_c3", "settlement_c4"]
    if FUTURE.symbol == "6J":
        return daily_vwaps.with_columns(
            [pl.when(pl.col(c) < 0.1).then(pl.col(c) * 100).otherwise(pl.col(c)).alias(c) for c in cols]
        )
    if FUTURE.symbol == "6Z":
        return daily_vwaps.with_columns(
            [pl.when(pl.col(c) > 10).then(100 - pl.col(c)).otherwise(pl.col(c)).alias(c) for c in cols]
        )
    return daily_vwaps


def process_future(FUTURE: Future, sync_target: str = "et") -> pl.DataFrame:
    """Process a future's data and return the returns series."""
    # Self-provision the debug output dirs (regenerable; moved out by the reorg),
    # mirroring futures.py — the write_csv calls below assume they exist.
    os.makedirs(TICKHISTORY_PATH / "debug" / "dates", exist_ok=True)
    os.makedirs(TICKHISTORY_PATH / "debug" / "tables", exist_ok=True)

    if sync_target == "london":
        DATETIME_COLUMN = "datetime_london"
        SETTLEMENT_START = time(16, 0)
        SETTLEMENT_END = time(16, 1)
    elif sync_target == "et":
        DATETIME_COLUMN = "datetime_et"
        SETTLEMENT_START = time(9, 30)
        SETTLEMENT_END = time(9, 31)
    else:
        DATETIME_COLUMN = "local_datetime"
        SETTLEMENT_START = FUTURE.settlement_start
        SETTLEMENT_END = FUTURE.settlement_end

    # Mapping one future on TickHistory to multiple futures (clscodes) in Datastream
    if FUTURE.symbol == "G":
        clscodes = [1181, 1176]
    elif FUTURE.symbol == "BRN":
        clscodes = [1175, 1970]
    elif FUTURE.symbol == "160120006":
        clscodes = [3729, 1051]
    elif FUTURE.symbol == "164120019":
        clscodes = [1622, 4648]
    # TODO: check if we can get earlier data for this
    # elif FUTURE.symbol == "MFS":
    #     clscodes = [244, 3930]
    elif FUTURE.symbol == "RL":
        clscodes = [290, 3590]
    else:
        clscodes = [FUTURE.clscode]

    assert FUTURE.ric is not None
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
    dates.write_csv(TICKHISTORY_PATH / "debug" / "dates" / f"{FUTURE.symbol}.csv")
    time_filtered_data = trades_data.filter(
        (pl.col(DATETIME_COLUMN).dt.time() >= SETTLEMENT_START) &
        (pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_END)
    )

    # Compute VWAP in the designated settlement interval: 9:30-9:31 AM EST
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
    )

    fallback_ok = (pl.col("fallback_date") - pl.duration(days=1) == pl.col("prev_trading_day"))
    if FUTURE.exchange in US_EXCHANGES:
        open_prices = OPEN_PRICES.filter(pl.col("clscode").is_in(clscodes))
        lasttrdprices = lasttrdprices.join(open_prices, left_on="date_", right_on="date", how="left").sort(["date_"])
        lasttrdprices = lasttrdprices.with_columns([
            pl.when(pl.col("lasttrdprice_today").is_not_null())
            .then(pl.col("lasttrdprice_today"))
            .when(pl.col("order") == 1)
            .then(pl.coalesce(pl.col("open_1"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 2)
            .then(pl.coalesce(pl.col("open_2"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 3)
            .then(pl.coalesce(pl.col("open_3"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 4)
            .then(pl.coalesce(pl.col("open_4"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .otherwise(None)
            .alias("lasttrdprice")
        ]).select(["#RIC", "date_", "order", "lasttrdprice"]).sort(["date_", "order"])
    elif FUTURE.exchange == 9079:
        settlement_prices = SETTLEMENT_PRICES.filter(pl.col("clscode").is_in(clscodes))
        lasttrdprices = lasttrdprices.join(settlement_prices, left_on="date_", right_on="date", how="left").sort(["date_"])
        lasttrdprices = lasttrdprices.with_columns([
            pl.when(pl.col("lasttrdprice_today").is_not_null())
            .then(pl.col("lasttrdprice_today"))
            .when(pl.col("order") == 1)
            .then(pl.coalesce(pl.col("settlement_1"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 2)
            .then(pl.coalesce(pl.col("settlement_2"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 3)
            .then(pl.coalesce(pl.col("settlement_3"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .when(pl.col("order") == 4)
            .then(pl.coalesce(pl.col("settlement_4"), pl.when(fallback_ok).then(pl.col("lasttrdprice"))))
            .otherwise(None)
            .alias("lasttrdprice")
        ]).select(["#RIC", "date_", "order", "lasttrdprice"]).sort(["date_", "order"])
    else:
        lasttrdprices = lasttrdprices.with_columns([
            # Only use fallback if it's from the previous trading day
            # If today's price exists, use it; otherwise use fallback only if it's from prev trading day
            pl.when(pl.col("lasttrdprice_today").is_not_null())
            .then(pl.col("lasttrdprice_today"))
            # IMPORTANT: Reverse the previous +1 day shift to fallback_date. Only consider yesterday's last trade prices for Asian symbols
            .when(fallback_ok).then(pl.col("lasttrdprice")).otherwise(None).alias("lasttrdprice")
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
    ]).sort("date_")
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

    daily_vwaps = daily_vwaps.with_columns([
        compute_settlement_price(1),
        compute_settlement_price(2),
        compute_settlement_price(3),
        compute_settlement_price(4),
    ])

    # If a settlement price cannot be determined, for US assets, it is safe to use the open price because open prices occur on the same day.
    if FUTURE.exchange in US_EXCHANGES:
        open_prices = OPEN_PRICES.filter(pl.col("clscode").is_in(clscodes)).sort(["date"])
        daily_vwaps = daily_vwaps.join(open_prices, left_on="date_", right_on="date", how="full").with_columns(
            pl.coalesce(pl.col("date_"), pl.col("date")).alias("date_")
        ).sort(["date_"])
        daily_vwaps = daily_vwaps.with_columns([
            pl.coalesce(pl.col("settlement_c1"), pl.col("open_1")).alias("settlement_c1"),
            pl.coalesce(pl.col("settlement_c2"), pl.col("open_2")).alias("settlement_c2"),
            pl.coalesce(pl.col("settlement_c3"), pl.col("open_3")).alias("settlement_c3"),
            pl.coalesce(pl.col("settlement_c4"), pl.col("open_4")).alias("settlement_c4"),
        ])
    # For LME Metals, it is safe to use the settlement price because settlement prices 1-2 hours before 9:30am EST, so it is the most recent information.
    elif FUTURE.exchange == 9079:
        settlement_prices = SETTLEMENT_PRICES.filter(pl.col("clscode").is_in(clscodes)).sort(["date"])
        daily_vwaps = daily_vwaps.join(settlement_prices, left_on="date_", right_on="date", how="full").with_columns(
            pl.coalesce(pl.col("date_"), pl.col("date")).alias("date_")
        ).sort(["date_"])
        daily_vwaps = daily_vwaps.with_columns([
            pl.coalesce(pl.col("settlement_c1"), pl.col("settlement_1")).alias("settlement_c1"),
            pl.coalesce(pl.col("settlement_c2"), pl.col("settlement_2")).alias("settlement_c2"),
            pl.coalesce(pl.col("settlement_c3"), pl.col("settlement_3")).alias("settlement_c3"),
            pl.coalesce(pl.col("settlement_c4"), pl.col("settlement_4")).alias("settlement_c4"),
        ])

    daily_vwaps = daily_vwaps.filter(
        (pl.col("settlement_c1").is_not_null()) |
        (pl.col("settlement_c2").is_not_null()) |
        (pl.col("settlement_c3").is_not_null()) |
        (pl.col("settlement_c4").is_not_null())
    )

    expiry_dates = LSEG_DATA.filter(pl.col('clscode').is_in(clscodes)).sort(['lasttrddate']).select([pl.col('startdate'), pl.col('lasttrddate')])
    expiry_list = expiry_dates["lasttrddate"].to_list()
    daily_vwaps = daily_vwaps.join_asof(expiry_dates, left_on="date_", right_on="lasttrddate", strategy="forward").with_columns([
        pl.when(
            (pl.col("date_").dt.year() == pl.col("lasttrddate").dt.year()) & (pl.col("date_").dt.month() == pl.col("lasttrddate").dt.month())
        ).then(pl.lit(1)).otherwise(pl.lit(0)).alias("expiring_this_month"),
        pl.col("date_").dt.month().alias("current_month"),
    ])

    # Before calculating returns, remove weekends in our final returns dataset
    daily_vwaps = daily_vwaps.filter((pl.col("date_").dt.weekday() != 6) & (pl.col("date_").dt.weekday() != 7))

    # Manual corrections for prices due to erroneous data points
    daily_vwaps = manual_corrections(FUTURE, daily_vwaps)

    # Determine the front month contract
    if ASSET_CLASS == AssetClass.TRADITIONAL:
        # Use this column trad_lasttrddate for knowing when an active contract expires
        assert FUTURE.ct is not None
        trad_expiry_dates = expiry_dates.filter(pl.col("lasttrddate").dt.month().is_in(FUTURE.ct)).with_columns([
            pl.col("lasttrddate").alias("trad_lasttrddate")
        ]).select([pl.col("trad_lasttrddate")]).sort(["trad_lasttrddate"])
        daily_vwaps = daily_vwaps.join_asof(trad_expiry_dates, left_on="date_", right_on="trad_lasttrddate", strategy="forward")

        daily_vwaps = daily_vwaps.with_columns([
            # month_diff really represents the number of contracts valid between date_ and trad_lasttrddate
            pl.struct(["date_", "trad_lasttrddate"]).map_elements(
                lambda s: count_expiries_between(s["date_"], s["trad_lasttrddate"], expiry_list),
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

    # Historical price adjustment when the exchange changes contract details -> scales the price appropriately
    if FUTURE.adjustments is not None:
        for adjustment in FUTURE.adjustments:
            daily_vwaps = daily_vwaps.with_columns([
                pl.when(pl.col("front_month_settlement") > adjustment.get("threshold")).then(pl.col("front_month_settlement") / adjustment.get("divisor")).otherwise(pl.col("front_month_settlement")).alias("front_month_settlement"),
                pl.when(pl.col("settlement_c1") > adjustment.get("threshold")).then(pl.col("settlement_c1") / adjustment.get("divisor")).otherwise(pl.col("settlement_c1")).alias("settlement_c1"),
                pl.when(pl.col("settlement_c2") > adjustment.get("threshold")).then(pl.col("settlement_c2") / adjustment.get("divisor")).otherwise(pl.col("settlement_c2")).alias("settlement_c2"),
                pl.when(pl.col("settlement_c3") > adjustment.get("threshold")).then(pl.col("settlement_c3") / adjustment.get("divisor")).otherwise(pl.col("settlement_c3")).alias("settlement_c3"),
                pl.when(pl.col("settlement_c4") > adjustment.get("threshold")).then(pl.col("settlement_c4") / adjustment.get("divisor")).otherwise(pl.col("settlement_c4")).alias("settlement_c4"),
            ])
    daily_vwaps = apply_unit_transforms(FUTURE, daily_vwaps)

    # Returns ret1 without any price adjustment + other returns ret_c1 and ret_c2 which are just the returns for c1 and c2 series
    symbol_data = daily_vwaps.with_columns([
        (pl.col("front_month_settlement") / pl.col("front_month_settlement").shift(1) - 1).alias("ret1"),
        (pl.col("settlement_c1") / pl.col("settlement_c2").shift(1) - 1).alias("rollback_c2_to_c1"),
        (pl.col("settlement_c2") / pl.col("settlement_c3").shift(1) - 1).alias("rollback_c3_to_c2"),
        (pl.col("settlement_c3") / pl.col("settlement_c4").shift(1) - 1).alias("rollback_c4_to_c3"),
        (pl.col("settlement_c1") / pl.col("settlement_c1").shift(1) - 1).alias("ret_c1"),
        (pl.col("settlement_c2") / pl.col("settlement_c2").shift(1) - 1).alias("ret_c2"),
        (pl.col("settlement_c3") / pl.col("settlement_c3").shift(1) - 1).alias("ret_c3"),
        (pl.col("settlement_c4") / pl.col("settlement_c4").shift(1) - 1).alias("ret_c4"),
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
            pl.when(pl.col("shift") == 1).then(pl.coalesce(pl.col("ret_c2"))).when(
                pl.col("shift") == -1).then(pl.coalesce(pl.col("rollback_c2_to_c1"))
            ).otherwise(
                pl.when(pl.col("expiring_this_month") == 1).then(
                    pl.coalesce(pl.col("ret1"), pl.col("ret_c2"), pl.col("ret_c1"))
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

    if ASSET_CLASS != AssetClass.TRADITIONAL:
        symbol_data.write_csv(TICKHISTORY_PATH / "debug" / "tables" / f"{FUTURE.symbol}.csv")
    else:
        symbol_data.write_csv(TICKHISTORY_PATH / "debug" / "tables" / f"TRADITIONAL_{FUTURE.symbol}.csv")

    returns_series = symbol_data.select(["date_", "ret1_adjusted"]).with_columns([
        pl.lit(FUTURE.symbol).alias("symbol")
    ]).sort("date_")
    return returns_series


def main():
    returns_data = []
    for FUTURE in tqdm(CONFIG):
        print(f"Processing {FUTURE.symbol}...", flush=True)
        returns_data.append(process_future(FUTURE, sync_target=SYNC_TARGET))

    all_returns = pl.concat(returns_data, how="vertical_relaxed")
    returns_wide = (
        all_returns
        .pivot(on="symbol", values="ret1_adjusted", index="date_")
        .sort("date_")
        .with_columns(pl.col("date_").alias("date"))
    )
    if SYNC_TARGET == "london":
        output_subdir = "sync_london"
    elif SYNC_TARGET == "et":
        output_subdir = "sync"
    else:
        output_subdir = "async"

    os.makedirs(DATASETS_ROOT / f"tier{args.tier}" / output_subdir, exist_ok=True)
    returns_wide.write_csv(DATASETS_ROOT / f"tier{args.tier}" / output_subdir / f"{ASSET_CLASS.name.lower()}_daily_returns.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset_class', type=str, default='currency', help='Asset class to process')
    parser.add_argument('--tier', type=int, default=1, help='Tier of universe to process')
    parser.add_argument('--sync_target', choices=['et', 'london', 'none'], default='et', help='Timezone to sync settlement times to')
    parser.add_argument('--london_sync', action='store_true', help='Use London sync target (overrides sync_target)')
    args = parser.parse_args()

    SYNC_TARGET = "london" if args.london_sync else args.sync_target
    TIME_SYNCED = (SYNC_TARGET != "none")

    ASSET_CLASS = AssetClass(args.asset_class)
    if ASSET_CLASS is None:
        raise ValueError(f"Invalid asset class: {args.asset_class}")

    CONFIG = []
    for future in load_config(PROJECT_ROOT / f"tier{args.tier}.yaml"):
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
    # CME, CBOT, KCBT, eCBOT, NYBOT, NYM, COMEX, ICE Futures US, CBOE
    US_EXCHANGES = [9051, 1022, 2709, 9144, 1012, 9044, 9250, 9371, 9120]
    ALL_RICS = []
    for future in CONFIG:
        ALL_RICS.extend([f"{future.ric[0]}c1", f"{future.ric[0]}c2", f"{future.ric[0]}c3", f"{future.ric[0]}c4"])

    # Processing US equity indices and Non-US equities separately due to memory constraints
    filename = ASSET_CLASS.value if "equity" not in ASSET_CLASS.value else "equity"
    TRADES_DATA = load_trades_data(f"tier{args.tier}_{filename}_trades.csv")
    QUOTES_DATA = load_quotes_data(f"tier{args.tier}_{filename}_quotes.csv")
    OPEN_PRICES = load_open_prices()
    SETTLEMENT_PRICES = load_settlement_prices()
    LSEG_DATA = load_expiry_dates()

    main()

