import argparse
import copy
import os
from datetime import date, time

import polars as pl
from tqdm import tqdm

from globalmacro.pipeline.tickhistory_shards import shard_dir_name
from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass, Future
from globalmacro.utils.paths import (
    DATASETS_ROOT,
    FUTURES_PATH,
    PROJECT_ROOT,
    TICKHISTORY_PATH,
)
from globalmacro.utils.trade_presence import is_rescuable_mask, trade_counts

# clscode 290 (ER2) stopped backing RL after this handoff; its later Datastream rows are
# stale and must not enter any RL input. Every loader that reads a clscode's dates --
# settlement prices, the expiry calendar, and the expiry ladder the rescue rule rolls
# against -- applies the same cutoff so the three cannot disagree about which expiries RL
# actually had.
_STALE_HANDOFF_CLSCODE = 290
_STALE_HANDOFF_CUTOFF = date(2003, 12, 22)


def _keeps_after_stale_handoff(date_col: str) -> pl.Expr:
    """A keep-mask: True for every row except the stale post-handoff clscode-290 ones,
    comparing `date_col`. The same expression drives `.filter()` on the settlement,
    expiry-calendar, and expiry-ladder inputs so they cannot disagree about RL."""
    return (pl.col("clscode") != _STALE_HANDOFF_CLSCODE) | (pl.col(date_col) < _STALE_HANDOFF_CUTOFF)


def load_trades_data(path: str) -> pl.DataFrame:
    """
    Loads trades data from LSEG TickHistory.
    Args:
        path: Path to the trades data file.
    Returns:
        pl.DataFrame: Trades data.
    """
    return (
        pl.scan_parquet(TICKHISTORY_PATH / "trades" / shard_dir_name(path) / "*.parquet")
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
        # maintain_order=True: a plain .unique() returns its surviving rows in a different
        # physical order on every call (verified: 3 back-to-back runs on one identical shard
        # gave 3 different row orders, on both the streaming and in-memory engines -- not a
        # streaming artifact). The dedup itself never changes a value (it spans every column,
        # so which duplicate survives is irrelevant), but every downstream .sort()/.last()/
        # tie-break inherits a fresh permutation each run: .sort() is a deterministic function
        # of its input, but sorting a permuted frame on the non-unique key `local_datetime`
        # yields a permuted output, not a canonical one. Pinning the order here is what makes
        # everything downstream reproducible.
        .unique(keep='first', maintain_order=True)
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
        pl.scan_parquet(TICKHISTORY_PATH / "quotes" / shard_dir_name(path) / "*.parquet")
        .filter((pl.col("#RIC").is_in(ALL_RICS)) & (pl.col("Type") == "Intraday 1Min"))
        .select(["#RIC", "Date-Time", "Close Bid", "Close Ask", "GMT Offset"])
        .with_columns([
            # Parse datetime including nanoseconds and timezone
            pl.col("Date-Time").str.replace("Z$", "+00:00").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z").alias("datetime"),
            # Ordering contracts
            pl.col("#RIC").str.extract(r"c(\d)$").cast(pl.UInt8).alias("order"),
        ])
        # See load_trades_data above: maintain_order=True pins the row order so downstream
        # sorts and tie-breaks are reproducible. (The quotes' own last_bid/last_ask
        # aggregation, built later in process_future, is provably a no-op on current
        # data -- 0 of 801,531 groups ambiguous -- but this still keeps the loader
        # itself deterministic for any future consumer.)
        .unique(keep='first', maintain_order=True)
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
    open_cols = ["open_1", "open_2", "open_3", "open_4"]
    return (
        pl.scan_parquet(FUTURES_PATH / f"datastream_futures_open_{cs_or_ct}.parquet")
        .filter(pl.col("date") >= pl.date(1996, 1, 4))
        .select(["clscode", "date", *open_cols])
        .filter((pl.col("clscode") != 290) | (pl.col("date") < date(2003, 12, 22)))
        # A literal 0.0 is Datastream's "no trade yet" marker, not a real price -- same family
        # as the FKLI zero-settlement bug fixed in 5468d2c. Left unfiltered it can reach
        # `lasttrdprice_cN` via the `coalesce(open_N, ...)` fallback in process_future's
        # US-exchange branch (coalesce treats 0.0 as a valid non-null value), and from there
        # `compute_settlement_price` may accept it outright whenever a stale bid/ask spread
        # happens to straddle zero. Nulled per-column here, mirroring load_trades_data's own
        # zero-price handling, rather than dropping the whole row the way load_quotes_data's
        # filter does: open_1..4 are four independent contract-order prices sharing one row
        # per (clscode, date), so a row-level filter would also discard the other three
        # orders' valid prices on a date where only one order happened to print zero.
        .with_columns([pl.when(pl.col(c) == 0.0).then(None).otherwise(pl.col(c)).alias(c) for c in open_cols])
        .collect()
    )


def load_settlement_prices() -> pl.DataFrame:
    """Load settlement prices from the LSEG Datastream; only used for LME Metals."""
    cs_or_ct = "CS" if ASSET_CLASS == AssetClass.TRADITIONAL else "CT"
    settlement_cols = ["settlement_1", "settlement_2", "settlement_3", "settlement_4"]
    return (
        pl.scan_parquet(FUTURES_PATH / f"datastream_futures_settlement_{cs_or_ct}.parquet")
        .filter(pl.col("date") >= pl.date(1996, 1, 4))
        .select(["clscode", "date", *settlement_cols])
        .filter(_keeps_after_stale_handoff("date"))
        # See load_open_prices above: same zero-price trap, same per-column null (not a
        # row-drop), one column per contract order.
        .with_columns([pl.when(pl.col(c) == 0.0).then(None).otherwise(pl.col(c)).alias(c) for c in settlement_cols])
        .collect()
    )

def load_expiry_dates() -> pl.DataFrame:
    """Load expiry dates from the LSEG Datastream."""
    return (
        pl.scan_csv(FUTURES_PATH / "dsfutcontrinfo.csv", ignore_errors=True, schema_overrides={"lasttrddate": pl.Date, "startdate": pl.Date})
        .filter(_keeps_after_stale_handoff("lasttrddate"))
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


def load_expiry_ladder() -> pl.DataFrame:
    """Load the rolling expiry ladder from the LSEG Datastream settlement file --
    `lasttrddate_1..lasttrddate_5`, never `dsfutcontrinfo.csv`. `dsfutcontrinfo.csv`
    lists months a contract's cycle allows, not months the ladder actually rolled
    through; the two differ for `GC`. The rescue rule needs the ladder the tick data
    itself rolled against, not a static contract-info table.
    """
    cs_or_ct = "CS" if ASSET_CLASS == AssetClass.TRADITIONAL else "CT"
    lasttrddate_cols = [f"lasttrddate_{i}" for i in range(1, 6)]
    return (
        pl.scan_parquet(FUTURES_PATH / f"datastream_futures_settlement_{cs_or_ct}.parquet")
        .select(["clscode", *lasttrddate_cols])
        .collect()
    )


def ladder_expiries_for(clscodes: list[int], ladder: pl.DataFrame) -> list[date]:
    """The full expiry list one or more clscodes rolled through, read off the ladder's
    own rolling `lasttrddate_1..5` columns rather than a static contract-info table.
    Each column is a snapshot of the next N expiries as of some date, so the union of
    all five across every row is the complete history, not just what was upcoming on
    any single date.
    """
    rows = ladder.filter(pl.col("clscode").is_in(clscodes))
    expiries: set[date] = set()
    for i in range(1, 6):
        col = f"lasttrddate_{i}"
        # Same stale-handoff cutoff the price and expiry-calendar loaders apply: a
        # clscode-290 date from after RL moved off it is stale, and the rescue rule must
        # not roll against an expiry the price/expiry inputs already excluded. Applied
        # per ladder column because each row carries five.
        kept = rows.filter(_keeps_after_stale_handoff(col))[col].drop_nulls().to_list()
        expiries.update(kept)
    return sorted(expiries)


def build_config(tier: int, asset_class: AssetClass, config_path: str | None = None) -> list[Future]:
    """The list of `Future`s one (tier, asset class) run actually processes.

    A two-RIC future is split into a historical leg (renamed to `ric[0]`, the panel
    column becoming the RIC itself) and an active leg (`ric[1]`) -- mirroring how a
    splice's older data and current data are ingested as two separate series.
    `traditional` carries one more restriction on top of the asset-class match: only
    futures with a trading-cycle `ct` set enter it.
    """
    path = config_path if config_path is not None else str(PROJECT_ROOT / f"tier{tier}.yaml")
    config: list[Future] = []
    for future in load_config(path):
        if future.ric is not None and asset_class in future.asset_class:
            if len(future.ric) > 1:
                historical_future = copy.deepcopy(future)
                historical_future.symbol = f"{future.ric[0]}"
                historical_future.ric = [future.ric[0]]
                config.append(historical_future)

                active_future = copy.deepcopy(future)
                active_future.ric = [future.ric[1]]
                config.append(active_future)
            else:
                config.append(future)

    if asset_class == AssetClass.TRADITIONAL:
        config = list(filter(lambda f: f.ct is not None, config))
    return config


def clscodes_for(FUTURE: Future) -> list[int]:
    """The Datastream clscode(s) that back one TickHistory future.

    Most futures map one to one; a handful span an exchange or ticker change and are
    stitched from two Datastream series -- kept as one lookup so every caller that
    needs a future's clscodes (settlement prices, the expiry calendar, the expiry
    ladder) agrees on the same mapping instead of re-deriving it.
    """
    if FUTURE.symbol == "G":
        return [1181, 1176]
    if FUTURE.symbol == "BRN":
        return [1175, 1970]
    if FUTURE.symbol == "160120006":
        return [3729, 1051]
    if FUTURE.symbol == "164120019":
        return [1622, 4648]
    # TODO: check if we can get earlier data for this
    # if FUTURE.symbol == "MFS":
    #     return [244, 3930]
    if FUTURE.symbol == "RL":
        return [290, 3590]
    return [FUTURE.clscode]


def compute_vwap(time_filtered_data: pl.DataFrame, datetime_column: str) -> pl.DataFrame:
    """Settlement-window VWAP per (#RIC, date_, order), plus its day-over-day pct change.
    `time_filtered_data` is already restricted to the settlement window (9:30-9:31 AM EST
    or the London/local equivalent); `datetime_column` is whichever of `datetime_et` /
    `datetime_london` / `local_datetime` that filtering used.

    The two `.sum()`s are keyed by `.sort_by(datetime_column, Price, Volume)` rather than
    left plain: float summation is not associative, so a group's addition order (which, for
    a plain `.sum()`, follows whatever physical row/chunk order the engine happens to hand
    it) changes the result's last bit or two. This was measured as the largest single
    contributor to pipeline drift -- 73% of drifting cells, 98% of them landing on a VWAP
    date. Sorting first fixes the addition order outright: any two rows tying on all three
    keys carry an identical (Price*Volume, Volume) pair, so their relative order within the
    tie cannot change the sum.

    Invariance, as actually tested: byte-identical under a full row shuffle, and across
    POLARS_MAX_THREADS in {1, 2, 4, 8, 16, 32}, checked both in-process and via a parquet
    round-trip, at two data scales and two group densities, and in BOTH a 2-key
    `group_by(["#RIC","date_"])` shape and the shipped 3-key
    `group_by(["#RIC","date_","order"])` shape (0 of 388 groups differ, real 2014-06
    commodity shard -- see `test_vwap_is_identical_under_a_row_shuffle` and
    `test_vwap_is_identical_across_polars_max_threads` in
    tests/test_tickhistory_determinism.py). Of those two tests, it is the row-shuffle one
    that kills a revert of this fix: the plain, unfixed `.sum()` shuffles to a different
    value on 119-124 of 388 groups. The thread-count test does NOT kill a revert -- on this
    data, the plain sum in the exact shipped 3-key shape is already thread-stable at 1-32
    threads; only the 2-key shape shows thread sensitivity. (A prior version of this
    docstring reported "4 distinct hashes across 5 thread counts" for the plain sum without
    noting that this was measured on the 2-key shape, not the shipped one -- that was wrong
    and is corrected here.)

    The fix is still worth keeping even though the shipped shape isn't currently
    thread-exposed: the 2-key sensitivity tracks polars' internal chunk count, which it
    derives from the thread count, and its choice of aggregation strategy flips on
    cardinality, chunking, and frame provenance -- none of it visible at this call site.
    The shipped shape is one polars upgrade, one cardinality shift, or one grouping-key
    edit away from being exposed the same way, and nothing would announce it if it were.
    """
    return time_filtered_data.filter(
        pl.col("Price").is_not_null() & pl.col("Volume").is_not_null()
    ).group_by(["#RIC", "date_", "order"]).agg([
        (pl.col("Price") * pl.col("Volume")).sort_by(pl.col(datetime_column), pl.col("Price"), pl.col("Volume")).sum().alias("price_volume_sum"),
        pl.col("Volume").sort_by(pl.col(datetime_column), pl.col("Price"), pl.col("Volume")).sum().alias("total_volume"),
    ]).with_columns([
        (pl.col("price_volume_sum") / pl.col("total_volume")).alias("vwap")
    ]).sort(["#RIC", "date_", "order"]).with_columns([
        (pl.col("vwap").pct_change().over(["#RIC", "order"])).alias("daily_return")
    ]).select(["#RIC", "date_", "order", "vwap", "daily_return"]).sort(["date_", "order"])


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


def attach_front_slot(daily_vwaps: pl.DataFrame) -> pl.DataFrame:
    """Pick the front-month price AND record which slot it came from.

    Recorded here, not reconstructed downstream: settlement_c1 and settlement_c2 are
    numerically equal on many days, so recovering the slot by comparing values
    misidentifies them and over-nulls by roughly 8x.
    """
    chosen = pl.when(pl.col("expiring_this_month") == 1).then(pl.col("settlement_c2")).otherwise(pl.col("settlement_c1"))
    slot = pl.when(pl.col("expiring_this_month") == 1).then(pl.lit(2)).otherwise(pl.lit(1))
    daily_vwaps = daily_vwaps.with_columns([
        chosen.alias("front_month_settlement"),
        slot.cast(pl.Int8).alias("front_slot"),
    ])
    fell_back = (pl.col("expiring_this_month") == 0) & pl.col("front_month_settlement").is_null()
    return daily_vwaps.with_columns([
        pl.when(fell_back).then(pl.col("settlement_c2")).otherwise(pl.col("front_month_settlement")).alias("front_month_settlement"),
        pl.when(fell_back).then(pl.lit(2)).otherwise(pl.col("front_slot")).cast(pl.Int8).alias("front_slot"),
    ])


def attach_traditional_front_slot(daily_vwaps: pl.DataFrame) -> pl.DataFrame:
    """Pick the front-month price for a traditional-cycle contract from `front_month`
    AND record which settlement slot supplied it.

    `front_month` is supposed to land in 1..4, but nothing upstream guarantees that (a
    join miss leaves it null; an upstream bug could push it out of range), so both the
    price chain and the slot fall back to c1 together -- if only one of them fell back,
    `front_slot` would stop describing where the shipped price actually came from.
    """
    daily_vwaps = daily_vwaps.with_columns(
        pl.when(pl.col("front_month") == 1).then(pl.col("settlement_c1"))
        .when(pl.col("front_month") == 2).then(pl.col("settlement_c2"))
        .when(pl.col("front_month") == 3).then(pl.col("settlement_c3"))
        .when(pl.col("front_month") == 4).then(pl.col("settlement_c4"))
        .otherwise(pl.col("settlement_c1"))  # fallback
        .alias("front_month_settlement")
    )
    return daily_vwaps.with_columns(
        pl.when(pl.col("front_month").is_between(1, 4)).then(pl.col("front_month"))
          .otherwise(pl.lit(1)).cast(pl.Int8).alias("front_slot")
    )


def observation_flag(symbol_data: pl.DataFrame) -> pl.Series:
    """Whether a front-month price existed, independent of whether a return survived. The
    USD FX leg needs real observation dates, which a return panel cannot supply once the
    provenance guard nulls observed cells."""
    return symbol_data.select(pl.col("front_month_settlement").is_not_null().alias("observed")).to_series()


def provenance_break_mask(symbol_data: pl.DataFrame) -> pl.Series:
    """The row-level condition `apply_provenance_guard` nulls on, exposed on its own so
    the rescue rule can identify exactly the cells the guard flagged without
    re-deriving the condition a second time and risking it drifting out of sync with
    what the guard actually does. See `apply_provenance_guard` for why each clause
    is there; this only extracts the boolean, it does not change it.
    """
    front_slot_moved = (
        # A plain != is safe here, not the null-safe ne_missing this codebase
        # otherwise reaches for on a slot comparison: front_slot is populated on
        # every row this runs against, on both the traditional and non-traditional
        # branches, so the only null .shift(1) can introduce is a symbol's first
        # row -- and fill_null(False) below keeps that row from ever matching.
        pl.col("front_slot") != pl.col("front_slot").shift(1)
    )
    provenance_break = (
        (~pl.col("shift").is_in([1, -1]))
        & front_slot_moved
        & pl.col("ret1").is_not_null()
    )
    return symbol_data.select(provenance_break.fill_null(False).alias("break")).to_series()


def apply_provenance_guard(symbol_data: pl.DataFrame) -> pl.DataFrame:
    """Null a finalised ret1_adjusted cell wherever its two ends came from different
    contract slots. Applied after every coalesce that can build ret1_adjusted --
    including the edge-case rollback adjustment that runs after the main one -- so a
    cell this guard suppresses cannot be silently refilled by a later fallback.

    A row-level precondition, not an identity re-derivation. A guard that instead
    re-derived which contract ret1_adjusted's own formula would have tracked and
    compared that to the price actually used would stay silent here: the
    re-derivation runs against the same formula the backfill already passed through,
    so on a same-day substitution it agrees with itself. front_slot is recorded
    once, at the moment the price is chosen, and a later fallback cannot move it.

    Scoped away from shift in {1, -1}: those rows are rolls, where the two ends
    legitimately come from different slots -- rollback_c2_to_c1 deliberately divides
    settlement_c1(t) by settlement_c2(t-1), so a slot mismatch there is the roll
    working as intended, not a defect. Leaving the mask unscoped nulls the sync
    panels' roll returns wholesale (34,066 cells instead of 927).

    ret1 IS NOT NULL is a precondition, not a rescue for one particular row shape:
    front_slot records the slot the price chain most recently selected, and this
    clause restricts the guard to cells where ret1 is the value the coalesce actually
    shipped into ret1_adjusted, so a slot label can never be read against a price it
    does not describe. On the data this branch ships today the clause is a verified
    no-op -- the rows it excludes already carry a null ret1_adjusted, so nothing it
    protects is currently reaching a shipped cell. In particular, the expiry-month
    rows where the second slot is dark (front_slot reads 2, ret1 is null, the
    coalesce falls through to ret_c1) are already out of scope before this clause is
    ever consulted: the transition into the expiry month is a roll and is removed by
    the shift scope, and every row inside the expiry month keeps front_slot at 2 on
    both ends and is removed by the slot comparison above. The clause stays in place
    because the traditional branch's coalesce is not indexed by front_month the way
    attach_front_slot's is, so a future change to it could produce a row where
    front_slot and ret1_adjusted disagree in exactly the shape this clause guards
    against.

    Assumes symbol_data is already sorted by date_: front_slot.shift(1) walks
    physical row order, not date_ itself. The non-traditional branch guarantees this
    with an explicit .sort("date_") right before ret1_adjusted is finalised; the
    traditional branch never re-sorts after computing front_slot, so it is trusting
    whatever order upstream joins already left it in. This is not a new dependency --
    ret1 itself is a lag-1 return over the same physical order
    (front_month_settlement.shift(1)), so a frame that broke this guard's ordering
    would already have broken ret1 first.
    """
    break_ = provenance_break_mask(symbol_data)
    return symbol_data.with_columns(
        pl.when(break_).then(None).otherwise(pl.col("ret1_adjusted")).alias("ret1_adjusted")
    )


def rescue_ret1_adjusted(
    symbol_data: pl.DataFrame, ric_root: str, expiries: list[date], counts: pl.DataFrame
) -> pl.DataFrame:
    """Restore ret1_adjusted on cells `apply_provenance_guard` nulled but that were
    actually a genuine roll: slot 1 dark from a ladder expiry through the panel's
    previous row, then trading again at the flagged row.

    Recomputes which cells the guard flagged from `ret1`/`shift`/`front_slot` (never
    touched by the guard) rather than trusting a mask captured before the guard ran,
    so this function's idea of "what got nulled" cannot drift from what the guard
    actually nulled. Column order and dtype are preserved: the debug table this feeds
    depends on both staying stable.

    Only a forward slot move (front_slot 2 -> 1, the price selector rolling back into
    the nearer contract) is ever a candidate: a continuation chain never reverts to an
    earlier contract, so a backward move (1 -> 2) can only be the price selector's
    gap-fill firing when the front month failed to print -- it is a genuine
    cross-contract return by construction and `is_rescuable_mask`'s darkness reading
    is never consulted for it. Measured on the shipped panels: every backward-moved
    cell that reaches this guard has a null settlement_c1 and none sits in the
    contract's own expiry month, which is the gap-fill signature, not a roll's.
    """
    symbol_data = symbol_data.with_columns([
        provenance_break_mask(symbol_data).alias("_provenance_break"),
        pl.col("date_").shift(1).alias("_prev_row_date"),
        (
            (pl.col("front_slot") == 1) & (pl.col("front_slot").shift(1) == 2)
        ).alias("_rolled_forward"),
    ])
    flagged = symbol_data.filter(pl.col("_provenance_break")).select([
        pl.lit(ric_root).alias("symbol"),
        pl.col("date_"),
        pl.col("_prev_row_date").alias("prev_row_date"),
        pl.lit(expiries, dtype=pl.List(pl.Date)).alias("expiries"),
        pl.col("_rolled_forward"),
    ])
    rescuable = flagged["_rolled_forward"] & is_rescuable_mask(flagged, counts)
    rescued_dates = flagged.filter(rescuable)["date_"].to_list()
    symbol_data = symbol_data.with_columns(
        pl.when(pl.col("_provenance_break") & pl.col("date_").is_in(rescued_dates))
        .then(pl.col("ret1"))
        .otherwise(pl.col("ret1_adjusted"))
        .alias("ret1_adjusted")
    )
    return symbol_data.drop(["_provenance_break", "_prev_row_date", "_rolled_forward"])


def attach_trade_presence_flags(symbol_data: pl.DataFrame, ric_root: str, counts: pl.DataFrame) -> pl.DataFrame:
    """Attach c1_traded/c2_traded, the trade-presence signal repair_stale_relabel
    reads, from the same per-RIC trade-count artifact the rescue rule reads.

    Absence of a (ric, date) row in counts is the "did not trade" signal -- counts
    never emits a zero-trade row -- so this is membership in the artifact's date set
    for that RIC, not a comparison against a count value.
    """
    c1_dates = counts.filter(pl.col("ric") == f"{ric_root}c1")["date_"]
    c2_dates = counts.filter(pl.col("ric") == f"{ric_root}c2")["date_"]
    # implode(): is_in against a bare Series of the same dtype is ambiguous to polars
    # (elementwise compare vs. membership test) and deprecated; imploding it into a
    # single-list Series keeps this a membership test unambiguously.
    return symbol_data.with_columns([
        pl.col("date_").is_in(c1_dates.implode()).alias("c1_traded"),
        pl.col("date_").is_in(c2_dates.implode()).alias("c2_traded"),
    ])


def repair_stale_relabel(symbol_data: pl.DataFrame) -> pl.DataFrame:
    """Recompute a shift == -1 relabel cell to ret_c2 when settlement_c1(t) is a
    lingering stale print rather than a genuine relabel.

    rollback_c2_to_c1 divides by settlement_c1(t), which the pipeline's own
    last-trade fallback can carry forward one day past a RIC's last real print. That
    shape reads as: c1 recorded no trade on the relabel day (c1_traded is False)
    while settlement_c1 is still non-null there and goes null again the next row.
    When that happens and slot 2 actually traded (c2_traded and ret_c2 both present),
    ret_c2 -- the deferred contract's own same-contract return -- is the value a live
    relabel would have produced; substitute it in. A row failing any clause (a live
    relabel, or a relabel where slot 2 is itself dark) ships unchanged.

    Assumes symbol_data is sorted by date_, matching every other roll-branch
    computation in this module: settlement_c1.shift(-1) reads the panel's next
    physical row, not the next calendar day.
    """
    stale_c1 = (
        ~pl.col("c1_traded")
        & pl.col("settlement_c1").is_not_null()
        & pl.col("settlement_c1").shift(-1).is_null()
    )
    recoverable = stale_c1 & pl.col("c2_traded") & pl.col("ret_c2").is_not_null()
    fire = (pl.col("shift") == -1) & recoverable
    return symbol_data.with_columns(
        pl.when(fire).then(pl.col("ret_c2")).otherwise(pl.col("ret1_adjusted")).alias("ret1_adjusted")
    )


def apply_stale_relabel_repair(
    symbol_data: pl.DataFrame, ric_root: str, counts: pl.DataFrame, asset_class: AssetClass
) -> pl.DataFrame:
    """Run repair_stale_relabel, scoped to non-traditional asset classes.

    repair_stale_relabel always substitutes ret_c2 -- settlement_c2(t) over
    settlement_c2(t-1) -- for a stale relabel cell, which is the live-contract's own
    return only when the relabel roll itself is priced off settlement_c1/settlement_c2.
    That holds for every non-traditional shift == -1 roll. It does not hold for every
    traditional one: the traditional branch's shift == -1 roll is front_month-indexed
    (rollback_c2_to_c1 / rollback_c3_to_c2 / rollback_c4_to_c3), so ret_c2 matches the
    roll's own contract pair only when front_month == 1 -- for front_month in {2, 3}
    it is an unrelated contract's return, and substituting it would overwrite a correct
    rollback with a wrong one. The front_month == 1 case has not been independently
    checked against the trade-count artifact the way the non-traditional population
    has, so every traditional frame skips the repair entirely and keeps the roll
    branch's own value, right or wrong alike.
    """
    if asset_class == AssetClass.TRADITIONAL:
        return symbol_data
    return repair_stale_relabel(attach_trade_presence_flags(symbol_data, ric_root, counts)).drop(
        ["c1_traded", "c2_traded"]
    )


def substitute_backward_cross_contract_returns(symbol_data: pl.DataFrame) -> pl.DataFrame:
    """Fill a backward-flagged cell with ret_c2, the deferred contract's own
    same-contract return -- run after `rescue_ret1_adjusted`, which never restores
    this direction, so every cell this touches is still the guard's null.

    A backward-flagged cell (`front_slot` 1 -> 2) is a genuine cross-contract break:
    `apply_provenance_guard` nulled it because settlement_c1(t) was dark and the price
    chain fell back to settlement_c2(t), a different contract than settlement_c1(t-1).
    But `ret_c2` never involves settlement_c1 at all -- it is settlement_c2(t) over
    settlement_c2(t-1), the SAME contract at both ends, present whenever slot 2 traded
    on both days regardless of what slot 1 was doing. Measured against an external
    reference with magnitude-matched controls, this substitute lands closer to the
    reference than the null it replaces on the large majority of cells, and its error
    sits below the noise floor ordinary (never-flagged) cells carry -- a same-contract
    return beats no return wherever one is available.

    Never applied to a forward-flagged cell (`front_slot` 2 -> 1), rescued or not. A
    rescued forward cell is a genuine relabel where the roll moved slot 2 as well as
    slot 1, so `ret_c2` there is itself cross-contract -- substituting it would
    reintroduce the exact defect the guard removed. A forward cell the rescue left
    null stays null here too: this function's own mask (front_slot 1 -> 2) cannot
    match a forward move (front_slot 2 -> 1) in the first place, so neither forward
    shape is reachable regardless of rescue's verdict on it.

    Only fills where `ret_c2` is itself non-null -- a cell where slot 2 is also dark
    has no same-contract return to fall back to and stays null, same as before.
    """
    symbol_data = symbol_data.with_columns(
        provenance_break_mask(symbol_data).alias("_provenance_break")
    )
    backward = (
        pl.col("_provenance_break")
        & (pl.col("front_slot") == 2)
        & (pl.col("front_slot").shift(1) == 1)
    )
    symbol_data = symbol_data.with_columns(
        pl.when(backward & pl.col("ret_c2").is_not_null())
        .then(pl.col("ret_c2"))
        .otherwise(pl.col("ret1_adjusted"))
        .alias("ret1_adjusted")
    )
    return symbol_data.drop("_provenance_break")


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
    clscodes = clscodes_for(FUTURE)

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

    # Compute VWAP in the designated settlement interval: 9:30-9:31 AM EST. See compute_vwap's
    # docstring for why the sum is keyed rather than plain.
    daily_vwaps = compute_vwap(time_filtered_data, DATETIME_COLUMN)
    daily_vwaps_c1 = daily_vwaps.filter(pl.col("order") == 1).select(["date_", "vwap"]).rename({"vwap": "vwap_c1"})
    daily_vwaps_c2 = daily_vwaps.filter(pl.col("order") == 2).select(["date_", "vwap"]).rename({"vwap": "vwap_c2"})
    daily_vwaps_c3 = daily_vwaps.filter(pl.col("order") == 3).select(["date_", "vwap"]).rename({"vwap": "vwap_c3"})
    daily_vwaps_c4 = daily_vwaps.filter(pl.col("order") == 4).select(["date_", "vwap"]).rename({"vwap": "vwap_c4"})
    daily_vwaps = (dates.join(daily_vwaps_c1, on=["date_"], how="left").join(
        daily_vwaps_c2, on=["date_"], how="left").join(
        daily_vwaps_c3, on=["date_"], how="left").join(
        daily_vwaps_c4, on=["date_"], how="left")
    .sort(["date_"]))

    # Compute TODAYS last traded prices prior to the settlement start time.
    #
    # NOT given a secondary tie-break key, by measurement rather than by oversight. Both
    # this site and the one below have a non-unique sort key (many rows can share a
    # timestamp), which the shard-migration investigation flagged as a live tie-break risk
    # (up to 14,881 ambiguous groups/site over the full history). Verified directly on the
    # real 2014-06 commodity shard, once the loader's row order is pinned (the .unique(...,
    # maintain_order=True) above): the VALUE assigned to a given (#RIC, date_, order) key by
    # this exact `.sort_by(datetime).last()` is a repeatable, deterministic function of the
    # input -- identical across 20 repeated calls on the same physical frame, and identical
    # across POLARS_MAX_THREADS in {1, 2, 4, 8, 16, 32} in fresh processes. This is expected:
    # `.last()` is a selection over one group's rows, not a cross-thread reduction like the
    # VWAP sum above, so there is no partial-result merge step for thread count to disturb.
    # The *choice* the tie-break makes is still arbitrary (no more "correct" than picking the
    # other tied row), and it does change vs. today's shipped output, because it now always
    # makes the SAME arbitrary choice instead of a fresh one each run -- but that choice no
    # longer depends on anything unpinned, so no secondary key is needed for reproducibility.
    #
    # DECISION: do not add a secondary tie-break key (e.g. pl.col("Price")) at this site or
    # at the all_lasttrdprices computation below. Adding one would not be a no-op: it would
    # change WHICH tied price wins, from "last in pinned row order" (today's behavior, once
    # the loader's row order is pinned) to "highest price at the tied timestamp" -- across up
    # to 14,881 ambiguous commodity groups and 5,751 ambiguous currency groups (full-history
    # tie census). That is a value change with no analytical benefit: "last in pinned order"
    # and "highest price" are both arbitrary conventions, neither more correct than the other,
    # and the current one is already deterministic -- the only property reproducibility
    # requires. Adding a secondary key later would cost a SECOND gated production rerun, on
    # top of the one this determinism fix already requires; this decision not to is made now,
    # deliberately, while that first rerun is still unpaid, specifically to avoid paying for a
    # second one for no analytical gain. Re-verified on two shards the implementer never used
    # for the original measurement (2008-09 and 2021-03 commodity, 6 repeated calls each): 1
    # distinct value-per-key hash for both this site and the all_lasttrdprices computation
    # below, both months.
    #
    # This decision depends on the .unique(..., maintain_order=True) call above continuing
    # to pin the loader's row order; that dependency is guarded at its origin, not here --
    # test_load_trades_data_row_order_is_stable_across_repeated_calls and
    # test_load_quotes_data_row_order_is_stable_across_repeated_calls in
    # tests/test_tickhistory_determinism.py fail if that pinning regresses, which is what
    # should catch a reopening of this decision, not a change at this site.
    trades_data = trades_data.filter((pl.col("Price").is_not_null()) & (pl.col("Volume").is_not_null()))
    today_lasttrdprices = trades_data.filter(
        pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_START
    ).group_by(["#RIC", "date_", "order"]).agg([
        pl.col("Price").sort_by(pl.col(DATETIME_COLUMN)).last().alias("lasttrdprice_today")
    ]).sort(["date_"])

    # Compute last traded price for ALL days (regardless of time); does not include days where
    # there is no trade data. Same tie-break shape and same "pinning alone is sufficient"
    # finding as today_lasttrdprices above -- see that comment.
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

    # Check the last trade price against the bid-ask spread; if outside, use the midpoint.
    # Secondary tie-break key (Close Bid / Close Ask themselves) added as pure hardening for
    # a future venue with multiple bars per minute -- provably a no-op on current data: 0 of
    # 801,531 groups (full tier1_commodity_quotes history, 1996-2025) even need a tie-break,
    # let alone disagree on value; re-verified directly, not assumed. A full re-aggregation
    # of that same history with vs. without this key produced byte-identical frames. Two
    # currency shards spot-checked too (2008-09, 2021-03: 0/727 and 0/925 groups need a
    # tie-break). Free to add now because it changes nothing today; if a future tier-2 venue
    # does introduce a tie, both sorts pick the tied row with the highest value independently
    # (bid highest bid, ask highest ask) -- they are not guaranteed to come from the same
    # underlying quote row in that case, same as before this change. An alternative that
    # co-sources bid/ask from the same underlying quote row was considered and not applied.
    last_bid_ask_spread = quotes_data.filter(pl.col(DATETIME_COLUMN).dt.time() <= SETTLEMENT_END).with_columns([
        pl.col(DATETIME_COLUMN).dt.date().alias("date_")
    ]).group_by(["#RIC", "date_", "order"]).agg([
        pl.col("Close Bid").sort_by(pl.col(DATETIME_COLUMN), pl.col("Close Bid")).last().alias("last_bid"),
        pl.col("Close Ask").sort_by(pl.col(DATETIME_COLUMN), pl.col("Close Ask")).last().alias("last_ask"),
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
        ])
        daily_vwaps = attach_traditional_front_slot(daily_vwaps)
    else:
        daily_vwaps = attach_front_slot(daily_vwaps)

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

    symbol_data = apply_provenance_guard(symbol_data)

    # The rescue rule needs a tick-derived trade-count artifact that only exists for
    # the (tier, asset class, sync target) triples it has been built for -- absent for
    # every async run (sync_target == "none") and for any sync run built before its
    # own artifact. trade_counts raises FileNotFoundError rather than returning an
    # empty frame in that case; wiring it in unconditionally would abort those runs
    # instead of leaving the guard's null in place, so its absence is caught here and
    # logged loudly rather than propagated.
    try:
        counts = trade_counts(args.tier, ASSET_CLASS.value, sync_target)
    except FileNotFoundError as exc:
        print(
            f"WARNING: no trade-presence artifact for {FUTURE.symbol} "
            f"(tier {args.tier}, {ASSET_CLASS.value}, {sync_target}) -- rescuing nothing: {exc}",
            flush=True,
        )
    else:
        symbol_data = rescue_ret1_adjusted(
            symbol_data, FUTURE.ric[0], ladder_expiries_for(clscodes, EXPIRY_LADDER), counts
        )

        # A relabel roll (shift == -1) never enters apply_provenance_guard's scope,
        # so it needs its own stale-print check against the same trade-count
        # artifact rather than the guard's null -- see apply_stale_relabel_repair
        # for why that check is scoped to non-traditional asset classes only.
        symbol_data = apply_stale_relabel_repair(symbol_data, FUTURE.ric[0], counts, ASSET_CLASS)

    # Backward breaks are never rescued above, so this only ever touches cells the
    # guard nulled and rescue left untouched.
    symbol_data = substitute_backward_cross_contract_returns(symbol_data)

    if ASSET_CLASS != AssetClass.TRADITIONAL:
        symbol_data.write_csv(TICKHISTORY_PATH / "debug" / "tables" / f"{FUTURE.symbol}.csv")
    else:
        symbol_data.write_csv(TICKHISTORY_PATH / "debug" / "tables" / f"TRADITIONAL_{FUTURE.symbol}.csv")

    returns_series = symbol_data.select(["date_", "ret1_adjusted"]).with_columns([
        pl.lit(FUTURE.symbol).alias("symbol"),
        observation_flag(symbol_data).alias("observed"),
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

    observations_dir = TICKHISTORY_PATH / "observations"
    os.makedirs(observations_dir, exist_ok=True)
    all_returns.select(pl.col("symbol"), pl.col("date_").alias("date"), pl.col("observed")).write_parquet(
        observations_dir / f"tier{args.tier}_{ASSET_CLASS.value}_{SYNC_TARGET}.parquet"
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

    CONFIG = build_config(args.tier, ASSET_CLASS)

    CLSCODES = [f.clscode for f in CONFIG]
    CALCSERIESNAMES = [f.calcseriesname for f in CONFIG]
    DATASET_START_DATE = pl.date(1996, 1, 1)
    DATASET_END_DATE = pl.date(2025, 6, 11)
    # CME, CBOT, KCBT, eCBOT, NYBOT, NYM, COMEX, ICE Futures US, CBOE
    US_EXCHANGES = [9051, 1022, 2709, 9144, 1012, 9044, 9250, 9371, 9120]
    ALL_RICS = []
    for future in CONFIG:
        assert future.ric is not None  # build_config only keeps futures with a ric
        ALL_RICS.extend([f"{future.ric[0]}c1", f"{future.ric[0]}c2", f"{future.ric[0]}c3", f"{future.ric[0]}c4"])

    # Processing US equity indices and Non-US equities separately due to memory constraints
    filename = ASSET_CLASS.value if "equity" not in ASSET_CLASS.value else "equity"
    TRADES_DATA = load_trades_data(f"tier{args.tier}_{filename}_trades.csv")
    QUOTES_DATA = load_quotes_data(f"tier{args.tier}_{filename}_quotes.csv")
    OPEN_PRICES = load_open_prices()
    SETTLEMENT_PRICES = load_settlement_prices()
    LSEG_DATA = load_expiry_dates()
    EXPIRY_LADDER = load_expiry_ladder()

    main()

