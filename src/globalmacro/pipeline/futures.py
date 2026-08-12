import argparse
import logging
import os
from datetime import date

import polars as pl

from globalmacro.utils.characteristics import (
    calc_price_with_roll,
    calc_returns_until_expiry,
    calc_returns_with_price_adj_and_roll,
    calc_total_returns_with_roll,
    contract_identity_exprs,
    zero_price_denominator,
)
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import (
    CHARACTERISTICS_ROOT,
    DATASETS_ROOT,
    FUTURES_PATH,
    PROJECT_ROOT,
)
from globalmacro.utils.splice import SPLICING_MAP

logger = logging.getLogger(__name__)


def splice_active_inactive_series(
    contr_data: pl.DataFrame,
    symbols: list[str],
    active: str,
    inactive: str
) -> pl.DataFrame:
    """
    Splicing the active and inactive series together.

    Args:
        contr_data: The contr_data dataframe containing the contract data
        symbols: The list of symbols in the dataset
        active: The active symbol
        inactive: The inactive symbol

    Returns:
        The contr_data dataframe with the active and inactive series spliced together, joined by the same symbol
    """
    if active not in symbols or inactive not in symbols:
        logger.info(
            "Skipping splicing %s to %s because one of the symbols is not in the dataset",
            active,
            inactive,
        )
        return contr_data

    logger.info("Splicing %s to %s", inactive, active)
    # Find the first date that the active contract is available
    active_data = contr_data.filter(pl.col('symbol') == active)
    inactive_data = contr_data.filter(pl.col('symbol') == inactive)
    active_min_date = active_data.select(pl.col('date').min()).item()
    active_max_date = active_data.select(pl.col('date').max()).item()
    inactive_max_date = inactive_data.select(pl.col('date').max()).item()

    logger.info("Active contract %s starts on: %s", active, active_min_date)
    logger.info("Inactive contract %s ends on: %s", inactive, inactive_max_date)

    # Find the date when active volume first exceeds inactive volume
    overlap_start = active_min_date
    overlap_end = min(inactive_max_date, active_max_date)

    if overlap_start <= overlap_end:
        # Get volume data for both contracts during overlap period
        active_overlap = active_data.filter(
            (pl.col('date') >= overlap_start) & (pl.col('date') <= overlap_end)
        ).select(['date', 'volume']).rename({'volume': 'active_volume'})

        inactive_overlap = inactive_data.filter(
            (pl.col('date') >= overlap_start) & (pl.col('date') <= overlap_end)
        ).select(['date', 'volume']).rename({'volume': 'inactive_volume'})

        # Join volume data on date
        volume_comparison = active_overlap.join(inactive_overlap, on='date', how='inner')

        # Find first date where active volume > inactive volume
        volume_crossover = volume_comparison.filter(
            pl.col('active_volume') > pl.col('inactive_volume')
        ).sort('date')

        if volume_crossover.height > 0:
            stitch_date = volume_crossover.select('date').head(1).item()
            logger.info(
                "Volume crossover found: active %s volume exceeded inactive %s volume on %s",
                active,
                inactive,
                stitch_date,
            )
        else:
            stitch_date = max(overlap_end, active_min_date)
            logger.warning(
                "No volume crossover found for %s/%s. Using fallback date: %s",
                active,
                inactive,
                stitch_date,
            )
    else:
        stitch_date = active_min_date
        logger.warning(
            "No overlap period found for %s/%s. Using active start date: %s",
            active,
            inactive,
            stitch_date,
        )

    # Check for reasonable gap
    gap_days = (stitch_date - inactive_max_date).days if inactive_max_date < stitch_date else 0
    if gap_days > 10:
        logger.warning(
            "Gap of %s days is too large. Skipping stitching for %s/%s",
            gap_days,
            active,
            inactive,
        )
        return contr_data

    # Get inactive data before stitch date
    inactive_before_stitch = inactive_data.filter(pl.col('date') < stitch_date).with_columns([
        pl.lit(active).alias('symbol')
    ])

    # Get active data from stitch date onwards
    active_from_stitch = active_data.filter(pl.col('date') >= stitch_date)

    # Remove original active and inactive entries from dataset FIRST
    contr_data = contr_data.filter(
        pl.col('symbol').is_null() |
        ((pl.col('symbol') != active) & (pl.col('symbol') != inactive))
    )

    combined_active_series = pl.concat([inactive_before_stitch, active_from_stitch]).sort('date')
    contr_data = pl.concat([contr_data, combined_active_series]).sort(['symbol', 'date'])

    logger.info(
        "Combined series for active %s: %s total observations",
        active,
        combined_active_series.height,
    )
    logger.info(
        "  - Inactive data (before %s): %s observations",
        stitch_date,
        inactive_before_stitch.height,
    )
    logger.info(
        "  - Active data (from %s): %s observations",
        stitch_date,
        active_from_stitch.height,
    )
    return contr_data


def apply_finalised_return_guard(wide_contr_data: pl.DataFrame) -> pl.DataFrame:
    """Null a finalised ret_i cell wherever its numerator and denominator contracts differ.

    Part two of a two-part fix: `calc_returns_until_expiry`'s guard (characteristics.py)
    catches this on ret_temp_i, but the coalesce that runs before this function backfills a
    null ret_1 from ret_2 -- the NEXT contract's return -- so guarding ret_temp_i alone
    leaves 366 of 1,074 defective cells alive wearing a different contract's number. Apply
    the same test to the FINALISED series.

    `.fill_null(False)` keeps this null-safe in the strict direction: a cell is nulled only
    where the two contracts are PROVABLY different. Where the denominator contract is
    unknown the value is left alone and reported by cross_contract_cells instead -- nulling
    on unprovable provenance would remove 2,156 cells rather than 1,074.
    """
    for _i in (1, 2):
        _num, _den = contract_identity_exprs(_i)
        wide_contr_data = wide_contr_data.with_columns(
            pl.when((_num != _den).fill_null(False)).then(None)
            .otherwise(pl.col(f"ret_{_i}")).alias(f"ret_{_i}")
        )
    return wide_contr_data


def apply_zero_price_denominator_guard(wide_contr_data: pl.DataFrame, price_type: str) -> pl.DataFrame:
    """Null a finalised ret_i cell wherever its OWN pre-backfill slot divided by a zero price.

    calc_returns_until_expiry already nulls ret_temp_i at the zero-price division itself
    (characteristics.py), but the null-coalesce two steps above this call in main() backfills
    a null ret_1 from ret_2 -- the NEXT contract's return -- whenever ret_2 happens to be
    non-null. apply_finalised_return_guard cannot catch what that backfill leaves behind:
    contract_identity_exprs judges the backfilled cell by which contract ret_i's OWN formula
    would have tracked, and for a same-day roll that is exactly the backfill source's
    contract too, so num == den and that guard stays silent -- even though the value stored
    is a DIFFERENT contract's return standing in for one that was genuinely undefined.

    Measured: without this guard, FKLI 2025-10-01's ret_1 does not ship as `inf` -- it ships
    as 0.007783, ret_2's value (a different, later-dated contract), because ret_temp_2 being
    null lets the coalesce fall through to ret_2, and apply_finalised_return_guard sees
    num == den (both lasttrddate_2's contract) and leaves it alone. That is a fabricated
    number wearing a plausible face, not a fix. This guard re-derives zero_price_denominator
    fresh from the base columns -- not from ret_temp_i, which the coalesce has already moved
    past -- for the SAME slot calc_returns_with_price_adj_and_roll selected for ret_i (its
    exp_1 branching), so it fires regardless of what the backfill did afterwards.
    """
    for _i in (1, 2):
        _own_zero_denom = (
            pl.when(pl.col('exp_1') == 1)
            .then(zero_price_denominator(_i + 1, price_type))
            .otherwise(zero_price_denominator(_i, price_type))
        ).over('clscode').fill_null(False)
        wide_contr_data = wide_contr_data.with_columns(
            pl.when(_own_zero_denom).then(None)
            .otherwise(pl.col(f"ret_{_i}")).alias(f"ret_{_i}")
        )
    return wide_contr_data


_BAD_SLOT2_MOVE = 0.30


def apply_bad_slot2_print_guard(wide_contr_data: pl.DataFrame) -> pl.DataFrame:
    """Substitute the traded front-month return for a zero-volume bad slot-2 print in the
    expiry roll, on the FINALISED ret_1 -- after the coalesce, mirroring the two finalised
    guards beside it: guarding ret_temp_2 upstream is backfilled away.

    A bad print (slot 2 recorded no trade and moved more than 30%) poisons two finalised
    cells: the day it is the numerator and the next day where it is the denominator. On a
    poisoned cell that used slot 2 (exp_1 == 1), use ret_temp_1 (the front month) when the
    front actually traded (volume_1 > 0), else null. Zero volume plus the traded front
    disagreeing is the discriminator.

    The bad print itself must fall on the roll (exp_1 == 1): a zero-volume spike while
    exp_1 == 0 is a settlement spike outside the roll, a different defect this guard doesn't
    cover. Without that restriction, a spike-and-revert straddling the roll boundary gets
    half-corrected -- the pre-roll spike ships untouched (fire never reaches it) while the
    revert that cancels it gets nulled or substituted away, turning a self-cancelling pair
    into a one-sided error.

    Out of scope, currently absent: the {t, t+1} pairing assumes the slot layout at t+1
    is the same one that produced the bad print at t. If the front contract itself rolls
    between t and t+1, that assumption can fail two ways -- ret_temp_1(t+1)'s own roll
    branch divides by yesterday's slot-2 price, so a roll landing exp_1 == 0 at t+1 can
    carry the bad print into a cell this guard never fires on (fire requires exp_1 == 1);
    or, if exp_1(t+1) == 1 (a serial expiry month), the guard does fire but its own
    substitute (ret_temp_1(t+1)) is the same poisoned roll-branch ratio, injecting the bad
    print rather than removing it. Neither shape is guarded against here; on the
    configured universe today, no bad print's following row is itself a roll, so both are
    measured absent, not merely unlikely.
    """
    bad = (
        (pl.col("exp_1") == 1)
        & (pl.col("volume_2") == 0)
        & (pl.col("ret_temp_2").abs() > _BAD_SLOT2_MOVE)
    ).fill_null(False)
    poisoned = (bad | bad.shift(1).over("clscode")).fill_null(False)
    substitute = pl.when(pl.col("volume_1") > 0).then(pl.col("ret_temp_1")).otherwise(None)
    fire = poisoned & (pl.col("exp_1") == 1)
    return wide_contr_data.with_columns(
        pl.when(fire).then(substitute).otherwise(pl.col("ret_1")).alias("ret_1"),
    )


def select_top_contracts_by_volume(contr_data: pl.DataFrame) -> pl.DataFrame:
    """Pick one row per (clscode, date_, order) slot: the highest-volume contract sharing
    that expiry rank on that day (`order` = the dense rank of distinct lasttrddate within
    the class-day, i.e. front month = 1, second month = 2, ...). Only order <= 5 (the top
    five expiries) survives the filter below.

    `futcode` (a per-contract identifier, present via the dsfutcontrinfo join) is a secondary
    sort key so the choice is a function of the data alone. Without it, `.sort_by("volume")
    .first()` has no tie-break whenever two contracts in the same slot report the same
    volume -- polars sorts are not stable by default, so which row wins would depend on the
    DataFrame's incoming row order, which is not itself guaranteed stable across runs (see
    the standing pipeline-reproducibility finding). Measured on the configured universe:
    337 (clscode, date_, order) groups have more than one row sharing the max volume (334
    with a genuinely different settlement across the tied rows, 3 value-duplicates) --
    ATX 215, 6Z 64, IR 57, PSI 1, MD 0. (MD's 18 pre-splice ties are consumed by the
    EMD <- MD splice before this function ever sees them; they are not part of the 337.)
    futcode is unique within every one of the 337, so it fully resolves the tie.

    Downstream effect for the gated rerun: adding this tie-break moves 325 cells across the
    8 futures-produced CSVs versus the order-dependent selection previously shipped --
    tier1 ret_1 18 (CS) / 19 (CT), tier2 ret_1 29 / 29, tier1 ret_2 47 (CS) / 53 (CT), tier2
    ret_2 63 / 67. Zero null<->value flips, no column-count movement, no first/last-date
    movement, so a rerun gate's count and date checks will not fire on this change -- only
    a cell-level (--cells) comparison sees it. ATX's moved cells surface under the
    post-splice `FATX` column, not `ATX`. `main()` feeds daily_ret_1_{CS,CT} into
    build.py's `load_async_dataset`, so this also carries into async_daily/async_monthly
    and their _usd siblings, for at most 58 symbol-days.
    """
    return (
        contr_data.sort(['clscode', 'date_', 'lasttrddate'])
        .with_columns(
            pl.col("lasttrddate").rank("dense").over(['clscode', 'date_']).alias("order")
        )
        .group_by(["clscode", "date_", "order"])
        .agg(pl.all().sort_by(["volume", "futcode"], descending=[True, False], nulls_last=True).first())
    ).filter(pl.col("order") <= 5)


def main():
    info_data = (
        dsfutcontr
        .join(dsfutclass.select(['contrcode', 'clscode', 'symbol', 'type']), on=['contrcode'], how='left')
        .filter(pl.col('clscode').is_not_null())
    )
    info_data = info_data.join(dsfutcontrinfo, on=['contrcode', 'clscode'], how='left').filter(pl.col('futcode').is_not_null())

    # Adding trading months
    info_data = (
        info_data
        .join(dsfuttrdcycle.group_by("clscode").agg(pl.col("trdmth").str.join(",")),
            on=['clscode'],
            how='left'
        )
    )

    # Adding contract values
    contr_data = (
        info_data
        .join(dsfutcontrval, on=['futcode'], how='left')
        .filter(pl.col('lasttrddate').is_not_null())
    )

    # Implement the CT option if the futures series has a CT option
    if CT == "CT":
        clscode_to_ct = {f.clscode: f.ct for f in futures if f.ct is not None}
        logger.info("Clscodes with CT cycles: %s", clscode_to_ct)
        for clscode, allowed_months in clscode_to_ct.items():
            # Filter the data to only include contracts with expiry months in the allowed list
            mask = (pl.col('clscode') == clscode) & (pl.col('lasttrddate').dt.month().is_in(allowed_months))
            contr_data = contr_data.filter(
                (pl.col('clscode') != clscode) | mask
            )

    # Adding month, binary for expiry month columns
    contr_data = contr_data.with_columns(
        (contr_data["date_"].dt.year() * 100 + contr_data["date_"].dt.month()).alias("month")
    )

    contr_data = contr_data.with_columns(
        ((contr_data["lasttrddate"].dt.year() * 100 + contr_data["lasttrddate"].dt.month()) == contr_data["month"])
        .cast(pl.Int8)  # makes it 1/0
        .alias("exp")
    )

    contr_data = contr_data.with_columns(
        contr_data["exp"].fill_null(0).alias("exp")
    )

    # Identifying the date of roll
    contr_data = contr_data.sort(['clscode', 'lasttrddate', 'date_']).with_columns(pl.col('exp').diff().over(['clscode', 'lasttrddate']).alias('switch')).with_columns(
        pl.when(pl.col('switch') == 1).then(pl.lit(1)).otherwise(pl.lit(0)).alias('switch')
    )

    # Add daystomaturity column which is the difference between the lasttrddate and the date_
    # filter out negative daystomaturity
    contr_data = (
        contr_data
        .with_columns([
            (pl.col("lasttrddate") - pl.col("date_")).dt.total_days().alias("daystomaturity")
        ])
        .filter(pl.col("daystomaturity") >= 0)
        .filter(pl.col("daystomaturity").is_not_null())
        .with_columns(
            pl.col("date_").alias("date")
        )
    )

    contr_data = (
        contr_data
        # Gas oil in the active contract has a gap in the data; use Gas oil from IPE and splice
        .filter((pl.col("clscode") != 1176) | (pl.col("date") > pl.date(2003, 1, 1)))
        # CAC 40 Franc-denominated to CAC 40 Euro-demoinated. The franc-demoninated is missing data from 91 to 95, so start at 95
        .filter((pl.col("clscode") != 498) | (pl.col("date") > pl.date(1995, 1, 1)))
    )
    for active, inactive in SPLICING_MAP.items():
        contr_data = splice_active_inactive_series(contr_data, symbols, active, inactive)


    """
    Manual fixes for erroneous data points in the data:
    1. Clscode 1025 - Norwegian Krone to USD
    - Around 2003-06-16, erroneous data points where the price is ~99 -> TODO: 100 - price
    2. Clscode 1065 - Kospi 200 Index Future
    - Around 2013-2014, erroneous data points with a common last trade date of 2015-12-10
    where the price is ~800
    3. Clscode 3829 - OMX30 Index Future
    - Around 2006-2007, erroneous data points with a common last trade date of 2006-06-16
    where the price is ~97, 98
    4. Clscode 2181 - South African Rand Future
    - There are random days where contracts disappear
    5. Clscode 3388 - Czech Koruna Future. On 2025-07-27 the price is 41088 but usally its 0.04
    """
    contr_data = (
        contr_data
        .with_columns(
            pl.when((pl.col('clscode') == 3388) & (pl.col(PRICE_TYPE) > 10_000))
            .then(pl.col(PRICE_TYPE) / 1_000_000)
            .otherwise(pl.col(PRICE_TYPE))
            .alias(PRICE_TYPE),
        )
        .with_columns(
            pl.when((pl.col('clscode') == 2108) & (pl.col(PRICE_TYPE) > 10_000))
            .then(pl.col(PRICE_TYPE) / 100_000)
            .otherwise(pl.col(PRICE_TYPE))
            .alias(PRICE_TYPE),
        )
        .filter((pl.col('clscode') != 1025) | (pl.col(PRICE_TYPE) < 1))
        .filter((pl.col('clscode') != 1065) | (pl.col(PRICE_TYPE) < 800))
        .filter((pl.col('clscode') != 3829) | (pl.col(PRICE_TYPE) > 100))
        .filter((pl.col('clscode') != 2181) | (pl.col("daystomaturity") < 1000))
        .filter((pl.col('clscode') != 259) | (pl.col(PRICE_TYPE) > 1))
    )

    #ordering; if duplicate, keep row with the highest volume, tie-broken deterministically by futcode
    contr_data = select_top_contracts_by_volume(contr_data)

    contr_data = contr_data.unique(keep='first').sort(['clscode', 'date_'])

    # Important: this price is in its local currency
    contr_data = (
        contr_data
        .with_columns(pl.col(PRICE_TYPE).cast(pl.Float64))
    )

    wide_contr_data = contr_data.pivot(
        index=["clscode", "symbol", "type", "contrcode", "date"],
        on="order",
        values=[PRICE_TYPE, "volume", "lasttrddate", "daystomaturity", "exp", "switch"]
    )
    wide_contr_data = (
        wide_contr_data
        .select(sorted(wide_contr_data.columns))
        .filter(pl.col("daystomaturity_1").is_not_null())
        .sort(['clscode', 'date'])
    )

    wide_contr_data = wide_contr_data.filter(
        (pl.col("clscode") != 1602) | (
            (pl.col('volume_1').is_not_null() & (pl.col('volume_1') > 0)) &
            (pl.col('volume_2').is_not_null() & (pl.col('volume_2') > 0))
        )
    )

    def flatten(xss):
        return [x for xs in xss for x in xs]

    calc_price_with_roll_exprs = [
        calc_price_with_roll(1, PRICE_TYPE),
        calc_price_with_roll(2, PRICE_TYPE),
        calc_price_with_roll(3, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_price_with_roll_exprs)
    )

    calc_returns_until_expiry_exprs = [
        calc_returns_until_expiry(1, PRICE_TYPE),
        calc_returns_until_expiry(2, PRICE_TYPE),
        calc_returns_until_expiry(3, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_returns_until_expiry_exprs)
    )

    calc_returns_with_price_adj_and_roll_exprs = [
        calc_returns_with_price_adj_and_roll(1),
        calc_returns_with_price_adj_and_roll(2),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_returns_with_price_adj_and_roll_exprs)
    )

    wide_contr_data = (
        wide_contr_data
        .with_columns(pl.coalesce(pl.col("ret_2"), pl.col("ret_temp_2"), pl.col("ret_1"), pl.col("ret_temp_1")).alias("ret_2"))
        .with_columns(pl.coalesce(pl.col("ret_1"), pl.col("ret_2")))
    )

    wide_contr_data = apply_finalised_return_guard(wide_contr_data)
    wide_contr_data = apply_zero_price_denominator_guard(wide_contr_data, PRICE_TYPE)
    wide_contr_data = apply_bad_slot2_print_guard(wide_contr_data)

    os.makedirs(FUTURES_PATH / "debug" / "tables", exist_ok=True)
    os.makedirs(FUTURES_PATH / "debug" / "dates", exist_ok=True)
    if PRICE_TYPE == "settlement" and CT == "CS":
        symbol_to_clscode_map = {f.symbol: f.clscode for f in futures}
        for symbol, clscode in symbol_to_clscode_map.items():
            wide_contr_data.filter(pl.col('clscode') == clscode).sort('date').write_csv(FUTURES_PATH / "debug" / "tables" / f"{symbol}.csv")
            wide_contr_data.filter(pl.col('clscode') == clscode).select('date').sort('date').write_csv(FUTURES_PATH / "debug" / "dates" / f"{symbol}.csv")

    calc_total_returns_with_roll_exprs = [
        calc_total_returns_with_roll(1, PRICE_TYPE),
        calc_total_returns_with_roll(2, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_total_returns_with_roll_exprs)
    )

    # Save a parquet file for comparison with Datastream continuous series and for us in tickhistory script
    wide_contr_data.write_parquet(FUTURES_PATH / f"datastream_futures_{PRICE_TYPE}_{CT}.parquet")

    if PRICE_TYPE == "settlement":
        # Create the raw Datastream FuturesPanel data for settlement prices only to be used in creating characteristics panels
        futures_panel = (
            wide_contr_data
            .filter(pl.col('clscode').is_in(clscodes))
            .select([
                "date",
                "symbol",
                "clscode",
                pl.when(pl.col("exp_1") == 1)
                .then(pl.col("lasttrddate_2"))
                .otherwise(pl.col("lasttrddate_1"))
                .alias("lasttrddate1"),
                pl.when(pl.col("exp_1") == 1)
                .then(pl.col("lasttrddate_3"))
                .otherwise(pl.col("lasttrddate_2"))
                .alias("lasttrddate2"),
                pl.col("adj_settlement_1").alias("prc1"),
                pl.col("adj_settlement_2").alias("prc2"),
                pl.col("ret_1").alias("ret1"),
                pl.col("ret_2").alias("ret2"),
            ])
            .sort(["date", "symbol", "clscode"])
        )
        futures_panel.write_csv(CHARACTERISTICS_ROOT / f"raw_datastream_futures_panel_{CT}.csv")

    # Save the raw returns datasets to be processed in the main script
    wide_contr_data = wide_contr_data.filter(pl.col('clscode').is_in(clscodes))
    tier1_symbols_filtered = list(set(tier1_symbols).intersection(set(wide_contr_data['symbol'].unique())))

    def save_returns(returns_variable: str, tier: int = 2):
        # Single local-currency return series (ret_1/ret_2), original filenames.
        out_dir = DATASETS_ROOT / f"tier{tier}" / "async"
        returns_data = wide_contr_data.pivot(on="symbol", values=returns_variable, index="date").sort("date")
        columns = (
            sorted(tier1_symbols_filtered)
            if tier == 1
            else sorted([col for col in returns_data.columns if col != "date"])
        )
        returns_data = returns_data.select(["date"] + columns)
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"daily_{returns_variable}_{CT}.csv"
        returns_data.write_csv(out_path)
        logger.info("Returns data saved to %s", out_path)

    if PRICE_TYPE == "settlement":
        logger.info("Saving datasets for Settlement prices")
        for tier in [1, 2]:
            save_returns("ret_1", tier=tier)   # local currency
            save_returns("ret_2", tier=tier)   # local currency


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--price_type', type=str, default='settlement', choices=['settlement', 'open'], help='Price type to use')
    parser.add_argument('--ct', action='store_true', help='Use CT option if available')
    args = parser.parse_args()
    PRICE_TYPE = args.price_type
    CT = "CT" if args.ct else "CS"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Using %s prices and %s option (if available)", PRICE_TYPE, CT)

    tier1_futures = load_config(PROJECT_ROOT / "tier1.yaml")
    tier2_futures = load_config(PROJECT_ROOT / "tier2.yaml")
    futures = tier1_futures + tier2_futures
    tier1_symbols = [f.symbol for f in tier1_futures]

    symbols = [f.symbol for f in futures]
    clscodes = [f.clscode for f in futures]
    types = [f.asset_class[0] for f in futures]
    key = pl.DataFrame({
        "clscode": clscodes,
        "symbol": symbols,
        "type": types,
    })

    dsfutclass = (
        pl.scan_csv(FUTURES_PATH / "dsfutclass.csv", ignore_errors=True)
        .select(['contrcode', 'clscode'])
        .collect()
    )
    dsfutclass = dsfutclass.join(key, on=['clscode'], how='left')
    dsfutcontr = pl.scan_csv(FUTURES_PATH / "dsfutcontr.csv", ignore_errors=True).collect()
    dsfuttrdcycle = (
        pl.scan_csv(FUTURES_PATH / "dsfuttrdcycle.csv")
        .select(['clscode', 'trdmth'])
        .collect()
    )
    dsfutcontrinfo = (
        pl.scan_csv(FUTURES_PATH / "dsfutcontrinfo.csv", ignore_errors=True, schema_overrides={"lasttrddate": pl.Date})
        .filter((pl.col("clscode") != 290) | (pl.col("lasttrddate") < date(2003, 12, 22)))
        .filter(pl.col("lasttrddate").is_not_null())
        .select(['futcode', 'contrcode', 'clscode', 'lasttrddate', 'dsmnem', 'isocurrcode', 'currunitcode'])
        .collect()
    )
    dsfutcontrval = (
        pl.scan_csv(FUTURES_PATH / "dsfutcontrval.csv", ignore_errors=True, schema_overrides={"date_": pl.Date})
        .select(['futcode', 'date_', 'volume', 'settlement', 'open_'])
        .with_columns(pl.col('open_').alias('open'))
        .collect()
    )
    main()
