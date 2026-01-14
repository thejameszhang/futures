import argparse
from datetime import date
import polars as pl
from utils.characteristics import *
from utils.config import load_config
from utils.splice import SPLICING_MAP
from utils.paths import (
    PROJECT_ROOT,
    DATASETS_ROOT, 
    GLOBALMACRO_ROOT, 
    FUTURES_PATH, 
    COMPUSTAT_PATH,
)


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
        print(f"Skipping splicing {active} to {inactive} because one of the symbols is not in the dataset")
        return contr_data

    print(f"Splicing {inactive} to {active}")
    # Find the first date that the active contract is available
    active_data = contr_data.filter(pl.col('symbol') == active)
    inactive_data = contr_data.filter(pl.col('symbol') == inactive)
    active_min_date = active_data.select(pl.col('date').min()).item()
    active_max_date = active_data.select(pl.col('date').max()).item()
    inactive_max_date = inactive_data.select(pl.col('date').max()).item()

    print(f"Active contract {active} starts on: {active_min_date}")
    print(f"Inactive contract {inactive} ends on: {inactive_max_date}")

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
            print(f"Volume crossover found: active {active} volume exceeded inactive {inactive} volume on {stitch_date}")
        else:
            stitch_date = max(overlap_end, active_min_date)
            print(f"WARNING: No volume crossover found for {active}/{inactive}. Using fallback date: {stitch_date}")
    else:
        stitch_date = active_min_date
        print(f"WARNING: No overlap period found for {active}/{inactive}. Using active start date: {stitch_date}")

    # Check for reasonable gap
    gap_days = (stitch_date - inactive_max_date).days if inactive_max_date < stitch_date else 0
    if gap_days > 10:
        print(f"WARNING: Gap of {gap_days} days is too large. Skipping stitching for {active}/{inactive}")
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

    print(f"Combined series for active {active}: {combined_active_series.height} total observations")
    print(f"  - Inactive data (before {stitch_date}): {inactive_before_stitch.height} observations")
    print(f"  - Active data (from {stitch_date}): {active_from_stitch.height} observations")
    return contr_data


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
        print(f"Clscodes with CT cycles: {clscode_to_ct}")
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
    - Around 2003-06-16, erroneous data points where the price is ~99
    2. Clscode 1065 - Kospi 200 Index Future
    - Around 2013-2014, erroneous data points with a common last trade date of 2015-12-10 
    where the price is ~800 
    3. Clscode 3829 - OMX30 Index Future
    - Around 2006-2007, erroneous data points with a common last trade date of 2006-06-16
    where the price is ~97, 98
    4. Clscode 2523 - 2 Year US Government Note
    - On 1992-12-30, there's an edge case in the data where on expiry date, the future with that expiry date 
    is not in the datasets -> essentially this return has a roll, so I'm manually editing this. The correct 
    return is 99.4375 / 99.40625 - 1 = 0.0003143665514
    """
    contr_data = contr_data.filter(
        (pl.col('clscode') != 1025) | (pl.col(PRICE_TYPE) < 1)
    ).filter(
        (pl.col('clscode') != 1065) | (pl.col(PRICE_TYPE) < 800)
    ).filter(
        (pl.col('clscode') != 3829) | (pl.col(PRICE_TYPE) > 100)
    )

    #ordering; if duplicate, keep row with the highest volume
    contr_data = (
        contr_data.sort(['clscode', 'date_', 'lasttrddate'])
        .with_columns(
            pl.col("lasttrddate").rank("dense").over(['clscode', 'date_']).alias("order")
        )
        .group_by(["clscode", "date_", "order"])
        .agg(pl.all().sort_by("volume", descending=True, nulls_last=True).first())
    ).filter(pl.col("order") <= 5)

    contr_data = contr_data.unique(keep='first').sort(['clscode', 'date_'])

    # Important: this price is in its local currency
    contr_data = (
        contr_data
        .with_columns(pl.col(PRICE_TYPE).cast(pl.Float64).alias(f"{PRICE_TYPE}_local"))
        .drop(PRICE_TYPE)
    )
    
    fx_long = (
        tousd
        .unpivot(index="date", variable_name="isocurrcode", value_name="fx")
        .with_columns(pl.col("fx").cast(pl.Float64).alias("fx"))
    )
    contr_data = (
        contr_data
        .join(fx_long, on=["date", "isocurrcode"], how="left")
        .with_columns((pl.col(f"{PRICE_TYPE}_local") * pl.col("fx")).alias(PRICE_TYPE))
    )
    
    wide_contr_data = contr_data.pivot(
        index=["clscode", "symbol", "type", "contrcode", "date", "fx"],
        on="order",
        values=[PRICE_TYPE, f"{PRICE_TYPE}_local", "volume", "lasttrddate", "daystomaturity", "exp", "switch"]
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
        .with_columns([
            pl.coalesce(pl.col("ret_1"), pl.col("ret_2")).alias("ret_1"),
            pl.coalesce(pl.col("ret_2"), pl.col("ret_1")).alias("ret_2"),
        ])
        .with_columns([
            pl.coalesce(pl.col("ret_local_1"), pl.col("ret_local_2")).alias("ret_local_1"),
            pl.coalesce(pl.col("ret_local_2"), pl.col("ret_local_1")).alias("ret_local_2"),
        ])
    )

    if PRICE_TYPE == "settlement" and CT == "CS":
        symbol_to_clscode_map = {f.symbol: f.clscode for f in futures}
        for symbol, clscode in symbol_to_clscode_map.items():
            wide_contr_data.filter(pl.col('clscode') == clscode).sort('date').write_csv(FUTURES_PATH / "debug" / "tables" / f"{symbol}.csv")
            wide_contr_data.filter(pl.col('clscode') == clscode).select('date').sort('date').write_parquet(FUTURES_PATH / "debug" / "dates" / f"{symbol}.parquet")

    calc_total_returns_with_roll_exprs = [
        calc_total_returns_with_roll(1, PRICE_TYPE),
        calc_total_returns_with_roll(2, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_total_returns_with_roll_exprs)
    )
    
    calc_basis_until_expiry_exprs = [
        calc_basis_until_expiry(1, PRICE_TYPE),
        calc_basis_until_expiry(2, PRICE_TYPE),
        calc_basis_until_expiry(3, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_basis_until_expiry_exprs)
    )
    
    calc_basis_with_roll_exprs = [
        calc_basis_with_roll(1),
        calc_basis_with_roll(2),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_basis_with_roll_exprs)
    )
    
    # Compute futures-implied spot and basis
    calc_spot_price_until_expiry_exprs = [
        calc_spot_price_until_expiry(1, PRICE_TYPE),
        calc_spot_price_until_expiry(2, PRICE_TYPE),
        calc_spot_price_until_expiry(3, PRICE_TYPE),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_spot_price_until_expiry_exprs)
    )
    
    calc_spot_with_roll_exprs = [
        calc_spot_price_with_roll(1),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_spot_with_roll_exprs)
    )
    
    calc_spot_returns_until_expiry_exprs = [
        calc_spot_returns_until_expiry(1),
        calc_spot_returns_until_expiry(2),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_spot_returns_until_expiry_exprs)
    )
    
    calc_spot_return_with_price_adj_and_roll_exprs = [
        calc_spot_return_with_price_adj_and_roll(1),
    ]
    wide_contr_data = wide_contr_data.with_columns(
        flatten(calc_spot_return_with_price_adj_and_roll_exprs)
    )
    
    # Save the start dates and end dates for ALL clscodes
    if PRICE_TYPE == 'settlement' and CT == "CS":
        data_availability = (
            wide_contr_data.filter(pl.col('settlement_1').is_not_null())
            .group_by('clscode').agg([
                pl.col('date').min().alias('earliest_date'),
                pl.col('date').max().alias('latest_date'),
            ]).sort('earliest_date', descending=True)
        )
        data_availability.write_csv(GLOBALMACRO_ROOT / "validation" / "all_data_availability.csv")

    # Save a parquet file for comparison with Datastream continuous series
    wide_contr_data.write_parquet(FUTURES_PATH / f"datastream_futures_{PRICE_TYPE}_{CT}.parquet")

    # Save the datasets and characteristics
    wide_contr_data = wide_contr_data.filter(pl.col('clscode').is_in(clscodes))
    tier1_symbols_filtered = list(set(tier1_symbols).intersection(set(wide_contr_data['symbol'].unique())))

    def save_returns(returns_variable: str, tier: int=2):
        dir = (
            GLOBALMACRO_ROOT / "characteristics" / f"tier{tier}" / "async" 
            if "spot" in returns_variable 
            else DATASETS_ROOT / f"tier{tier}" / "async"
        )
        returns_data = wide_contr_data.pivot(on='symbol', values=returns_variable, index='date').sort('date')
        columns = (
            sorted(tier1_symbols_filtered) 
            if tier == 1 
            else sorted([col for col in returns_data.columns if col != "date"])
        )
        returns_data = returns_data.select(["date"] + columns)
        returns_data.write_csv(dir / f"daily_{returns_variable}_{CT}.csv")
        print(f"Returns data saved to {dir} / daily_{returns_variable}_{CT}.csv")

    if PRICE_TYPE == 'settlement':
        print(f"Saving datasets and characteristics for Settlement prices")
        for tier in [1, 2]:
            save_returns("ret_1", tier=tier)
            save_returns("ret_2", tier=tier)
            save_returns("spot_ret_1", tier=tier)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--price_type', type=str, default='settlement', choices=['settlement', 'open'], help='Price type to use')
    parser.add_argument('--ct', action='store_true', help='Use CT option if available')
    args = parser.parse_args()
    PRICE_TYPE = args.price_type
    CT = "CT" if args.ct else "CS"
    print(f"Using {PRICE_TYPE} prices and {CT} option (if available)")

    tier1_symbols = [f.symbol for f in load_config(PROJECT_ROOT / "tier1.yaml")]
    futures = load_config(PROJECT_ROOT / "tier2.yaml")
    symbols = [f.symbol for f in futures]
    clscodes = [float(f.clscode) for f in futures]
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
    tousd = pl.read_csv(COMPUSTAT_PATH / "merged_tousd.csv", schema_overrides={"date": pl.Date})

    main()
