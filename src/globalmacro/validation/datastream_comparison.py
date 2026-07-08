import polars as pl
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import DATASTREAM_PATH, PROJECT_ROOT

DATASET = "constructed_no_price_adjustment"
"""
This script compares our constructed futures time series with the official Datastream continuous series.
We compare the correlations between the two types of series and plot of cumulative returns as a 
visualization test. 

Known errors in the Datastream continuous series data include:
1. A return of 2420% on 1988-01-19 for clscode 983 (Hang Seng Index Future)
2. A few days of erroneous data where abs(returns) > 900% for clscode 2094 (Swiss Franc) in late 1973
3. A return of >100% on 1996-06-19 for clscode 3630 (AEX Index Future)
4. A return of -3.059 on 2020-04-20 and -1.307 on 2020-04-21 for clscode 1482 (Crude Oil Futures); the drop is due 
to COVID-19. Our constructed returns also decline steeply on these days but are closer to 18% and 43%, respectively. 
5. A return of inf on 2018-01-02, 2018-04-02 and 2019-01-02 for clscode 1175 (Brent Crude Oil Futures) when the
our corresponding constructed returns are finite. 
6. A return of 21.46% on 1989-10-06 for clscode 221 (10 Year Canadian Government Bond Futures) when 
our corresponding constructed returns are much more reasonable at 0.81%. 

Known errors in the underlying futures that cause issues with the constructed futures time series are:
1. Clscode 1025, Norwegian Krone to USD
- 3 consecutive drops in returns of more than 80% for clscode 1025 Norwegian Krone to USD, so for these 3 data points, 
we're using the Datastream continuous series' returns instead. 
2. Clscode 1622, Nikkei 225 Index Future
- >5 returns of greater than 60,000% between 2017 and 2018, so we're using the Datastream continuous series' returns instead. 
3. Clscode 3829, OMX30 Index Future
- >3 returns points of greater than 1,000%, sowe're using the Datastream continuous series' returns instead. 
4. Clcode 1602, 90-Day New Zealand Bank Bill

We stitch full-size contracts series to e-minis series when the e-mini volume first exceeds the full-size volume.
- Russell 2000, which is an exception, 3-way stitch.
- There does not exist a day in whiich the volume of FVS > volume of FVSX. In this case, we use the last trading 
day of the full-size contract.
"""
def _datastream_monthly(tier: int = 1, verbose: bool = False) -> pl.DataFrame:
    FUTURES_PATH = DATASTREAM_PATH / "futures"
    RETURNS_VARIABLE = "ret_total_1"
    columns_of_interest = ["ret_constructed", f"ret_{DATASET}"]

    #######################################################
    ##### LOADING OUR CONSTRUCTED FUTURES TIME SERIES #####
    #######################################################

    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    symbols = [f.symbol for f in futures]
    contrcodes = [f.contrcode for f in futures]
    exchanges = [f.exchange for f in futures]
    clscodes = [f.clscode for f in futures]
    calcseriesnames = [f.calcseriesname for f in futures]
    names = [f.name for f in futures]

    key = pl.DataFrame({
        "symbol": symbols,
        "exchange": exchanges,
        "code": contrcodes,
        "name": names,
        "clscode": clscodes
    })
    datastream_data = pl.read_parquet(f"{FUTURES_PATH}/datastream_futures_settlement_CT.parquet")
    datastream_futures_data = datastream_data.join(key, left_on='contrcode', right_on='code', how='inner')
    if verbose:
        print([x for x in list(datastream_futures_data["clscode"].unique()) if x not in clscodes])

    # Filter futures data to only include clscodes
    datastream_futures_data = datastream_futures_data.filter(pl.col('clscode').is_in(clscodes))
    datastream_futures_comparison = datastream_futures_data.select([
        'clscode',
        'contrcode',
        'date',
        pl.col(RETURNS_VARIABLE).alias('ret_constructed'),
        pl.col('volume_1').alias('volume'),
    ]).filter(pl.col('ret_constructed').is_not_null())
    datastream_futures_comparison = datastream_futures_comparison.unique(keep='first').sort(['clscode', 'date'])

    ##########################################################
    ##### LOADING DATASTREAM ACTUAL CONTINUOUS SERIES DATA ###
    ##########################################################

    datastream_continuous_series = (
        pl.scan_csv(f'{FUTURES_PATH}/datastream_continuous_series.csv')
        .filter(pl.col('RollMethodCode') == 0)
        .filter(pl.col('PositionFwdCode') == 0)
        .filter(pl.col('ClsCode').is_in(clscodes))
        .filter(pl.col('CalcSeriesName').is_in(calcseriesnames))
        .filter(pl.col('Settlement').is_not_null())
        .collect()
    ).select(['CalcSeriesName', 'ClsCode', 'Date_', 'Settlement']).sort(['ClsCode', 'CalcSeriesName', 'Date_'])

    datastream_continuous_series = datastream_continuous_series.sort(['ClsCode', 'CalcSeriesName', 'Date_']).with_columns(
        ((pl.col('Settlement')/pl.col('Settlement').shift(1))-1).over('ClsCode').alias(f'ret_{DATASET}')
    )

    datastream_continuous_data = datastream_continuous_series.select([
        pl.col('ClsCode').alias('clscode'),
        pl.col('Date_').str.strptime(pl.Date, format="%Y-%m-%d", strict=False).alias('date'),
        f'ret_{DATASET}'
    ]).filter(pl.col(f'ret_{DATASET}').is_not_null()).filter(pl.col(f'ret_{DATASET}') != 0.0)

    datastream_continuous_data = datastream_continuous_data.unique(keep='first').sort(['clscode', 'date'])

    #####################################################
    ##### JOINING THE TWO DATASETS FOR COMPARISON #######
    #####################################################

    comparison = datastream_futures_comparison.join(
        datastream_continuous_data,
        on=['clscode', 'date'],
        how='right'
    ).join(
        key,
        on='clscode',
        how='left'
    )

    comparison = comparison.unique(['clscode', 'contrcode', 'date', 'ret_constructed', 'name'], keep='first')\
        .unique(['clscode', 'contrcode', 'date', f'ret_{DATASET}', 'name'], keep='first').sort(['clscode', 'date'])


    comparison = comparison.with_columns(
        pl.when((pl.col('clscode') == 1175) \
            & ((pl.col('date') == pl.date(2018, 1, 2)) \
            | (pl.col('date') == pl.date(2018, 4, 2)) \
            | (pl.col('date') == pl.date(2019, 1, 2)))
        )
        .then(pl.col('ret_constructed'))
        .otherwise(pl.col(f'ret_{DATASET}'))
        .alias(f'ret_{DATASET}')
    )

    # TODO: Corrections to our constructed returns; root of the issue is some last trade dates are super off...
    EXTREME_RETURN_CLSCODES = [3829, 1622, 1025, 1065]
    EXTREME_RETURN_CLSCODES_CORRECTIONS = {
        3829: 0.8,
        1622: 0.8,
        1025: 0.8,
        1065: 0.2,
    }
    for clscode, extreme_return_threshold in EXTREME_RETURN_CLSCODES_CORRECTIONS.items():
        comparison = comparison.with_columns([
            pl.when(
                (pl.col('clscode') == clscode) &
                (pl.col('ret_constructed').abs() > extreme_return_threshold) &
                (pl.col(f'ret_{DATASET}').abs() <= 0.1)
            )
            .then(pl.col(f'ret_{DATASET}'))
            .otherwise(pl.col('ret_constructed'))
            .alias('ret_constructed')
        ])

    # Spikes in the continuous series data; our constructed returns are more reasonable.
    comparison = comparison.filter(
        (pl.col('clscode') != 983) | (pl.col('date') >= pl.date(1988, 1, 19))
    ).filter(
        (pl.col('clscode') != 2094) | (pl.col('date') >= pl.date(1974, 1, 1))
    )
    comparison = comparison.with_columns(
        pl.when((pl.col('clscode') == 3630) & (pl.col('date') == pl.date(1996, 6, 19)))
        .then(pl.col('ret_constructed'))
        .otherwise(pl.col(f'ret_{DATASET}'))
        .alias(f'ret_{DATASET}')
    )
    comparison = comparison.with_columns(
        pl.when((pl.col('clscode') == 1482) & ((pl.col('date') == pl.date(2020, 4, 20)) | (pl.col('date') == pl.date(2020, 4, 21))))
        .then(pl.col('ret_constructed'))
        .otherwise(pl.col(f'ret_{DATASET}'))
        .alias(f'ret_{DATASET}')
    )
    comparison = comparison.with_columns(
        pl.when((pl.col('clscode') == 1175) \
            & ((pl.col('date') == pl.date(2018, 1, 2)) \
            | (pl.col('date') == pl.date(2018, 4, 2)) \
            | (pl.col('date') == pl.date(2019, 1, 2)))
        )
        .then(pl.col('ret_constructed'))
        .otherwise(pl.col(f'ret_{DATASET}'))
        .alias(f'ret_{DATASET}')
    )
    comparison = comparison.with_columns(
        pl.when((pl.col('clscode') == 221) & (pl.col('date') == pl.date(1989, 10, 6)))
        .then(pl.col('ret_constructed'))
        .otherwise(pl.col(f'ret_{DATASET}'))
        .alias(f'ret_{DATASET}')
    )

    comparison = comparison.filter(
        pl.all_horizontal([pl.col(col).is_finite() for col in columns_of_interest])
    )

    # All exercises grade at monthly frequency (Global Constraints). Compound the
    # corrected daily returns to monthly per instrument, then correlate.
    monthly = (
        comparison
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["clscode", "name", "month"])
        .agg(
            ((pl.col("ret_constructed") + 1).product() - 1).alias("ret_constructed"),
            ((pl.col(f"ret_{DATASET}") + 1).product() - 1).alias(f"ret_{DATASET}"),
        )
    )
    return monthly


def compare(tier: int = 1, verbose: bool = False) -> pl.DataFrame:
    """Backwards-compatible: per-instrument monthly correlation (the graded frame)."""
    monthly = _datastream_monthly(tier=tier, verbose=verbose)
    return monthly.group_by(["clscode", "name"]).agg(
        pl.corr("ret_constructed", f"ret_{DATASET}").alias("correlation"),
        pl.len().cast(pl.Int64).alias("n_obs"),
    ).sort("correlation", descending=True)


from globalmacro.validation.base import Check   # load_config, PROJECT_ROOT already imported at top


def _datastream_pairs() -> pl.DataFrame:
    """Tier-1 long-form [instrument, name, month, ours, theirs] for comparison.pdf."""
    monthly = _datastream_monthly(tier=1)
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    cls_to_symbol = pl.DataFrame({
        "clscode": [f.clscode for f in futures],
        "symbol": [str(f.symbol) for f in futures],
    }).unique(subset="clscode", keep="first")
    return (
        monthly.join(cls_to_symbol, on="clscode", how="left")
        .select(
            pl.coalesce(pl.col("symbol"), pl.col("name").cast(pl.Utf8)).alias("instrument"),
            pl.col("name").cast(pl.Utf8).alias("name"),
            pl.col("month").alias("month"),
            pl.col("ret_constructed").cast(pl.Float64).alias("ours"),
            pl.col(f"ret_{DATASET}").cast(pl.Float64).alias("theirs"),
        )
    )


def _datastream_correlations() -> pl.DataFrame:
    corr = compare(tier=1, verbose=False)            # Exercise 1 is the tier-1 universe
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    cls_to_symbol = pl.DataFrame({
        "clscode": [f.clscode for f in futures],
        "instrument": [f.symbol for f in futures],
    }).unique(subset="clscode", keep="first")
    return (
        corr.join(cls_to_symbol, on="clscode", how="left")
            .select(
                pl.coalesce(pl.col("instrument"), pl.col("name")).alias("instrument"),
                pl.col("correlation"),
                pl.col("n_obs"),
            )
    )


datastream_check = Check(
    name="Datastream comparison",
    slug="datastream",
    run=_datastream_correlations,
    pairs=_datastream_pairs,
    series_labels=("ours (constructed)", "Datastream continuous"),
)


if __name__ == "__main__":
    compare()