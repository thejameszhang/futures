import argparse
import polars as pl
import os
os.chdir("../")
from utils.config import load_config
from utils.splice import SPLICING_MAP


parser = argparse.ArgumentParser()
parser.add_argument('--price_type', type=str, default='settlement', choices=['settlement', 'open'], help='Price type to use')
parser.add_argument('--ct', action='store_true', help='Use CT option if available')
args = parser.parse_args()
PRICE_TYPE = args.price_type
CT = "CT" if args.ct else "CS"
DATA_PATH = 'data/datastream/futures/lseg'
print(f"Using {PRICE_TYPE} prices and {CT} option (if available)")

tier1_symbols = [f.symbol for f in load_config("../../tier1.yaml")]
futures = load_config("../../tier2.yaml")
symbols = [f.symbol for f in futures]
clscodes = [f.clscode for f in futures]
types = [f.asset_class[0] for f in futures]
key = pl.DataFrame({
    "clscode": clscodes,
    "symbol": symbols,
    "type": types,
})

dsfutclass= pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfutclass.csv", ignore_errors=True).select([
    'contrcode', 'clscode'
]).collect()
dsfutclass = dsfutclass.join(key, on=['clscode'], how='left')
dsfutcontr = pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfutcontr.csv", ignore_errors=True).collect()
dsfuttrdcycle = pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfuttrdcycle.csv").select([
    'clscode', 'trdmth'
]).collect()
dsfutcontrinfo = pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfutcontrinfo.csv", ignore_errors=True).with_columns([
    pl.col("startdate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").alias("startdate"),
    pl.col("lasttrddate").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").alias("lasttrddate")
    ]).select([
        'futcode', 'contrcode', 'clscode', 'unitcode', 'lasttrddate', 'dsmnem'
    ]).collect()
dsfutcode = pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfutcode.csv", ignore_errors=True).collect()
dsfutcontrval = pl.scan_csv(f"{DATA_PATH}/tr_ds_fut_dsfutcontrval.csv", ignore_errors=True).with_columns([
    pl.col("date_").cast(pl.Utf8).str.strptime(pl.Date, format="%Y-%m-%d").alias("date_")
    ]).select([
        'futcode', 'date_', 'volume','settlement', 'open_',
    ]).with_columns(pl.col('open_').alias('open')).collect()

#preparing futures data:
info_data = dsfutcontr.join(dsfutclass.select(['contrcode', 'clscode', 'symbol', 'type']), on=['contrcode'], how='left').filter(pl.col('clscode').is_not_null())
info_data = info_data.join(dsfutcontrinfo, on=['contrcode', 'clscode'], how='left').filter(pl.col('futcode').is_not_null())

#adding underlying instrument category
info_data = info_data.join(
    dsfutcode.filter(pl.col('type_')==2).select([pl.col('code').alias('undrinstrcode'), pl.col('desc_').alias('undrinstrtype')]),
    on=['undrinstrcode'],
    how='left'
)

#adding exchanges
info_data = info_data.join(
   dsfutcode.filter(pl.col('type_')==1).select([pl.col('code').alias('srccode'), pl.col('desc_').alias('exchange')]),
    on=['srccode'],
    how='left'
)

#adding underlying unit info
info_data = info_data.join(
   dsfutcode.filter(pl.col('type_')==3).select([pl.col('code').alias('unitcode'), pl.col('desc_').alias('unit')]),
    on=['unitcode'],
    how='left'
)

#adding trading months
info_data = info_data.join(
   dsfuttrdcycle.group_by("clscode").agg(pl.col("trdmth").str.join(",")),
    on=['clscode'],
    how='left'
)

#adding contract values
contr_data = info_data.join(dsfutcontrval, on=['futcode'], how='left').filter(pl.col('lasttrddate').is_not_null())

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

#adding a column to work at month level:
contr_data = contr_data.with_columns(
    (contr_data["date_"].dt.year() * 100 + contr_data["date_"].dt.month()).alias("month")
)

#adding a binary for expiry month observations:
contr_data = contr_data.with_columns(
    (
        ((contr_data["lasttrddate"].dt.year() * 100 + contr_data["lasttrddate"].dt.month()) == contr_data["month"])
        .cast(pl.Int8)  # makes it 1/0
        .alias("exp")
    )
)

contr_data = contr_data.with_columns(
    contr_data["exp"].fill_null(0).alias("exp")
)


#identifying the date of roll:
contr_data = contr_data.sort(['clscode', 'lasttrddate', 'date_']).with_columns(pl.col('exp').diff().over(['clscode', 'lasttrddate']).alias('switch')).with_columns(
    pl.when(pl.col('switch')==1).then(pl.lit(1)).otherwise(pl.lit(0)).alias('switch')
)

# Add daystomaturity column which is the difference between the lasttrddate and the date_
# filter out negative daystomaturity
contr_data = contr_data.with_columns([
    (pl.col("lasttrddate") - pl.col("date_")).dt.total_days().alias("daystomaturity")
]).filter(pl.col("daystomaturity") >= 0).filter(pl.col("daystomaturity").is_not_null()).with_columns(
    pl.col("date_").alias("date")
)

# Splicing
# symbol_mini_to_fullsize_mapping = {
#     # S&P 500
#     "ES": "SP",
#     # S&P 400 Midcap
#     "EMD": "MD",
#     # Nasdaq
#     "NQ": "ND",
#     # VSTOXX
#     "FVS": "FVSX",
#     # Dow Jones
#     "YM": "DJ",
#     # Russell 2000 is an exception, 4-way stitch; see docs
#     "RL": "ER2",
#     "TF": "RL",
#     "RTY": "TF",
#     # Austrian index,
#     "FATX": "ATX",
#     # Nikkei 225
#     "164120019": "167120018",
#     # TOPIX Index
#     "160120006": "161060005",
#     # MSCI EAFE,
#     "MFS": "EFE",
#     # Unleaded gasoline replaced by RBOB Gasoline in 2006; stitch together
#     "RB": "HU",
#     # Short term interest rates
#     # 3-Month SOFR replaced by 3-Month Eurodollar in 2018
#     "SR3": "GE",
#     # 3-Month TONA replaced the Euroyen in 2024
#     "91": "TIFEY",
#     # 3-Month SONIA replaced the 3-Month Short Sterling in 2018
#     "SO3": "L",
#     # 3-Month SARON replaced the 3-Month Euroswiss
#     "SA3": "FES",
#     # Euro Schatz replaced the Schatz in 1998
#     "FGBS": "SH2Z",
#     # Euro Bund replaced the Bund
#     "FGBL": "BDL",
#     # Euro Bobl replaced the Bobl
#     "FGBM": "BDM",
#     # CAC 40 Index
#     "FCE": "FCH",
#     # Euro 
#     "6E": "DM",
#     # FIB
#     "FIB": "IFX",
#     # Gas Oil (Dead) to Gas Oil Present
#     "G": "GG",
#     # Brent Crude Oil (Dead) to Brent Crude Oil
#     "BRN": "BR",
# }
# Gas oil in the active contract has a gap in the data; use Gas oil from IPE and splice
contr_data = contr_data.filter(
    (pl.col("clscode") != 1176) | (pl.col("date") > pl.date(2003, 1, 1))
).filter(
    # CAC 40 Franc-denominated to CAC 40 Euro-demoinated. The franc-demoninated is missing data from 91 to 95, so start at 95
    (pl.col("clscode") != 498) | (pl.col("date") > pl.date(1995, 1, 1))
)
for mini, fullsize in SPLICING_MAP.items():
    if mini not in symbols or fullsize not in symbols:
        print(f"Skipping splicing {mini} to {fullsize} because one of the symbols is not in the dataset")
        continue

    print(f"Splicing {fullsize} to {mini}")
    # Find the first date that the e-mini contract is available
    mini_data = contr_data.filter(pl.col('symbol') == mini)
    fullsize_data = contr_data.filter(pl.col('symbol') == fullsize)
    mini_start_date = mini_data.select('date').min().item()
    fullsize_end_date = fullsize_data.select('date').max().item()
    
    print(f"E-Mini contract {mini} starts on: {mini_start_date}")
    print(f"Full-size contract {fullsize} ends on: {fullsize_end_date}")
    
    # Find the date when e-mini volume first exceeds full-size volume
    overlap_start = mini_start_date
    overlap_end = min(fullsize_end_date, mini_data.select('date').max().item())
    
    if overlap_start <= overlap_end:
        # Get volume data for both contracts during overlap period
        mini_overlap = mini_data.filter(
            (pl.col('date') >= overlap_start) & (pl.col('date') <= overlap_end)
        ).select(['date', 'volume']).rename({'volume': 'mini_volume'})
        
        fullsize_overlap = fullsize_data.filter(
            (pl.col('date') >= overlap_start) & (pl.col('date') <= overlap_end)
        ).select(['date', 'volume']).rename({'volume': 'fullsize_volume'})
        
        # Join volume data on date
        volume_comparison = mini_overlap.join(fullsize_overlap, on='date', how='inner')
        
        # Find first date where mini volume > fullsize volume
        volume_crossover = volume_comparison.filter(
            pl.col('mini_volume') > pl.col('fullsize_volume')
        ).sort('date')
        
        if volume_crossover.height > 0:
            stitch_date = volume_crossover.select('date').head(1).item()
            print(f"Volume crossover found: E-Mini {mini} volume exceeded full-size {fullsize} volume on {stitch_date}")
        else:
            stitch_date = max(overlap_end, mini_start_date)
            print(f"WARNING: No volume crossover found for {mini}/{fullsize}. Using fallback date: {stitch_date}")
    else:
        stitch_date = mini_start_date
        print(f"WARNING: No overlap period found for {mini}/{fullsize}. Using mini start date: {stitch_date}")
    
    # Check for reasonable gap
    gap_days = (stitch_date - fullsize_end_date).days if fullsize_end_date < stitch_date else 0
    if gap_days > 10:
        print(f"WARNING: Gap of {gap_days} days is too large. Skipping stitching for {mini}/{fullsize}")
        continue
    
    # Get fullsize data before stitch date
    fullsize_before_stitch = fullsize_data.filter(pl.col('date') < stitch_date).with_columns([
        pl.lit(mini).alias('symbol') 
    ])
    
    # Get mini data from stitch date onwards
    mini_from_stitch = mini_data.filter(pl.col('date') >= stitch_date)
    
    # Remove original mini and fullsize entries from dataset FIRST
    contr_data = contr_data.filter(
        pl.col('symbol').is_null() | 
        ((pl.col('symbol') != mini) & (pl.col('symbol') != fullsize))
    )
    
    combined_mini_series = pl.concat([fullsize_before_stitch, mini_from_stitch]).sort('date')
    contr_data = pl.concat([contr_data, combined_mini_series]).sort(['symbol', 'date'])
    
    print(f"Combined series for E-Mini {mini}: {combined_mini_series.height} total observations")
    print(f"  - Full-size data (before {stitch_date}): {fullsize_before_stitch.height} observations")
    print(f"  - E-Mini data (from {stitch_date}): {mini_from_stitch.height} observations")


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
    .group_by(["clscode", "date_", "order"]) # .agg(pl.all().first())
    .agg(pl.all().sort_by("volume", descending=True, nulls_last=True).first())
).filter(pl.col("order") <= 5)

# Compute futures-implied spot and basis
contr_data = contr_data.unique(keep='first').sort(['clscode', 'date_'])
contr_data_wide = contr_data.pivot(
    index=["clscode", "symbol", "type", "contrcode", "srccode", "date"], # "dsp", "iv"
    on="order",
    values=[PRICE_TYPE, "volume", "lasttrddate", "daystomaturity", "exp", "switch"]
)
contr_data_wide = contr_data_wide.select(sorted(contr_data_wide.columns)).sort(['clscode', 'date'])

contr_data_wide = contr_data_wide.filter(
    (pl.col("clscode") != 1602) | (
        (pl.col('volume_1').is_not_null() & (pl.col('volume_1') > 0)) & 
        (pl.col('volume_2').is_not_null() & (pl.col('volume_2') > 0))
    )
)

# Use second-month price if front month contract is in its expiry month; use THIS column if using futures prices in characteristics calculations
def calc_price_with_roll(i: int):
    return pl.when(pl.col('exp_1') == 1).then(pl.col(f'{PRICE_TYPE}_{i + 1}')).otherwise(pl.col(f'{PRICE_TYPE}_{i}')).over(pl.col('clscode')).alias(f'adj_{PRICE_TYPE}_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_price_with_roll(1),
])

# First calculate the returns of the contracts until expiry date of each contract 
def calc_returns_until_expiry(i: int):
    # TODO: validate this second clause
    return pl.when((pl.col('daystomaturity_1').shift(1) == 0) | (pl.col('lasttrddate_1') == pl.col('lasttrddate_2').shift(1)))\
    .then((pl.col(f'{PRICE_TYPE}_{i}') / pl.col(f'{PRICE_TYPE}_{i + 1}').shift(1)) - 1)\
    .otherwise(((pl.col(f'{PRICE_TYPE}_{i}') / pl.col(f'{PRICE_TYPE}_{i}').shift(1)) - 1))\
    .over('clscode').alias(f'ret_temp_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_returns_until_expiry(1),
    calc_returns_until_expiry(2),
    calc_returns_until_expiry(3),
])

# Now calculate the returns of the contracts WITH price adjustment with rolling on the first trading day of the month
def calc_returns_with_price_adj_and_roll(i: int):
    return pl.when(pl.col('exp_1') == 1).then(pl.col(f'ret_temp_{i + 1}')).otherwise(pl.col(f'ret_temp_{i}')).over(pl.col('clscode')).alias(f'ret_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_returns_with_price_adj_and_roll(1),
    calc_returns_with_price_adj_and_roll(2),
]).with_columns([
    # If either column is null, use the other column 
    pl.coalesce(pl.col("ret_1"), pl.col("ret_2")).alias("ret_1"),
    pl.coalesce(pl.col("ret_2"), pl.col("ret_1")).alias("ret_2"),
])    

# if PRICE_TYPE == "settlement" and CT == "CS":
#     symbol_to_clscode_map = {f.symbol: f.clscode for f in futures}
#     for symbol, clscode in symbol_to_clscode_map.items():
#         contr_data_wide.filter(pl.col('clscode') == clscode).write_csv(f"validation/datastream_comparison/{symbol}.csv")

# For comparison purposes, finally calculate the returns of the contracts WITHOUT price adjustment with rolling on the first trading day of the month
# Edge case: front month contract rolls and expires on the day before
def calc_total_returns_with_roll(i: int):
    return pl.when(((pl.col('exp_1') == 1) & (pl.col('switch_1') == 1)) & (pl.col(f'daystomaturity_1').shift(1) != 0))\
    .then((pl.col(f'{PRICE_TYPE}_{i + 1}').shift(1) / pl.col(f'{PRICE_TYPE}_{i}').shift(1)) * (pl.col(f'ret_temp_{i + 1}') + 1) - 1)\
    .otherwise(pl.col(f'ret_{i}')).over(pl.col('clscode')).alias(f'ret_total_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_total_returns_with_roll(1),
    calc_total_returns_with_roll(2),
])

# Calculate the the futures-implied basis
# def calculate_basis(i: int):
#     """See Gorton and Rouwenhorst (2012) Page 83"""
#     return (365 * (pl.col(f'{PRICE_TYPE}_{i}') / pl.col(f'{PRICE_TYPE}_{i + 1}') - 1) / (pl.col(f'daystomaturity_{i + 1}') - pl.col(f'daystomaturity_{i}'))).over(pl.col('clscode')).alias(f'basis_temp_{i}')
# contr_data_wide = contr_data_wide.with_columns([
#     calculate_basis(1),
#     calculate_basis(2),
# ])

# Calculate the the futures-implied basis until expiry
def calculate_basis_until_expiry(i: int):
    """See Gorton and Rouwenhorst (2012) Page 83"""
    return (365 * (pl.col(f'{PRICE_TYPE}_{i}') / pl.col(f'{PRICE_TYPE}_{i + 1}') - 1) / (pl.col(f'daystomaturity_{i + 1}') - pl.col(f'daystomaturity_{i}'))).over(pl.col('clscode')).alias(f'basis_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calculate_basis_until_expiry(1),
    calculate_basis_until_expiry(2),
    calculate_basis_until_expiry(3),
])

# Strange things happen to futures prices too close to expiry, so use the second-month basis if front month contract is in its expiry month
def calculate_basis_with_roll(i: int):
    return pl.when(pl.col('exp_1') == 1).then(pl.col(f'basis_{i + 1}')).otherwise(pl.col(f'basis_{i}')).over(pl.col('clscode')).alias(f'adj_basis_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calculate_basis_with_roll(1),
    calculate_basis_with_roll(2),
])
# # Calculate the the futures-implied spot price
# def calculate_spot_price(i: int):
#     """See Gorton and Rouwenhorst (2012) Page 84"""
#     return (pl.col(f'{PRICE_TYPE}_{i}') * (1 + (pl.col(f'basis_{i}') / 365) * pl.col(f'daystomaturity_{i}'))).over(pl.col('clscode')).alias(f'spot_price_{i}')
# contr_data_wide = contr_data_wide.with_columns([
#     calculate_spot_price(1),
#     calculate_spot_price(2),
# ])

# Calculate the the futures-implied spot price until expiry
def calculate_spot_price_until_expiry(i: int):
    return (pl.col(f'{PRICE_TYPE}_{i}') * (1 + (pl.col(f'basis_{i}') / 365) * pl.col(f'daystomaturity_{i}'))).over(pl.col('clscode')).alias(f'spot_price_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calculate_spot_price_until_expiry(1),
    calculate_spot_price_until_expiry(2),
    calculate_spot_price_until_expiry(3),
])

# Calculate the the futures-implied spot price with roll
def calc_spot_price_with_roll(i: int):
    return pl.when(pl.col('exp_1') == 1).then(pl.col(f'spot_price_{i + 1}')).otherwise(pl.col(f'spot_price_{i}')).over(pl.col('clscode')).alias(f'adj_spot_price_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_spot_price_with_roll(1),
])

# Calculate the the futures-implied spot return with price adjustment and roll
def calc_spot_returns_until_expiry(i: int):
    return pl.when(pl.col('daystomaturity_1').shift(1) == 0)\
    .then((pl.col(f'spot_price_{i}') / pl.col(f'spot_price_{i + 1}').shift(1)) - 1)\
    .otherwise(((pl.col(f'spot_price_{i}') / pl.col(f'spot_price_{i}').shift(1)) - 1))\
    .over('clscode').alias(f'spot_ret_temp_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calc_spot_returns_until_expiry(1),
    calc_spot_returns_until_expiry(2),
])

# Calculate the the futures-implied spot return
def calculate_spot_return_with_price_adj_and_roll(i: int):
    return pl.when(pl.col('exp_1') == 1).then(pl.col(f'spot_ret_temp_{i + 1}')).otherwise(pl.col(f'spot_ret_temp_{i}')).over(pl.col('clscode')).alias(f'spot_ret_{i}')
contr_data_wide = contr_data_wide.with_columns([
    calculate_spot_return_with_price_adj_and_roll(1),
])

# Save the start dates and end dates for ALL clscodes
if PRICE_TYPE == 'settlement' and CT == "CS":
    data_availability = (
        contr_data_wide.filter(pl.col('settlement_1').is_not_null())
        .group_by('clscode').agg([
            pl.col('date').min().alias('earliest_date'),
            pl.col('date').max().alias('latest_date'),
        ]).sort('earliest_date', descending=True)
    )
    data_availability.write_csv(f"validation/availability/all_data_availability.csv")

# Save a parquet file for comparison with Datastream continuous series
contr_data_wide.write_parquet(f"{DATA_PATH}/datastream_futures_{PRICE_TYPE}_{CT}.parquet")


def save_returns(returns_variable: str, tier: int=2):
    dir = f"characteristics/tier{tier}/async" if "spot" in returns_variable else f"datasets/tier{tier}/async"
    returns_data = contr_data_wide.pivot(on='symbol', values=returns_variable, index='date').sort('date')
    columns = sorted(tier1_symbols) if tier == 1 else sorted([col for col in returns_data.columns if col != "date"])
    returns_data = returns_data.select(["date"] + columns)
    returns_data.write_csv(f"{dir}/daily_{returns_variable}_{CT}.csv")
    print(f"Returns data saved to {dir}/daily_{returns_variable}_{CT}.csv")

def save_characteristics(variable: str, tier: int=2):
    dir = f"characteristics/tier{tier}/async"
    characteristics_data = contr_data_wide.pivot(on='symbol', values=f'adj_{variable}', index='date').sort('date')
    columns = sorted(tier1_symbols) if tier == 1 else sorted([col for col in characteristics_data.columns if col != "date"])
    characteristics_data = characteristics_data.select(["date"] + columns)
    characteristics_data.write_csv(f'{dir}/daily_{variable}_{CT}.csv')
    print(f"Characteristics data saved to {dir}/daily_{variable}_{CT}.csv")

# Save the datasets and characteristics
contr_data_wide = contr_data_wide.filter(pl.col('clscode').is_in(clscodes))
tier1_symbols = list(set(tier1_symbols).intersection(set(contr_data_wide['symbol'].unique())))
if PRICE_TYPE == 'settlement':
    print(f"Saving datasets and characteristics for Settlement prices")
    for tier in [1, 2]:
        save_returns("ret_1", tier=tier)
        save_returns("ret_2", tier=tier)
        save_returns("spot_ret_1", tier=tier)

        save_characteristics("basis_1", tier=tier)
        save_characteristics("settlement_1", tier=tier)
        save_characteristics("spot_price_1", tier=tier)