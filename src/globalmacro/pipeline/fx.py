from datetime import time, date
import pandas_market_calendars as pmc
import polars as pl
from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass
from globalmacro.utils.paths import (
    CHARACTERISTICS_ROOT,
    COMPUSTAT_PATH,
    PROJECT_ROOT, 
    ECONOMICS_PATH, 
    FX_PATH, 
    FUTURES_PATH,
    DATASTREAM_PATH,
)

def load_expiry_dates() -> pl.DataFrame:
    """Load expiry dates from the LSEG Datastream."""
    return (
        pl.scan_csv(FUTURES_PATH / "dsfutcontrinfo.csv", ignore_errors=True, schema_overrides={"lasttrddate": pl.Date})
        .filter((pl.col("clscode") != 290) | (pl.col("lasttrddate") < date(2003, 12, 22)))
        .filter(pl.col("lasttrddate").is_not_null())
        .select(['clscode', 'lasttrddate'])
        .collect()
    )

def __estimate_fx_rates(fx_rates: pl.DataFrame) -> pl.DataFrame:
    """
    Given a DataFrame of FX rates (Compustat or Datastream), return the tousd rate.
    """
    # These rates are estimated using the X/GBP rate; works better with Compustat
    fx1 = (
        fx_rates.lazy()
        .filter(pl.col("fromcurd") == "GBP")
        .join(
            fx_rates.lazy()
            .filter(pl.col("tocurd") == "USD")
            .select(["fromcurd", "datadate", "exratd"])
            .rename({"exratd": "exratd_usd"}),
            on=["fromcurd", "datadate"],
            how="inner",
        )
        .select([
            pl.col("tocurd").alias("curcdd"),
            pl.col("datadate"),
            (pl.col("exratd_usd") / pl.col("exratd")).alias("fx"),
        ])
        .drop_nulls()
        .unique()
        .collect()
        .sort(["curcdd", "datadate"])
    )

    wide_fx1 = (
        fx1.pivot(on="curcdd", values="fx", index="datadate", aggregate_function="first") # Euro has duplicates, replaced by direct rate
        .with_columns(pl.col("datadate").dt.date().alias("date"))
        .drop("datadate")
        .sort("date")
        .drop("USD")
    )
    date_end = wide_fx1.select(pl.col("date").max()).item()
    calendar = pl.DataFrame(
        {"date": pl.date_range(date(1950, 1, 1), date_end, "1d", eager=True)}
    )
    wide_fx1 = (
        calendar
        .join(wide_fx1, on="date", how="left")
        .with_columns(pl.lit(1.0).alias("USD"))
        .sort("date")
        .fill_null(strategy="forward")
    )
    wide_fx1 = wide_fx1.select(["date"] + [col for col in wide_fx1.columns if col != "date"])
    return wide_fx1    


def save_compustat_fx_rates() -> pl.DataFrame:
    """
    Save the Compustat FX rates to a CSV file.
    """
    compustat_fx_rates = pl.read_csv(COMPUSTAT_PATH / "exrt_dly.csv", schema_overrides={"datadate": pl.Date})
    estimated_compustat_fx_rates = __estimate_fx_rates(compustat_fx_rates)
    estimated_compustat_fx_rates.write_csv(COMPUSTAT_PATH / "compustat_fx_rates.csv")
    return estimated_compustat_fx_rates


def save_datastream_fx_rates() -> pl.DataFrame:
    """
    Save the Datastream FX rates to a CSV file.
    """
    datastream_fx_rates = (
        pl.read_csv(FX_PATH / "ds2fxrate.csv", schema_overrides={"exratedate": pl.Date, "midrate": pl.Float64, "bidrate": pl.Float64, "offerrate": pl.Float64})
        .join(pl.read_csv(FX_PATH / "ds2fxcode.csv"), on="exrateintcode", how="left")
        .filter(pl.col("ratetypecode") == "SPOT")
        .with_columns(
            pl.coalesce(
                pl.col("midrate"),
                (pl.col("bidrate") + pl.col("offerrate")) / 2,
                pl.col("offerrate"),
                pl.col("bidrate"), 
            )
            .alias("rate")
        )
        .select(["exratedate", "fromcurrcode", "tocurrcode", "rate"])
        .rename({"exratedate": "datadate", "fromcurrcode": "fromcurd", "tocurrcode": "tocurd", "rate": "exratd"})
    )

    estimated_datastream_fx_rates = __estimate_fx_rates(datastream_fx_rates)

    # Supplement with longer history direct rates where available
    currencies = datastream_fx_rates.select("fromcurd").unique().to_series().to_list() + datastream_fx_rates.select("tocurd").unique().to_series().to_list()
    currencies = sorted(list(set(currencies)))
    for curcdd in currencies:
        if curcdd == "USD":
            continue
        print(f"Processing {curcdd}...")
        tousd = datastream_fx_rates.filter((pl.col("fromcurd") == curcdd) & (pl.col("tocurd") == "USD"))
        tousd_start_date = tousd.select("datadate").min().item()
        fromusd = datastream_fx_rates.filter((pl.col("fromcurd") == "USD") & (pl.col("tocurd") == curcdd))
        fromusd_start_date = fromusd.select("datadate").min().item()

        if tousd_start_date is None and fromusd_start_date is None:
            print(f"No direct rates found for {curcdd}, skipping...")
            continue

        if (fromusd_start_date is None and tousd_start_date is not None) or (tousd_start_date is not None and tousd_start_date < fromusd_start_date):
            # Invert the rate if necessary
            tousd = tousd.with_columns((1 / pl.col("exratd")).alias("exratd"))
            tousd = tousd.pivot(on="fromcurd", values="exratd", index="datadate").sort("datadate")
            if curcdd in estimated_datastream_fx_rates.columns:
                estimated_start_date = estimated_datastream_fx_rates.select("date", curcdd).drop_nulls().min().select("date").item()
                if tousd_start_date > estimated_start_date:
                    print("Estimated rate has longer history than direct rate, skipping...")
                    continue
                else:
                    estimated_datastream_fx_rates = (
                        estimated_datastream_fx_rates
                        .drop(curcdd)
                        .join(tousd, left_on="date", right_on="datadate", how="left")
                    )
                    print(f"Replacing estimated rate with direct tousd rate...")
            else:
                estimated_datastream_fx_rates = estimated_datastream_fx_rates.join(tousd, left_on="date", right_on="datadate", how="left")
                print(f"Joined direct tousd rate...")
        else:
            fromusd = fromusd.pivot(on="tocurd", values="exratd", index="datadate").sort("datadate")

            if curcdd in estimated_datastream_fx_rates.columns:
                estimated_start_date = estimated_datastream_fx_rates.select("date", curcdd).drop_nulls().min().select("date").item()
                if fromusd_start_date > estimated_start_date:
                    print("Estimated rate has longer history than direct rate, skipping...")
                    continue
                else:
                    estimated_datastream_fx_rates = (
                        estimated_datastream_fx_rates
                        .drop(curcdd)
                        .join(fromusd, left_on="date", right_on="datadate", how="left")
                    )
                    print(f"Replacing estimated rate with direct fromusd rate...")
            else:
                estimated_datastream_fx_rates = estimated_datastream_fx_rates.join(fromusd, left_on="date", right_on="datadate", how="left")
                print(f"Joined direct fromusd rate...")

    estimated_datastream_fx_rates = (
        estimated_datastream_fx_rates
        .sort("date")
        .fill_null(strategy="forward")
        .with_columns([pl.col(col).cast(pl.Float64).alias(col) for col in estimated_datastream_fx_rates.columns if col != "date"])
    )
    estimated_datastream_fx_rates.write_csv(DATASTREAM_PATH / "fx" / "datastream_fx_rates.csv")
    return estimated_datastream_fx_rates


def estimate_fx_rates() -> pl.DataFrame:
    """
    Estimate the FX rates using the Compustat and Datastream data.
    """
    compustat_fx_rates = save_compustat_fx_rates()
    datastream_fx_rates = save_datastream_fx_rates()
    cols = set(compustat_fx_rates.columns + datastream_fx_rates.columns) - set(["date"])

    compustat_fx_rates = compustat_fx_rates.rename({col: f"compustat_{col}" for col in compustat_fx_rates.columns if col != "date"})
    datastream_fx_rates = datastream_fx_rates.rename({col: f"datastream_{col}" for col in datastream_fx_rates.columns if col != "date"})

    result_df = compustat_fx_rates.join(datastream_fx_rates, on="date", how="full").sort("date").fill_null(strategy="forward")
    reuslt_df = result_df.drop([col for col in result_df.columns if result_df[col].is_null().all()])

    for col in cols:
        compustat_col = f"compustat_{col}"
        datastream_col = f"datastream_{col}"
        compustat_has = compustat_col in compustat_fx_rates.columns 
        datastream_has = datastream_col in datastream_fx_rates.columns
        
        if compustat_has and datastream_has:
            compustat_start_date = compustat_fx_rates.select("date", compustat_col).drop_nulls().min().select("date").item()
            datastream_start_date = datastream_fx_rates.select("date", datastream_col).drop_nulls().min().select("date").item()
            if (compustat_start_date is not None and datastream_start_date is not None) and compustat_start_date < datastream_start_date:
                print(f"{col} present in both compustat and datastream, using compustat - {compustat_start_date}...")
                result_df = result_df.rename({compustat_col: col}).drop(datastream_col)
            else:
                datastream_start_date = datastream_fx_rates.select("date", datastream_col).drop_nulls().min().select("date").item()
                print(f"{col} present in both compustat and datastream, using datastream - {datastream_start_date}...")
                result_df = result_df.rename({datastream_col: col}).drop(compustat_col)
        elif compustat_has:
            compustat_start_date = compustat_fx_rates.select("date", compustat_col).drop_nulls().min().select("date").item()
            print(f"{col} present in compustat only, using compustat - {compustat_start_date}...")
            result_df = result_df.rename({compustat_col: col})
        elif datastream_has:
            datastream_start_date = datastream_fx_rates.select("date", datastream_col).drop_nulls().min().select("date").item()
            print(f"{col} present in datastream only, using datastream - {datastream_start_date}...")
            result_df = result_df.rename({datastream_col: col})

    result_df = (
        result_df
        .drop([col for col in result_df.columns if "compustat_" in col or "datastream_" in col])
        .with_columns([pl.col(col).cast(pl.Float64).alias(col) for col in result_df.columns if col != "date"])
        .select(["date"] + sorted([col for col in result_df.columns if col != "date"]))
        .sort("date")
        .fill_null(strategy="forward")
    )
    result_df.write_csv(COMPUSTAT_PATH / "tousd_panel.csv")
    return result_df


def main():
    synthetic_returns = []
    # Infer expiry dates for NOK, SEK, and 6N before they were traded on listed on CME to mimic the future's return
    expiry_dates = (
        load_expiry_dates()
        .filter(pl.col("clscode").is_in(clscodes))
        .sort(["clscode", "lasttrddate"])
    )
    # TODO: Add the other (symbols x clscode) pairs for the tier 2 currencies. Algorithmically
    symbol_clscode_pairs = [(f.symbol, f.clscode) for f in fx if f.symbol not in ("6A", "6B", "6C", "6E", "6J", "6S")]
    for symbol, clscode in symbol_clscode_pairs:
        symbol_start_date = (
            expiry_dates
            .filter(pl.col("clscode") == clscode)
            .select(pl.col("lasttrddate").min())
            .item()
        )
        pre_symbol = (
            expiry_dates
            .filter(pl.col("clscode") == 2125)
            .with_columns(clscode=pl.lit(clscode).cast(pl.Int64))
            .filter(pl.col("lasttrddate") < symbol_start_date)
        )
        expiry_dates = pl.concat([expiry_dates, pre_symbol])
    
    expiry_dates = (
        expiry_dates
        .sort(["clscode", "lasttrddate"])
        .with_columns(pl.col("lasttrddate").shift(-1).over("clscode").alias("next_lasttrddate"))
    )
    
    FX_RATES = pl.read_csv(COMPUSTAT_PATH / "tousd_panel.csv", schema_overrides={"date": pl.Date})
    datastream_fx_rates = FX_RATES.with_columns([pl.col(col).cast(pl.Float64).alias(col) for col in FX_RATES.columns if col != "date"])

    for symbol, libor_symbol in CURRENCY_TO_LIBOR_SYMBOL_MAPPING.items():
        ds2fxrate = (
            datastream_fx_rates
            .select("date", SYMBOL_TO_CURCDD_MAPPING[symbol])
            .sort("date")
            .with_columns(symbol=pl.lit(symbol))
            .rename({SYMBOL_TO_CURCDD_MAPPING[symbol]: "rate"})
            .join(key, on="symbol", how="left")
        )
        
        fx_rates = ds2fxrate.join_asof(
            expiry_dates,
            by="clscode",
            left_on="date",
            right_on="lasttrddate",
            strategy="forward",
            check_sortedness=False,
        ).with_columns(
            pl.col("date").dt.truncate("1mo").alias("month_start"),
            pl.col("lasttrddate").dt.truncate("1mo").alias("expiry_month_start"),
        ).filter(pl.col("date").is_in(schedule_dates))

        fx_with_libor = (
            fx_rates.join_asof(
                LIBOR.sort("date"),
                left_on="date",
                right_on="date",
                strategy="backward",
            ).with_columns(
                [pl.col(col).forward_fill().cast(pl.Float64).alias(col) for col in LIBOR.columns if col != "date"]
            ).with_columns(
                (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("delta_days")
            )
        )

        usd_rate = pl.col("ded3") / 100.0
        foreign_rate = pl.col(libor_symbol) / 100.0
        tau_front = (pl.col("lasttrddate") - pl.col("date")).dt.total_days()
        tau_back = (pl.col("next_lasttrddate") - pl.col("date")).dt.total_days()

        fx_with_libor = (
            fx_with_libor
            .filter(pl.col("rate").is_not_null())
            .filter(pl.col("next_lasttrddate").is_not_null())
        )

        foreign_daycount = (
            pl.when(pl.col("symbol").is_in(["6B", "6C", "6J"]))
            .then(365.0)
            .otherwise(360.0)
        )

        prc1 = (
            pl.col("rate")
            * (1.0 + usd_rate * (tau_front / 360.0))
            / (1.0 + foreign_rate * (tau_front / foreign_daycount))
        ).alias("prc1")
        prc2 = (
            pl.col("rate")
            * (1.0 + usd_rate * (tau_back / 360.0))
            / (1.0 + foreign_rate * (tau_back / foreign_daycount))
        ).alias("prc2")

        fx_with_libor = fx_with_libor.with_columns(
            [prc1, prc2]
        ).filter(
            pl.col("date").dt.weekday().is_in([1, 2, 3, 4, 5])
        ).with_columns([
            (pl.col("prc1") / pl.col("prc1").shift(1) - 1).alias("front_return"),
            (pl.col("prc2") / pl.col("prc2").shift(1) - 1).alias("back_return"),
            (pl.col("month_start") == pl.col("expiry_month_start")).alias("use_back_month"),
        ]).with_columns([
            (pl.col("use_back_month").cast(pl.Int8).diff().over("clscode")).alias("shift"),
            (pl.col("prc1") / pl.col("prc2").shift(1) - 1).alias("rollback_back_to_front"),
        ]).with_columns(
            pl.when(pl.col("use_back_month"))
            .then(pl.col("back_return"))
            .when(pl.col("shift") == -1)
            .then(pl.col("rollback_back_to_front"))
            .otherwise(pl.col("front_return"))
            .alias("ret1")
        )

        synthetic_fx_returns = (
            fx_with_libor
            .select("symbol", "date", "prc1", "prc2", "ret1")
            .with_columns(
                year_month=pl.col("date").dt.truncate("1mo"),
                asset_class=pl.lit(AssetClass.CURRENCY),
                lasttrddate1=pl.lit(None),
                lasttrddate2=pl.lit(None),
            )
            .sort("symbol", "date")
            .select("date", "year_month", "symbol", "asset_class", "prc1", "prc2", "ret1", "lasttrddate1", "lasttrddate2")
        )
        synthetic_returns.append(synthetic_fx_returns)

    synthetic_fx_returns = pl.concat(synthetic_returns).sort(["symbol", "date"])
    synthetic_fx_returns.write_csv(CHARACTERISTICS_ROOT / "synthetic_fx_info.csv")
    wide_synthetic_fx_returns = synthetic_fx_returns.pivot(on="symbol", values="ret1", index="date").sort("date")
    wide_synthetic_fx_returns.write_csv(FX_PATH / "synthetic_fx_returns.csv")


if __name__ == "__main__":
    tier1_futures = load_config(PROJECT_ROOT / "tier1.yaml")
    tier2_futures = load_config(PROJECT_ROOT / "tier2.yaml")
    futures = tier1_futures + tier2_futures
    fx = list(filter(lambda f: f.asset_class[0] == AssetClass.CURRENCY and f.exrateintcode, futures))
    exrateintcodes = [f.exrateintcode for f in fx]
    clscodes = [f.clscode for f in fx]

    start_date = "1972-05-16"
    end_date = "2025-12-31"
    cme = pmc.get_calendar("CMEGlobex_FX")
    schedule = cme.schedule(start_date=start_date, end_date=end_date)
    schedule_dates = pl.Series(
        "exratedate",
        schedule.index.to_numpy(dtype="datetime64[D]"),
    ).cast(pl.Date).implode()
    
    key = pl.DataFrame({
        "exrateintcode": exrateintcodes,
        "symbol": [f.symbol for f in fx],
        "clscode": clscodes,
        "inverted_pair": [f.inverted_pair for f in fx],
    })

    SYMBOL_TO_CURCDD_MAPPING = {
        "6A": "AUD",
        "6C": "CAD",
        "6J": "JPY",
        "6B": "GBP",
        "6E": "EUR",
        "6N": "NZD",
        "6S": "CHF",
        "NOK": "NOK",
        "SEK": "SEK",
        "KRW": "KRW",
        "6Z": "ZAR",
        "CZK": "CZK",
        "HUF": "HUF",
        "6M": "MXN",
        "PLN": "PLN",
        "ILS": "ILS",
        "RMB": "CNY",  # RMB futures map to CNY spot series in tousd_panel.
        "6L": "BRL"
    }

    CURRENCY_TO_LIBOR_SYMBOL_MAPPING = {
        "6A": "Australia",
        "6J": "TIFEY",
        "6B": "L",
        "6E": "I",
        "6N": "BB",
        "6S": "FES",
        "NOK": "NOK_rate_monthly",
        "SEK": "Sweden",
        "KRW": "Korea",
        "6Z": "6Z_rate",
        "CZK": "CZK_rate",
        "HUF": "Hungary",
        "6M": "Mexico",
        "PLN": "Poland",
        "ILS": "Israel",
        "RMB": "China (People’s Republic of)",
        "6L": "6L_rate",
    }
    LIBOR = (
        pl.read_csv(ECONOMICS_PATH / "3month_libor_rates.csv", schema_overrides={"date": pl.Date})
        .fill_null(strategy="forward")
        .sort("date")
    )

    estimate_fx_rates()

    main()
