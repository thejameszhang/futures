from datetime import date
from functools import lru_cache

import pandas_market_calendars as pmc
import polars as pl

from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass
from globalmacro.utils.paths import (
    COMPUSTAT_PATH,
    ECONOMICS_PATH,
    FUTURES_PATH,
    FX_PATH,
    PROJECT_ROOT,
)

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
    "RMB": "CNY",  # RMB futures map to the CNY spot series in the async/sync spot panels.
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


@lru_cache(maxsize=1)
def _synthetic_inputs():
    """(fx_list, clscodes, key, schedule_dates, LIBOR) — built once, no shared-panel side effects."""
    tier1 = load_config(PROJECT_ROOT / "tier1.yaml")
    tier2 = load_config(PROJECT_ROOT / "tier2.yaml")
    fx_list = [f for f in (tier1 + tier2) if f.asset_class[0] == AssetClass.CURRENCY and f.exrateintcode]
    clscodes = [f.clscode for f in fx_list]
    cme = pmc.get_calendar("CMEGlobex_FX")
    schedule = cme.schedule(start_date="1972-05-16", end_date="2025-12-31")
    schedule_dates = pl.Series("exratedate", schedule.index.to_numpy(dtype="datetime64[D]")).cast(pl.Date).implode()
    key = pl.DataFrame({"exrateintcode": [f.exrateintcode for f in fx_list], "symbol": [f.symbol for f in fx_list],
                        "clscode": clscodes, "inverted_pair": [f.inverted_pair for f in fx_list]})
    LIBOR = (pl.read_csv(ECONOMICS_PATH / "3month_libor_rates.csv", schema_overrides={"date": pl.Date})
             .fill_null(strategy="forward").sort("date"))
    return fx_list, clscodes, key, schedule_dates, LIBOR

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


def build_datastream_direct_panel() -> pl.DataFrame:
    """Datastream direct X<->USD SPOT rates, oriented to USD-per-X, on the 1950.. calendar.

    An X->USD contract's raw rate is X-per-USD (invert to 1/rate); a USD->X contract's
    raw rate is already USD-per-X (keep). Prefer whichever direct direction starts earlier.
    This is the async spot panel (Datastream is an end-of-day snapshot).
    """
    long = (
        pl.read_csv(FX_PATH / "ds2fxrate.csv",
                    schema_overrides={"exratedate": pl.Date, "midrate": pl.Float64,
                                      "bidrate": pl.Float64, "offerrate": pl.Float64})
        .join(pl.read_csv(FX_PATH / "ds2fxcode.csv"), on="exrateintcode", how="left")
        .filter(pl.col("ratetypecode") == "SPOT")
        .with_columns(pl.coalesce(pl.col("midrate"),
                                  (pl.col("bidrate") + pl.col("offerrate")) / 2,
                                  pl.col("offerrate"), pl.col("bidrate"))
                                  .alias("rate"))
        .select(["exratedate", "fromcurrcode", "tocurrcode", "rate"])
        .rename({"exratedate": "datadate", "fromcurrcode": "fromcurd",
                 "tocurrcode": "tocurd", "rate": "exratd"})
    )
    currencies = sorted({c for c in long["fromcurd"].to_list() + long["tocurd"].to_list()
                         if c and c != "USD"})
    series = []
    for x in currencies:
        tousd = long.filter((pl.col("fromcurd") == x) & (pl.col("tocurd") == "USD"))
        fromusd = long.filter((pl.col("fromcurd") == "USD") & (pl.col("tocurd") == x))
        t0 = tousd.select(pl.col("datadate").min()).item() if tousd.height else None
        f0 = fromusd.select(pl.col("datadate").min()).item() if fromusd.height else None
        if t0 is None and f0 is None:
            continue
        if (f0 is None) or (t0 is not None and t0 <= f0):
            s = tousd.select(pl.col("datadate"), (1.0 / pl.col("exratd")).alias(x))
        else:
            s = fromusd.select(pl.col("datadate"), pl.col("exratd").alias(x))
        s = s.drop_nulls().unique("datadate").sort("datadate")
        if s.height:
            series.append(s)
    if not series:
        raise RuntimeError("no direct Datastream X<->USD SPOT contracts found")
    date_end = max(m for s in series
                   if (m := s.select(pl.col("datadate").max()).item()) is not None)
    panel = pl.DataFrame({"datadate": pl.date_range(date(1950, 1, 1), date_end, "1d", eager=True)})
    for s in series:
        panel = panel.join(s, on="datadate", how="left")
    return (panel.rename({"datadate": "date"}).sort("date")
            .with_columns(pl.lit(1.0).alias("USD")).fill_null(strategy="forward"))


def build_synthetic(spot_panel: pl.DataFrame) -> pl.DataFrame:
    fx_list, clscodes, key, schedule_dates, LIBOR = _synthetic_inputs()

    synthetic_returns = []
    # Infer expiry dates for NOK, SEK, and 6N before they were traded on listed on CME to mimic the future's return
    expiry_dates = (
        load_expiry_dates()
        .filter(pl.col("clscode").is_in(clscodes))
        .sort(["clscode", "lasttrddate"])
    )
    # TODO: Add the other (symbols x clscode) pairs for the tier 2 currencies. Algorithmically
    symbol_clscode_pairs = [(f.symbol, f.clscode) for f in fx_list if f.symbol not in ("6A", "6B", "6C", "6E", "6J", "6S")]
    for _symbol, clscode in symbol_clscode_pairs:
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

    datastream_fx_rates = spot_panel.with_columns(
        [pl.col(c).cast(pl.Float64).alias(c) for c in spot_panel.columns if c != "date"]
    )

    for symbol, libor_symbol in CURRENCY_TO_LIBOR_SYMBOL_MAPPING.items():
        if SYMBOL_TO_CURCDD_MAPPING[symbol] not in datastream_fx_rates.columns:
            continue
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
    wide_synthetic_fx_returns = synthetic_fx_returns.pivot(on="symbol", values="ret1", index="date").sort("date")
    return wide_synthetic_fx_returns


if __name__ == "__main__":
    # Two source-specific spot panels (no shared tousd panel):
    #   async  -> Datastream direct (end-of-day / settlement-timed)
    #   sync   -> Compustat cross   (WM/R 4pm London ~ 11am ET)
    # Async first, and completely. Compustat is a separate WRDS entitlement and its panel
    # feeds only the SYNC _usd output; a failure there must not also cost us the async
    # panel we have already computed.
    fx_async = build_datastream_direct_panel()
    fx_async.write_csv(FX_PATH / "fx_async.csv")
    build_synthetic(fx_async).write_csv(FX_PATH / "synthetic_fx_returns_async.csv")

    fx_sync = save_compustat_fx_rates()  # GBP cross on Compustat exrt_dly
    fx_sync.write_csv(FX_PATH / "fx_sync.csv")
    build_synthetic(fx_sync).write_csv(FX_PATH / "synthetic_fx_returns_sync.csv")
