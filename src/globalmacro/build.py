import argparse
import logging
import os
import sys
from collections.abc import Iterable
from datetime import date
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import polars as pl

from globalmacro.pipeline.fx import SYMBOL_TO_CURCDD_MAPPING
from globalmacro.pipeline.to_usd import usd_panel
from globalmacro.utils.capabilities import resolve_mode, sync_stage_outputs_ready
from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass, Future
from globalmacro.utils.panels import (
    first_finite_date,
    first_valid_date,
    is_present_expr,
)
from globalmacro.utils.paths import (
    COMPUSTAT_PATH,
    DATA_ROOT,
    DATASETS_ROOT,
    DATASTREAM_PATH,
    ECONOMICS_PATH,
    EQUITIES_PATH,
    FUTURES_PATH,
    FX_PATH,
    PROJECT_ROOT,
    TICKHISTORY_PATH,
    VALIDATION_OUTPUT,
)
from globalmacro.utils.splice import SPLICING_MAP
from globalmacro.utils.sync_fx import build_sync_fx_panel

LOG_WIDTH = 88

logger = logging.getLogger(__name__)

# The JKP sector files key returns by 2-digit GICS sector code. Map each code to
# the corresponding CME/ICE Select Sector futures Ticker Symbol so that sector
# columns match the ticker-based schema used for every other asset (see the
# "US Equity Sector" rows in universe.xlsx). Keep this the single source of truth
# for the GICS -> ticker relationship in the pipeline.
GICS_SECTOR_TICKERS: dict[str, str] = {
    "10": "XAE",  # Energy
    "15": "XAB",  # Materials
    "20": "XAI",  # Industrials
    "25": "XAY",  # Consumer Discretionary
    "30": "XAP",  # Consumer Staples
    "35": "XAV",  # Health Care
    "40": "XAF",  # Financials
    "45": "XAK",  # Information Technology
    "50": "XAZ",  # Communication Services
    "55": "XAU",  # Utilities
    "60": "XAR",  # Real Estate
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="globalmacro build", allow_abbrev=False)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--async-only", dest="mode", action="store_const", const="async-only",
                   help="build only the async datasets (no tick data required)")
    g.add_argument("--full", dest="mode", action="store_const", const="full",
                   help="require the sync inputs; fail rather than silently degrade")
    p.set_defaults(mode=None)
    return p.parse_args(argv)


def rename_gics_to_tickers(sectors: pl.DataFrame) -> pl.DataFrame:
    """Rename GICS-code sector columns to their Select Sector futures tickers.

    The JKP sector tables are pivoted on the ``gics`` code, producing columns
    named "10", "15", ...; this maps them to the ticker symbols (XAE, XAB, ...)
    used throughout the datasets. Fails loudly if an unexpected GICS code shows
    up, so a schema change in the source data cannot silently pass through.
    """
    gics_cols = [c for c in sectors.columns if c != "date"]
    unknown = sorted(set(gics_cols) - set(GICS_SECTOR_TICKERS))
    if unknown:
        raise ValueError(f"Unrecognized GICS sector codes in JKP data: {unknown}")
    return sectors.rename({c: GICS_SECTOR_TICKERS[c] for c in gics_cols})


def build_currency_map(futures: list[Future]) -> dict[str, str]:
    """Every published symbol -> its return currency (curcdd); JKP sectors are US -> USD.
    Fails loudly on a missing curcdd so an unmapped symbol can't slip into USD conversion."""
    ccy: dict[str, str] = {}
    for future in futures:
        if not future.curcdd:
            raise ValueError(f"Future {future.symbol} has no curcdd in config")
        ccy[future.symbol] = future.curcdd
    for ticker in GICS_SECTOR_TICKERS.values():
        ccy[ticker] = "USD"
    return ccy


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * LOG_WIDTH)
    logger.info(title)
    logger.info("=" * LOG_WIDTH)


def log_subsection(title: str) -> None:
    logger.info("")
    logger.info(title)
    logger.info("-" * len(title))


def format_date(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def ensure_date(df: pl.DataFrame, col: str = "date") -> pl.DataFrame:
    if col not in df.columns:
        return df
    dtype = df.schema.get(col)
    date_expr = pl.col(col)
    if dtype == pl.Date:
        return df
    if dtype == pl.Datetime:
        return df.with_columns(date_expr.cast(pl.Date).alias(col))
    if dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8, pl.Float64, pl.Float32):
        parsed = date_expr.cast(pl.Int64, strict=False).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False)
        return df.with_columns(parsed.alias(col))
    if dtype == pl.Utf8:
        parsed = pl.coalesce(
            [
                date_expr.str.strptime(pl.Date, "%Y%m%d", strict=False),
                date_expr.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
            ]
        )
        return df.with_columns(parsed.alias(col))
    return df.with_columns(date_expr.cast(pl.Date, strict=False).alias(col))


def is_missing_expr(col: str) -> pl.Expr:
    numeric_is_nan = (
        pl.col(col)
        .cast(pl.Float64, strict=False)
        .is_nan()
        .fill_null(False)
    )
    return pl.col(col).is_null() | numeric_is_nan


def read_csv(
    path: Path,
    *,
    date_col: str | None = "date",
    schema_overrides: dict[str, Any] | None = None,
    **kwargs: Any,
) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        try_parse_dates=True,
        schema_overrides=schema_overrides,
        infer_schema_length=10000,
        **kwargs,
    )
    if date_col:
        df = ensure_date(df, date_col)
    return drop_redundant_date_columns(df, date_col=date_col or "date")


def coerce_numeric_data(df: pl.DataFrame, *, date_col: str = "date") -> pl.DataFrame:
    cols = [col for col in df.columns if col != date_col]
    if not cols:
        return df
    return df.with_columns([pl.col(col).cast(pl.Float64, strict=False) for col in cols])


def drop_redundant_date_columns(df: pl.DataFrame, *, date_col: str = "date") -> pl.DataFrame:
    if date_col in df.columns and "date_" in df.columns:
        return df.drop("date_")
    return df


def combine_first_on_date(left: pl.DataFrame, right: pl.DataFrame, *, date_col: str = "date") -> pl.DataFrame:
    joined = full_join_on_date(left, right, date_col=date_col, suffix="_right")
    left_cols = [col for col in left.columns if col != date_col]
    right_cols = [col for col in right.columns if col != date_col]
    for col in left_cols:
        right_col = f"{col}_right"
        if right_col in joined.columns:
            joined = joined.with_columns(pl.coalesce([pl.col(col), pl.col(right_col)]).alias(col)).drop(right_col)
    select_cols = [date_col] + left_cols + [col for col in right_cols if col not in left_cols]
    return joined.select([col for col in select_cols if col in joined.columns]).sort(date_col)


def drop_all_null_rows(df: pl.DataFrame, *, date_col: str = "date") -> pl.DataFrame:
    data_cols = [col for col in df.columns if col != date_col]
    if not data_cols:
        return df
    mask = pl.any_horizontal(*[is_present_expr(col) for col in data_cols])
    return df.filter(mask)


def outer_join_on_date(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    date_col: str = "date",
    suffix: str = "_drop",
    drop_suffix: bool = True,
) -> pl.DataFrame:
    joined = full_join_on_date(left, right, date_col=date_col, suffix=suffix)
    if drop_suffix and suffix:
        drop_cols = [col for col in joined.columns if col.endswith(suffix)]
        if drop_cols:
            joined = joined.drop(drop_cols)
    return joined


def coalesce_join_date(df: pl.DataFrame, date_col: str = "date", suffix: str = "_right") -> pl.DataFrame:
    candidate = f"{date_col}{suffix}"
    if candidate not in df.columns:
        return df
    df = df.with_columns(pl.coalesce([pl.col(date_col), pl.col(candidate)]).alias(date_col))
    return df.drop(candidate)


def full_join_on_date(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    date_col: str = "date",
    suffix: str = "_right",
) -> pl.DataFrame:
    joined = left.join(right, on=date_col, how="full", suffix=suffix)
    return coalesce_join_date(joined, date_col, suffix=suffix)


def last_valid_date(df: pl.DataFrame, col: str, *, date_col: str = "date") -> Any:
    return df.select(pl.col(date_col).filter(is_present_expr(col)).max()).to_series().item()


def keep_after_date(df: pl.DataFrame, col: str, cutoff: date, *, inclusive: bool = True) -> pl.DataFrame:
    if inclusive:
        mask = pl.col("date") >= pl.lit(cutoff)
    else:
        mask = pl.col("date") > pl.lit(cutoff)
    return df.with_columns(pl.when(mask).then(pl.col(col)).otherwise(None).alias(col))


def set_null_on_date(df: pl.DataFrame, col: str, target: date) -> pl.DataFrame:
    return df.with_columns(
        pl.when(pl.col("date") == pl.lit(target)).then(None).otherwise(pl.col(col)).alias(col)
    )


def drop_columns_by_patterns(
    df: pl.DataFrame,
    *,
    contains: Iterable[str] | None = None,
    equals: Iterable[str] | None = None,
) -> pl.DataFrame:
    to_drop = set()
    if contains:
        for col in df.columns:
            if any(token in col for token in contains):
                to_drop.add(col)
    if equals:
        for col in equals:
            if col in df.columns:
                to_drop.add(col)
    return df.drop(list(to_drop)) if to_drop else df


def compute_monthly_returns(daily_ret_df: pl.DataFrame, min_observations: int = 15) -> pl.DataFrame:
    """Compound daily returns within each calendar month.

    min_observations answers "how many real observations does a month need before we
    are willing to state its return?", and it is deliberately NOT one number:

      * 15 (the default) -- a full month. This is what every validation call site wants:
        a thin month is a noisy statistic to correlate on, even when it is a legitimate
        figure to publish. It is also how filter_dataset_by_monthly_returns decides where
        a series STARTS, so that a series never opens on a partial-period return.
        Concretely: BXF's sync month for 2017-12 holds 10 observations compounding to
        -49.3% against an async +0.0%. Grading it drags BXF's async-vs-sync correlation
        from 0.874 to 0.702, pushing the symbol below the 0.80 floor.
      * 1 -- any month holding a real observation has a return. This is the shipped
        monthly product.

    A month with NO observations is null at every threshold. It is never 0.0.
    """
    df = daily_ret_df.sort("date").with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
    asset_cols = [col for col in df.columns if col not in {"date", "year_month"}]
    agg_exprs = [pl.col("date").max().alias("date")]
    for col in asset_cols:
        non_null_count = is_present_expr(col).sum()
        product_expr = (pl.col(col).cast(pl.Float64).fill_null(0).fill_nan(0) + 1).product() - 1
        agg_exprs.append(pl.when(non_null_count < min_observations).then(None).otherwise(product_expr).alias(col))
    monthly = df.group_by("year_month").agg(agg_exprs).sort("date")
    return monthly


def load_rf() -> pl.DataFrame:
    rf = read_csv(
        DATA_ROOT / "misc" / "F-F_Research_Data_Factors_daily.csv",
        date_col=None,
        skip_rows=4,
        schema_overrides={"RF": pl.Float64},
    )
    first_col = rf.columns[0]
    rf = rf.rename({first_col: "date", "RF": "rf"})
    rf = ensure_date(rf, "date")
    return rf.select(["date", (pl.col("rf") / 100).alias("rf")]).sort("date")


def load_async_dataset(tier: int = 1) -> pl.DataFrame:
    daily_ct = read_csv(DATASETS_ROOT / f"tier{tier}" / "async" / "daily_ret_1_CT.csv")
    daily_cs = read_csv(DATASETS_ROOT / f"tier{tier}" / "async" / "daily_ret_1_CS.csv")
    return coerce_numeric_data(combine_first_on_date(daily_ct, daily_cs))


def load_sectors_async() -> pl.DataFrame:
    sectors_async = read_csv(
        DATA_ROOT / "jkp" / "updated_daily_ind_gics.csv",
        schema_overrides={"gics": pl.Utf8},
    ).sort("date")
    sectors_async = sectors_async.pivot(values="ret_vw_cap", index="date", on="gics").sort("date")
    return coerce_numeric_data(rename_gics_to_tickers(sectors_async))


# Cash indices whose session (09:30-16:00 ET) falls entirely AFTER the sync panel's
# 09:31 ET sampling point. The sync return for day t spans [09:31 ET (t-1), 09:31 ET (t)],
# which therefore contains day t-1's Americas session, not day t's. Their synthetic
# backfill must be lagged one session -- in the SYNC panel ONLY. Async is settlement-timed.
#
# `exchange_pmc_name` now names the CASH exchange for the 37 indices carrying a
# `dsindexcode` (tier1.yaml / tier2.yaml) -- but not uniformly. Seven symbols with no
# `dsindexcode` still name a FUTURES venue (DJ, ER2, MD, ND, RL, SP, TF -- all
# CME_Equity/CBOT_Equity), and FESX deliberately stays on EUREX -- no single national
# exchange fits the Eurozone-wide Euro Stoxx 50. So the field alone can't drive this tuple --
# but the actual reason to keep it explicit is different: membership also requires that the
# symbol actually RECEIVES a synthetic backfill in the sync panel, which the config cannot
# express. ES and EMD both name NYSE but ship ZERO backfilled sync cells -- EMD is
# coalesced with historical MD via SPLICING_MAP, which pulls its pre-splice cutoff back
# to the panel floor -- so neither is listed here, and both drop out of
# validation/synthetic_equity.alignment() entirely.
#
# The tests and exercise (b)'s alignment invariant verify that the lag is applied correctly
# to the symbols listed here, and that no listed symbol is wrongly lagged -- they cannot
# detect a symbol wrongly OMITTED from this tuple. Membership is a human judgement, checked
# against each cash index's own exchange.
AMERICAS_CASH_INDICES = ("SXF", "YM", "NQ", "RTY", "IPC")


def lag_one_session(df: pl.DataFrame, symbols: Iterable[str]) -> pl.DataFrame:
    """Replace each symbol's value with the value at its PREVIOUS OWN observation.

    Gap-aware. `df` lives on a union date grid (every index's calendar merged), so a raw
    shift(1) would pull a value from a date the symbol did not trade. forward_fill().shift(1)
    yields the last value observed strictly before this row; masking to observed rows keeps
    the null pattern intact.

    A null NEVER becomes 0: rows where the symbol is null stay null, and a symbol's first
    observation becomes null (it has no previous observation).

    "Observed" means is_present_expr, as everywhere else in this module: a NaN is a missing
    observation, not a value to be carried forward as someone's lagged return.
    """
    # forward_fill().shift(1) is row-position-based, not date-value-based, so an unsorted
    # frame would silently produce a wrong lag. Sort defensively rather than trust the caller.
    df = df.sort("date")
    exprs = []
    for symbol in symbols:
        if symbol not in df.columns:
            logger.warning("lag_one_session: %s not in frame; skipping", symbol)
            continue
        present = is_present_expr(symbol)
        observed = pl.when(present).then(pl.col(symbol)).otherwise(None)
        previous = observed.forward_fill().shift(1)
        exprs.append(pl.when(present).then(previous).otherwise(None).alias(symbol))
    return df.with_columns(exprs) if exprs else df


def load_synthetic_returns(rf: pl.DataFrame, equities: list[Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    fx_async = coerce_numeric_data(drop_all_null_rows(
        read_csv(FX_PATH / "synthetic_fx_returns_async.csv").select(["date", "NOK", "SEK", "6N", "6A"])))
    fx_sync = coerce_numeric_data(drop_all_null_rows(
        read_csv(FX_PATH / "synthetic_fx_returns_sync.csv").select(["date", "NOK", "SEK", "6N", "6A"])))

    spot_equity_returns = read_csv(EQUITIES_PATH / "spot_equity_returns.csv")
    # No hand-written equity cutoffs live here: equities.py's first_daily_date truncates each
    # index at its first genuinely daily month (AUSTOLD: 12 observations/year until 1980).
    spot_equity_returns = coerce_numeric_data(drop_all_null_rows(spot_equity_returns).join(rf, on="date", how="left"))

    # Splice equity index returns together
    for equity in equities:
        if equity.dsindexcode and len(equity.dsindexcode) > 1:
            inactive = equity.dsindexmnem[0]
            if inactive not in spot_equity_returns.columns or equity.symbol not in spot_equity_returns.columns:
                logger.warning(f"Missing {inactive} or {equity.symbol} in spot_equity_returns")
                continue
            spot_equity_returns = spot_equity_returns.with_columns(
                (pl.coalesce(pl.col(equity.symbol), pl.col(inactive)) - pl.col("rf")).alias(equity.symbol)
            ).drop(inactive)
        else:
            if equity.symbol not in spot_equity_returns.columns:
                logger.warning(f"Missing {equity.symbol} in spot_equity_returns")
                continue
    spot_equity_returns = coerce_numeric_data(spot_equity_returns.drop("rf"))

    # The sync panel samples at 09:31 ET; an Americas cash session on day t falls entirely
    # after that point, so the day-t sync window actually contains day t-1's session. Lag
    # their synthetic one session. Applied to a sync-only copy -- async must NOT be lagged.
    spot_equity_sync = lag_one_session(spot_equity_returns, AMERICAS_CASH_INDICES)

    async_synth = coerce_numeric_data(outer_join_on_date(fx_async, spot_equity_returns).sort("date"))
    sync_synth = coerce_numeric_data(outer_join_on_date(fx_sync, spot_equity_sync).sort("date"))
    return async_synth, sync_synth


def splice_synthetic_returns(
    dataset: pl.DataFrame,
    synthetic: pl.DataFrame,
    dataset_label: str,
    *,
    skip_if_before: date | None = None,
) -> pl.DataFrame:
    log_subsection(f"Synthetic futures returns splicing ({dataset_label})")
    asset_cols = [col for col in synthetic.columns if col != "date"]
    adjusted = synthetic
    for asset in asset_cols:
        cutoff = first_valid_date(dataset, asset)
        if skip_if_before is not None and cutoff is not None and cutoff <= skip_if_before:
            logger.info(f"  {asset}: skipped (cutoff {format_date(cutoff)} <= {format_date(skip_if_before)})")
            continue
        logger.info(f"  {asset}: synthetic before {format_date(cutoff)} - new start date {format_date(first_valid_date(synthetic, asset))}")
        adjusted = adjusted.with_columns(
            pl.when(pl.col("date") < pl.lit(cutoff)).then(pl.col(asset)).otherwise(None).alias(asset)
        )
    adjusted = adjusted.rename({col: f"{col}_synthetic" for col in asset_cols})
    merged = outer_join_on_date(dataset, adjusted)
    for asset in asset_cols:
        synthetic_col = f"{asset}_synthetic"
        merged = merged.with_columns(
            pl.coalesce([pl.col(asset), pl.col(synthetic_col)]).alias(asset)
        ).drop(synthetic_col)
    return merged.sort("date")


def fill_gaps_with_synthetic_returns(
    dataset: pl.DataFrame,
    synthetic: pl.DataFrame,
    dataset_label: str,
    *,
    column: str,
) -> pl.DataFrame:
    log_subsection(f"Filling gaps with synthetic returns ({dataset_label})")
    if column not in dataset.columns or column not in synthetic.columns:
        logger.info(f"  {column}: skipped (not found in both datasets)")
        return dataset

    synthetic = synthetic.select(["date", column]).rename({column: f"{column}_synthetic_gap"})
    merged = outer_join_on_date(dataset, synthetic)

    gap_col = f"{column}_synthetic_gap"
    start = first_valid_date(synthetic, f"{column}_synthetic_gap")
    end = last_valid_date(synthetic, f"{column}_synthetic_gap")
    if start is None or end is None:
        logger.info(f"  {column}: skipped (no valid range)")
        return merged.drop(gap_col)
    logger.info(f"  {column}: fill gaps within {format_date(start)} to {format_date(end)}")
    merged = merged.with_columns(
        pl.when(
            (pl.col("date") >= pl.lit(start))
            & (pl.col("date") <= pl.lit(end))
            & is_missing_expr(column)
            & is_present_expr(gap_col)
        )
        .then(pl.col(gap_col))
        .otherwise(pl.col(column))
        .alias(column)
    )
    merged = merged.drop(gap_col)
    return merged.sort("date")


def fill_asynced_gaps_with_synthetic_returns(
    asynced: pl.DataFrame,
    synthetic: pl.DataFrame,
) -> pl.DataFrame:
    nok_gap_start = date(2004, 3, 16)
    nok_gap_end = date(2004, 4, 14)
    fce_gap_start = date(1998, 11, 2)
    fce_gap_end = date(1998, 12, 22)
    ap_gap_start = date(2010, 1, 1)
    ap_gap_end = date(2010, 2, 26)

    gap_synthetic = synthetic.select(["date", "NOK", "FCE", "AP"]).with_columns(
        [
            pl.when((pl.col("date") >= pl.lit(nok_gap_start)) & (pl.col("date") <= pl.lit(nok_gap_end)))
            .then(pl.col("NOK"))
            .otherwise(None)
            .alias("NOK"),
            pl.when((pl.col("date") >= pl.lit(fce_gap_start)) & (pl.col("date") <= pl.lit(fce_gap_end)))
            .then(pl.col("FCE"))
            .otherwise(None)
            .alias("FCE"),
            pl.when((pl.col("date") >= pl.lit(ap_gap_start)) & (pl.col("date") <= pl.lit(ap_gap_end)))
            .then(pl.col("AP"))
            .otherwise(None)
            .alias("AP"),
        ]
    )

    asynced = fill_gaps_with_synthetic_returns(
        asynced,
        gap_synthetic,
        f"Datastream NOK gap {format_date(nok_gap_start)} to {format_date(nok_gap_end)}",
        column="NOK",
    )
    asynced = fill_gaps_with_synthetic_returns(
        asynced,
        gap_synthetic,
        f"Datastream FCE gap {format_date(fce_gap_start)} to {format_date(fce_gap_end)}",
        column="FCE",
    )
    asynced = fill_gaps_with_synthetic_returns(
        asynced,
        gap_synthetic,
        f"Datastream AP gap {format_date(ap_gap_start)} to {format_date(ap_gap_end)}",
        column="AP",
    )
    return asynced


def build_synced_dataset(tier: int = 1) -> pl.DataFrame:
    USE_TRAD = ["I", "SI", "HG", "L", "AP", "GE", "TIFEY"]
    trad = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "traditional_daily_returns.csv"))
    trad = set_null_on_date(trad, "HG", date(2006, 7, 4))
    trad = set_null_on_date(trad, "TIFEY", date(2022, 9, 15))

    commodities = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "commodity_daily_returns.csv"))
    currencies = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "currency_daily_returns.csv"))

    bonds = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "bond_daily_returns.csv"))

    nonus = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "nonus_equity_daily_returns.csv"))
    nonus = keep_after_date(nonus, "FSMI", date(1998, 7, 21), inclusive=False)
    nonus = set_null_on_date(nonus, "OMX", date(1998, 4, 28)).drop("OMX")
    nonus = set_null_on_date(nonus, "FTI", date(1999, 1, 4))

    us = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "us_equity_daily_returns.csv"))
    us = keep_after_date(us, "YM", date(2002, 12, 9))
    us = keep_after_date(us, "EMD", date(2002, 2, 1))
    us = keep_after_date(us, "DJ", date(1997, 10, 7))

    volatilities = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "volatility_daily_returns.csv"))
    stirs = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier1" / "sync" / "stir_daily_returns.csv"))
    sectors = read_csv(
        DATA_ROOT / "jkp" / "updated_daily_ind_gics_synced.csv",
        schema_overrides={"gics": pl.Utf8},
    ).sort("date")
    sectors = sectors.pivot(values="ret_vw_cap", index="date", on="gics").sort("date")
    sectors = coerce_numeric_data(rename_gics_to_tickers(sectors))

    tables = [commodities, currencies, nonus, us, volatilities, stirs, sectors]
    if tier == 2:
        for t in ["currency", "equity"]:
            table = coerce_numeric_data(read_csv(DATASETS_ROOT / "tier2" / "sync" / f"{t}_daily_returns.csv"))
            if t == "currency":
                table = keep_after_date(table, "6Z", date(1997, 5, 8), inclusive=True)
            tables.append(table)

    synced = bonds
    for df in tables:
        synced = outer_join_on_date(synced, df)

    trad_columns = ["date"] + USE_TRAD
    trad_ct = trad.select(trad_columns).rename(
        {col: f"{col}_ct" for col in trad_columns if col != "date"}
    )
    trad_ct = keep_after_date(trad_ct, "TIFEY_ct", date(2002, 1, 1))
    trad_ct = outer_join_on_date(trad_ct, synced.select(trad_columns))

    for symbol in USE_TRAD:
        symbol_ct = f"{symbol}_ct"
        if symbol in trad_ct.columns and symbol_ct in trad_ct.columns:
            trad_ct = trad_ct.with_columns(
                pl.coalesce([pl.col(symbol_ct), pl.col(symbol)]).alias(symbol)
            ).drop(symbol_ct)
            logger.info(f"Spliced traditional symbol {symbol} with CT data.")
        else:
            logger.warning(f"{symbol} or its CT variant missing; skipping splice.")

    synced = synced.drop(USE_TRAD)
    synced = outer_join_on_date(synced, trad_ct)

    for active, historic in SPLICING_MAP.items():
        if active in synced.columns and historic in synced.columns:
            synced = synced.with_columns(
                pl.coalesce([pl.col(active), pl.col(historic)]).alias(active)
            ).drop(historic)
            logger.info(f"Spliced {active} with {historic} and dropped {historic}.")
        else:
            logger.warning(f"{active} or {historic} not found during splice; leaving untouched.")

    synced = drop_columns_by_patterns(synced, contains=["_open", "_drop"], equals=["date_"])
    synced = synced.sort("date")
    return synced


def filter_dataset_by_monthly_returns(dataset: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Where does each series START? At its first FULL month -- so that a series never
    # opens on a partial-period return wearing a month's label.
    dataset_monthly = compute_monthly_returns(dataset)
    dataset_monthly_cols = [col for col in dataset.columns if col not in {"date", "year_month"}]
    if dataset_monthly_cols:
        first_valid_months = dataset_monthly.select(
            [pl.col("year_month").filter(is_present_expr(col)).min().alias(col) for col in dataset_monthly_cols]
        ).row(0)
        cutoff_by_col = dict(zip(dataset_monthly_cols, first_valid_months, strict=False))
        cutoff_exprs = []
        for col in dataset_monthly_cols:
            cutoff = cutoff_by_col[col]
            if cutoff is None:
                expr = pl.when(pl.lit(False)).then(pl.col(col)).otherwise(None).alias(col)
            else:
                expr = pl.when(pl.col("date") >= pl.lit(cutoff)).then(pl.col(col)).otherwise(None).alias(col)
            cutoff_exprs.append(expr)
        dataset = dataset.with_columns(cutoff_exprs)
    # What is each month's RETURN? Any month with a real observation has one. Recomputed on
    # the CLIPPED daily panel, so months before a series starts stay null rather than
    # reappearing as thin leading fragments.
    dataset_monthly = compute_monthly_returns(dataset, min_observations=1).drop("year_month")
    return dataset, dataset_monthly

def save_datasets(
        synced: pl.DataFrame | None,
        asynced: pl.DataFrame,
        asynced_monthly: pl.DataFrame,
        tier1_symbols: list[str],
        tier2_symbols: list[str]
    ) -> tuple[
        pl.DataFrame | None, pl.DataFrame, pl.DataFrame,
        pl.DataFrame | None, pl.DataFrame, pl.DataFrame,
    ]:
    tier1_symbols = sorted(tier1_symbols)
    all_symbols = sorted(set(tier1_symbols + tier2_symbols))

    logger.info(f"Saving Tier 1 datasets: {len(tier1_symbols)} symbols")
    tier1_asynced_monthly = drop_all_null_rows(asynced_monthly.select(["date"] + tier1_symbols))
    tier1_asynced = drop_all_null_rows(asynced.select(["date"] + tier1_symbols))
    tier1_synced = (drop_all_null_rows(synced.select(["date"] + tier1_symbols))
                    if synced is not None else None)
    tier1_asynced_monthly.write_csv(DATASETS_ROOT / "tier1" / "async" / "async_monthly.csv")
    tier1_asynced.write_csv(DATASETS_ROOT / "tier1" / "async" / "async_daily.csv")
    if tier1_synced is not None:
        tier1_synced.write_csv(DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv")

    logger.info(f"Saving Tier 2 datasets: {len(tier2_symbols)} symbols")
    tier2_asynced_monthly = drop_all_null_rows(asynced_monthly.select(["date"] + all_symbols))
    tier2_asynced = drop_all_null_rows(asynced.select(["date"] + all_symbols))
    tier2_synced = None
    if synced is not None:
        tier2_synced_symbols = [symbol for symbol in all_symbols if symbol in synced.columns]
        missing_tier2_synced = sorted(set(all_symbols) - set(tier2_synced_symbols))
        if missing_tier2_synced:
            logger.warning(
                "Tier 2 synced missing symbols (keeping async-only): %s",
                ", ".join(missing_tier2_synced),
            )
        tier2_synced = drop_all_null_rows(synced.select(["date"] + tier2_synced_symbols))
    tier2_asynced_monthly.write_csv(DATASETS_ROOT / "tier2" / "async" / "async_monthly.csv")
    tier2_asynced.write_csv(DATASETS_ROOT / "tier2" / "async" / "async_daily.csv")
    if tier2_synced is not None:
        tier2_synced.write_csv(DATASETS_ROOT / "tier2" / "sync" / "sync_daily.csv")

    return tier1_synced, tier1_asynced, tier1_asynced_monthly, tier2_synced, tier2_asynced, tier2_asynced_monthly


def save_usd_datasets(
    tier1_synced: pl.DataFrame | None, tier1_asynced: pl.DataFrame,
    tier2_synced: pl.DataFrame | None, tier2_asynced: pl.DataFrame,
    symbol_to_ccy: dict[str, str], fx_async: pl.DataFrame, fx_sync: pl.DataFrame | None,
    out_root: Path = DATASETS_ROOT,
) -> None:
    """Write *_usd.csv siblings. async panels use Datastream FX (fx_async); sync panels
    use Compustat FX (fx_sync). Monthly USD is compounded from daily USD."""
    def _warn_first_obs_lost(df: pl.DataFrame, usd: pl.DataFrame, rel: str) -> None:
        # usd_panel correctly nulls any leading span where a currency's FX history
        # starts after the asset's own returns (safe: never a wrong value) — but
        # that truncation is otherwise silent. Surface it per symbol.
        for s in [c for c in df.columns if c != "date"]:
            local_first = df.filter(pl.col(s).is_not_null()).select(pl.col("date").min()).item()
            if local_first is None:
                continue
            usd_first = usd.filter(pl.col(s).is_not_null()).select(pl.col("date").min()).item()
            if usd_first is not None and usd_first <= local_first:
                continue
            if usd_first is None:
                lost = df.filter(pl.col(s).is_not_null()).height
            else:
                lost = df.filter(pl.col(s).is_not_null() & (pl.col("date") < usd_first)).height
            logger.warning(
                "USD panel %s: symbol %s (currency %s) FX coverage starts after local "
                "returns (local first %s, USD first %s) — %d observation(s) un-convertible",
                rel, s, symbol_to_ccy.get(s, "?"), local_first, usd_first, lost,
            )

    def _warn_stale_fx(df: pl.DataFrame, fx: pl.DataFrame, rel: str) -> None:
        # level.forward_fill() in usd_panel has no recency bound: if a needed
        # currency's series ever ends mid-panel, r_fx would silently become a
        # permanent 0 (USD == local), indistinguishable from a genuine zero FX
        # move. Not realized today (every currency quotes through the panel end),
        # but warn if that ever stops being true.
        symbols = [c for c in df.columns if c != "date"]
        panel_last = df.select(pl.col("date").max()).item()
        if panel_last is None:
            return
        needed = sorted({symbol_to_ccy[s] for s in symbols if s in symbol_to_ccy} - {"USD"})
        for c in needed:
            if c not in fx.columns:
                continue
            last_valid = fx.filter(pl.col(c).is_not_null()).select(pl.col("date").max()).item()
            if last_valid is not None and (panel_last - last_valid).days <= 7:
                continue
            logger.warning(
                "USD panel %s: currency %s FX quotes end at %s (panel ends %s) — "
                "forward-fill beyond this point would silently zero r_fx",
                rel, c, last_valid, panel_last,
            )

    def daily(df, fx, rel):
        usd = usd_panel(df, symbol_to_ccy, fx)
        _warn_first_obs_lost(df, usd, rel)
        _warn_stale_fx(df, fx, rel)
        (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
        usd.write_csv(out_root / rel)
        logger.info("Saved USD dataset to %s", out_root / rel)
        return usd
    def monthly(usd_daily, rel):
        m = drop_all_null_rows(compute_monthly_returns(usd_daily, min_observations=1).drop("year_month"))
        (out_root / rel).parent.mkdir(parents=True, exist_ok=True)
        m.write_csv(out_root / rel)
    u = daily(tier1_asynced, fx_async, "tier1/async/async_daily_usd.csv")
    monthly(u, "tier1/async/async_monthly_usd.csv")
    if tier1_synced is not None and fx_sync is not None:
        daily(tier1_synced, fx_sync, "tier1/sync/sync_daily_usd.csv")
    u = daily(tier2_asynced, fx_async, "tier2/async/async_daily_usd.csv")
    monthly(u, "tier2/async/async_monthly_usd.csv")
    if tier2_synced is not None and fx_sync is not None:
        daily(tier2_synced, fx_sync, "tier2/sync/sync_daily_usd.csv")


def currency_health(
    synced: pl.DataFrame, currency_symbols: list[str], threshold: float = 0.30, max_gap: int = 5
) -> list[str]:
    """One line per currency that has a return spike (|ret| > threshold) or a null
    run longer than max_gap, for validation_report.txt."""
    df = synced.with_columns(pl.col("date").cast(pl.Date, strict=False)).sort("date")
    lines: list[str] = []
    for sym in currency_symbols:
        if sym not in df.columns:
            continue
        r = df.get_column(sym).cast(pl.Float64, strict=False)
        n_spike = int((r.abs() > threshold).sum())
        # longest consecutive-null run within the symbol's observed span (first
        # non-null to last non-null) — leading/trailing listing-boundary pads
        # are not gaps.
        not_null = r.is_not_null().to_list()
        first_obs = next((i for i, v in enumerate(not_null) if v), None)
        last_obs = next((i for i in range(len(not_null) - 1, -1, -1) if not_null[i]), None)
        longest = 0
        if first_obs is not None and last_obs is not None:
            run = 0
            for v in r.is_null().to_list()[first_obs : last_obs + 1]:
                run = run + 1 if v else 0
                longest = max(longest, run)
        if n_spike:
            lines.append(f"currency {sym}: {n_spike} spike(s) |ret|>{threshold:.0%} (SHIPPED)")
        if longest > max_gap:
            lines.append(f"currency {sym}: max null gap {longest} days")
    return lines


def load_symbols(tier: int) -> tuple[list[Future], list[str]]:
    futures = load_config(PROJECT_ROOT / f"tier{tier}.yaml")
    symbols = [f.symbol for f in futures]
    return futures, symbols


def load_symbols_to_save(futures: list[Future]) -> list[str]:
    to_save = []
    for future in futures:
        if future.ric is not None and len(future.ric) > 1 or AssetClass.HISTORICAL not in future.asset_class:
            to_save.append(future.symbol)

    # Add the JKP equity sectors, using their Select Sector futures tickers
    # (see GICS_SECTOR_TICKERS) so the saved schema matches the renamed columns.
    to_save.extend(GICS_SECTOR_TICKERS.values())
    to_save = sorted(to_save)
    return to_save


class AsyncOutputs(TypedDict):
    asynced: pl.DataFrame
    asynced_monthly: pl.DataFrame


class SyncOutputs(TypedDict):
    synced: pl.DataFrame
    pre_splice_synced: pl.DataFrame


def build_async(synthetic_returns: pl.DataFrame) -> AsyncOutputs:
    """Everything the async datasets need. No tick data, no sync panel."""
    asynced = load_async_dataset(tier=2)
    asynced = keep_after_date(asynced, "FBTP", date(2009, 9, 14), inclusive=True)
    asynced = keep_after_date(asynced, "PLN", date(2004, 8, 1), inclusive=True)
    asynced = keep_after_date(asynced, "6Z", date(1997, 5, 8), inclusive=True)

    sectors_async = load_sectors_async()
    asynced = asynced.join(sectors_async, on="date", how="left")
    asynced = splice_synthetic_returns(asynced, synthetic_returns, "Datastream")
    asynced = fill_asynced_gaps_with_synthetic_returns(asynced, synthetic_returns)
    asynced = drop_all_null_rows(asynced).filter(pl.col("date") <= pl.date(2025, 12, 31))
    asynced, asynced_monthly = filter_dataset_by_monthly_returns(asynced)
    return {"asynced": asynced, "asynced_monthly": asynced_monthly}


def build_sync(synthetic_returns_synced: pl.DataFrame) -> SyncOutputs:
    """Everything that needs the tickhistory stage's outputs."""
    synced = build_synced_dataset(tier=2)
    pre_splice_synced = synced  # BEFORE splice_synthetic_returns: real-splice-aware (6E<-DM), synthetic-blind
    synced = splice_synthetic_returns(synced, synthetic_returns_synced, "TickHistory", skip_if_before=date(1996, 1, 4))
    # Capture-site guard (criterion c-2, "never the CIP synthetic"): a late-listing G10 future
    # (NOK, real 2002) is finite in the SYNTHETIC-BLIND pre-splice panel only from its real
    # start -- strictly LATER than in the post-splice `synced`, where the CIP synthetic backfills
    # it from 1996. If a future edit captured `pre_splice_synced` AFTER the splice, these dates
    # equalize and the blend would chain the synthetic below the real future; this assertion
    # fails the build loudly instead.
    assert first_finite_date(pre_splice_synced, "NOK") > first_finite_date(synced, "NOK"), (
        "pre_splice_synced is not synthetic-blind -- captured after splice_synthetic_returns?"
    )
    synced = drop_all_null_rows(synced).filter(
        (pl.col("date") >= pl.date(1996, 1, 4)) & (pl.col("date") <= pl.date(2025, 12, 31))
    )
    synced = keep_after_date(synced, "PLN", date(2004, 7, 14), inclusive=True)
    synced = keep_after_date(synced, "CZK", date(2004, 7, 14), inclusive=True)
    return {"synced": synced, "pre_splice_synced": pre_splice_synced}


def _validate_mode(mode: str) -> None:
    """`main()` is a public function callable directly with any string -- not just
    through the `__main__` CLI path, which already validates via `resolve_mode`.
    `main("Full")` or a typo'd `main("aysnc-only")` would otherwise silently take
    the async-only branch (mode == "full" is False for anything but the exact
    string) and report success on a truncated deliverable. Raise loudly instead.

    Factored out of `main()` so it can be exercised directly in tests -- this repo's
    tests must never call `main()` itself (it would build the real datasets).
    """
    if mode not in ("full", "async-only"):
        raise ValueError(
            f'globalmacro build: unrecognized mode {mode!r}; expected "full" or "async-only"'
        )


def main(mode: Literal["full", "async-only"] = "full") -> None:
    # A's TRIVIAL 2: validate BEFORE logging -- the old order let main("Bogus")
    # write "build mode: Bogus" to validation_report.txt and only THEN raise
    # (unreachable from the CLI, since resolve_mode() already only ever returns
    # "full"/"async-only", but main() is directly callable with any string).
    # _validate_mode raises immediately for anything else, producing no output of
    # its own (see test_validate_mode_produces_no_output_for_a_valid_mode) -- so
    # this reorder keeps the F5b guarantee intact: for the two mode values that
    # ever reach here in practice, the log line below is still the first EFFECTIVE
    # (output-producing) statement main() executes, even though it is no longer
    # the literal first statement in the function body (see
    # test_f5b_build_mode_is_logged_as_the_first_statement_in_main).
    _validate_mode(mode)
    # F5b (deliberate, owner-approved): the ONE intentional change to a shipped
    # artifact in this fix round. validation_report.txt is written by the
    # FileHandler attached in the __main__ block below, so this is its first line --
    # it now records which mode produced the datasets it describes. Adds exactly one
    # line; changes no number and no dataset.
    logger.info(f"build mode: {mode}")
    tier1_futures, tier1_symbols = load_symbols(1)
    tier2_futures, tier2_symbols = load_symbols(2)

    equities = [
        future
        for future in (tier1_futures + tier2_futures)
        if future.dsindexcode is not None
    ]
    rf = load_rf()
    synthetic_returns, synthetic_returns_synced = load_synthetic_returns(rf, equities)

    log_section("Dataset preparation")
    async_out = build_async(synthetic_returns)
    sync_out = build_sync(synthetic_returns_synced) if mode == "full" else None

    tier1_synced, tier1_asynced, tier1_asynced_monthly, tier2_synced, tier2_asynced, tier2_asynced_monthly = save_datasets(
        sync_out["synced"] if sync_out is not None else None,
        async_out["asynced"], async_out["asynced_monthly"],
        load_symbols_to_save(tier1_futures), load_symbols_to_save(tier2_futures))

    if tier2_synced is not None:
        log_subsection("Currency spike / gap diagnostics")
        currency_symbols = [
            f.symbol for f in tier1_futures + tier2_futures if AssetClass.CURRENCY in f.asset_class
        ]
        for line in currency_health(tier2_synced, currency_symbols):
            logger.warning(line)

    log_subsection("USD-converted datasets")
    fx_async = read_csv(FX_PATH / "fx_async.csv")
    fx_sync = None
    if sync_out is not None:
        fx_sync = read_csv(FX_PATH / "fx_sync.csv")
        # Blend the G10 FX-futures return into the sync Compustat levels (sync _usd only; async
        # untouched). Uses SYMBOL_TO_CURCDD_MAPPING (6E->EUR), NOT build_currency_map (all-USD).
        fx_sync = build_sync_fx_panel(fx_sync, sync_out["pre_splice_synced"], SYMBOL_TO_CURCDD_MAPPING)
    symbol_to_ccy = build_currency_map(tier1_futures + tier2_futures)
    save_usd_datasets(tier1_synced, tier1_asynced, tier2_synced, tier2_asynced,
                      symbol_to_ccy, fx_async, fx_sync)


if __name__ == "__main__":
    _args = _parse_args(sys.argv[1:])
    _mode = resolve_mode(_args.mode, sync_stage_outputs_ready(), "build")

    VALIDATION_DIR = VALIDATION_OUTPUT
    os.makedirs(VALIDATION_DIR, exist_ok=True)  # must exist before the FileHandler opens the report
    logger = logging.getLogger(__name__)
    report_path = VALIDATION_DIR / "validation_report.txt"
    handler = logging.FileHandler(report_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    folders_to_create = [
        VALIDATION_DIR,
        DATA_ROOT,
        DATASETS_ROOT,
        DATASETS_ROOT / "tier1" / "sync",
        DATASETS_ROOT / "tier1" / "async",
        DATASETS_ROOT / "tier2" / "sync",
        DATASETS_ROOT / "tier2" / "async",
        DATASTREAM_PATH,
        FUTURES_PATH,
        EQUITIES_PATH,
        FX_PATH,
        ECONOMICS_PATH,
        COMPUSTAT_PATH,
        TICKHISTORY_PATH,
    ]
    if _mode == "async-only":
        # No sync half is produced in this mode -- don't advertise a tree that
        # will never be filled.
        _sync_folders = {DATASETS_ROOT / "tier1" / "sync", DATASETS_ROOT / "tier2" / "sync"}
        folders_to_create = [f for f in folders_to_create if f not in _sync_folders]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    # resolve_mode()'s declared return type is `str` (it is a general-purpose
    # validator, not scoped to build's own Literal), but it only ever returns
    # "full" or "async-only" -- enforced dynamically by main()'s own
    # _validate_mode() check immediately below, so this narrows a value that is
    # already runtime-checked rather than asserting past a real gap.
    main(cast(Literal["full", "async-only"], _mode))

    logger.removeHandler(handler)
    handler.close()
