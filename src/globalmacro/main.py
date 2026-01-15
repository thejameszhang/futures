import logging
import math
import os
import textwrap
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import polars as pl
from utils.config import load_config
from utils.paths import *
from utils.splice import SPLICING_MAP

LOG_WIDTH = 88


def log_section(title: str) -> None:
    logger.info("")
    logger.info("=" * LOG_WIDTH)
    logger.info(title)
    logger.info("=" * LOG_WIDTH)


def log_subsection(title: str) -> None:
    logger.info("")
    logger.info(title)
    logger.info("-" * len(title))


def log_kv(label: str, value: str) -> None:
    logger.info(f"{label:<28} {value}")


def format_date(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    return str(value)


def log_wrapped_list(label: str, items: Sequence[str], indent: int = 2, width: int = 110) -> None:
    padding = " " * indent
    if not items:
        logger.info(f"{padding}{label}: None")
        return
    content = ", ".join(items)
    prefix = f"{padding}{label}: "
    wrap_width = max(20, width - len(prefix))
    lines = textwrap.wrap(content, width=wrap_width)
    for idx, line in enumerate(lines):
        if idx == 0:
            logger.info(prefix + line)
        else:
            logger.info(" " * len(prefix) + line)


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


def is_present_expr(col: str) -> pl.Expr:
    numeric_is_nan = (
        pl.col(col)
        .cast(pl.Float64, strict=False)
        .is_nan()
        .fill_null(False)
    )
    return pl.col(col).is_not_null() & ~numeric_is_nan


def read_csv(
    path: Path,
    *,
    date_col: str | None = "date",
    schema_overrides: Dict[str, Any] | None = None,
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


def read_excel(path: Path, *, skip_rows: int = 0) -> pl.DataFrame:
    if skip_rows:
        df = pl.read_excel(path, read_options={"skip_rows": skip_rows}, infer_schema_length=10000)
    else:
        df = pl.read_excel(path, infer_schema_length=10000)
    return drop_redundant_date_columns(df)


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


def first_valid_date(df: pl.DataFrame, col: str, *, date_col: str = "date") -> Any:
    return df.select(pl.col(date_col).filter(is_present_expr(col)).min()).to_series().item()


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


def compute_monthly_returns(daily_ret_df: pl.DataFrame) -> pl.DataFrame:
    df = daily_ret_df.sort("date").with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
    asset_cols = [col for col in df.columns if col not in {"date", "year_month"}]
    agg_exprs = [pl.col("date").max().alias("date")]
    for col in asset_cols:
        non_null_count = is_present_expr(col).sum()
        product_expr = (pl.col(col).cast(pl.Float64).fill_null(0).fill_nan(0) + 1).product() - 1
        agg_exprs.append(pl.when(non_null_count < 15).then(None).otherwise(product_expr).alias(col))
    monthly = df.group_by("year_month").agg(agg_exprs).sort("date")
    return monthly


def compare_tick_vs(
    tick_df: pl.DataFrame,
    other_df: pl.DataFrame,
    *,
    freq: str,
    threshold: float,
    figure_path: Path,
    other_label: str = "Datastream",
) -> List[str]:
    tick_df = coerce_numeric_data(tick_df)
    other_df = coerce_numeric_data(other_df)

    if freq == "monthly":
        tick_df = tick_df.with_columns(pl.col("date").dt.truncate("1mo").alias("period")).drop("date")
        other_df = other_df.with_columns(pl.col("date").dt.truncate("1mo").alias("period")).drop("date")
        join_col = "period"
    elif freq == "daily":
        join_col = "date"
    else:
        raise ValueError(f"Unsupported freq: {freq}")

    symbols = sorted(set(tick_df.columns).intersection(other_df.columns) - {join_col})
    if not symbols:
        return ["No shared symbols between datasets."]

    results = []
    for symbol in symbols:
        if symbol == "year_month":
            continue
        tick_col = f"{symbol}_tick"
        other_col = f"{symbol}_{freq}"

        tick_series = tick_df.select([join_col, pl.col(symbol).alias(tick_col)])
        other_series = other_df.select([join_col, pl.col(symbol).alias(other_col)])

        if freq == "monthly":
            comparison = other_series.join(tick_series, on=join_col, how="left").sort(join_col)
        else:
            comparison = full_join_on_date(
                tick_series,
                other_series,
                date_col=join_col,
                suffix=f"_{freq}",
            ).sort(join_col)

        valid_expr = is_present_expr(tick_col) & is_present_expr(other_col)
        valid_count = comparison.select(valid_expr.sum()).to_series().item()
        if not valid_count:
            continue

        comparison = comparison.with_row_index("row_idx")
        first_valid = comparison.filter(valid_expr).select(pl.col("row_idx").min()).to_series().item()
        comparison = comparison.filter(pl.col("row_idx") >= first_valid).drop("row_idx")
        if comparison.is_empty():
            continue

        comparison = comparison.with_columns(
            [
                pl.when(valid_expr)
                .then(pl.col(tick_col) - pl.col(other_col))
                .otherwise(None)
                .alias("diff"),
                pl.when(valid_expr)
                .then((pl.col(tick_col) - pl.col(other_col)).abs())
                .otherwise(None)
                .alias("abs_diff"),
            ]
        )

        filtered = comparison.filter(valid_expr)
        if filtered.is_empty():
            continue

        stats = filtered.select(
            [
                (pl.col(tick_col) - pl.col(other_col)).mean().alias("mean_diff"),
                (pl.col(tick_col) - pl.col(other_col)).abs().mean().alias("mean_abs_diff"),
                pl.corr(tick_col, other_col).alias("corr"),
                pl.col(tick_col).mean().alias("mean_tick"),
                pl.col(tick_col).std().alias("std_tick"),
                pl.col(other_col).mean().alias("mean_other"),
                pl.col(other_col).std().alias("std_other"),
            ]
        ).row(0)

        results.append(
            {
                "symbol": symbol,
                "comparison": comparison,
                "mean_diff": 100 * (stats[0] if stats[0] is not None else float("nan")),
                "mean_abs_diff": 100 * (stats[1] if stats[1] is not None else float("nan")),
                "correlation": stats[2],
                "mean_return_tick": 100 * (stats[3] if stats[3] is not None else float("nan")),
                "std_return_tick": 100 * (stats[4] if stats[4] is not None else float("nan")),
                "mean_return_other": 100 * (stats[5] if stats[5] is not None else float("nan")),
                "std_return_other": 100 * (stats[6] if stats[6] is not None else float("nan")),
            }
        )

    if not results:
        return ["No valid comparisons found."]

    results.sort(
        key=lambda r: (float("-inf") if r["correlation"] is None else r["correlation"]),
        reverse=True,
    )

    summary_lines = []
    n_symbols = len(results)
    n_cols = 3
    n_rows = int(math.ceil(n_symbols / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), sharex=False)
    if n_symbols == 1:
        axes = [axes]
    elif hasattr(axes, "flatten"):
        axes = axes.flatten()

    for idx, res in enumerate(results):
        ax = axes[idx]
        comparison = res["comparison"]
        symbol = res["symbol"]
        tick_col = f"{symbol}_tick"
        other_col = f"{symbol}_{freq}"

        comparison = comparison.with_columns(
            [
                pl.when(is_missing_expr(tick_col))
                .then(None)
                .otherwise(pl.col(tick_col).fill_null(0).fill_nan(0).log1p().cum_sum())
                .alias("cumlogret_tick"),
                pl.when(is_missing_expr(other_col))
                .then(None)
                .otherwise(pl.col(other_col).fill_null(0).fill_nan(0).log1p().cum_sum())
                .alias("cumlogret_other"),
            ]
        )

        x_vals = comparison[join_col].to_list()
        ax.plot(x_vals, comparison["cumlogret_other"].to_list(), label=other_label, color="red")
        ax.plot(x_vals, comparison["cumlogret_tick"].to_list(), label="TickHistory", color="blue")
        ax.set_ylabel("Cumulative Log Return")
        corr_value = res["correlation"]
        corr_display = f"{corr_value:.3f}" if corr_value is not None else "nan"
        ax.set_title(
            (
                f"{symbol}, Mean diff: {res['mean_diff']:.3f}%, "
                f"Mean abs diff: {res['mean_abs_diff']:.3f}%, Corr: {corr_display}\n"
                f"Tick μ={res['mean_return_tick']:.3f}%, σ={res['std_return_tick']:.3f}%; "
                f"{other_label} μ={res['mean_return_other']:.3f}%, σ={res['std_return_other']:.3f}%"
            ),
            fontsize=11,
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
        summary_lines.append(
            f"{symbol}: mean_diff={res['mean_diff']:.3f}%, "
            f"mean_abs_diff={res['mean_abs_diff']:.3f}%, "
            f"corr={corr_display}"
        )

    for idx in range(n_symbols, len(axes)):
        axes[idx].set_visible(False)

    for idx in range(max(0, n_rows - 1) * n_cols, min(n_rows * n_cols, n_symbols)):
        axes[idx].set_xlabel("Date")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(figure_path, format="pdf")
    plt.close(fig)
    return summary_lines


def find_gaps(
    df: pl.DataFrame,
    col: str,
    min_gap_size: int,
    *,
    freq: str = "daily",
    dates: List[Any] | None = None,
) -> List[Dict[str, Any]]:
    if freq not in {"daily", "monthly"}:
        raise ValueError(f"Unsupported freq: {freq}")
    if dates is None:
        dates = df["date"].to_list()
    values = df[col].to_list()
    gaps = []
    in_gap = False
    start_idx = 0
    for idx, value in enumerate(values):
        is_na = value is None or (isinstance(value, float) and math.isnan(value))
        if is_na and not in_gap:
            start_idx = idx
            in_gap = True
        if not is_na and in_gap:
            end_idx = idx - 1
            size = end_idx - start_idx + 1
            gaps.append({"start": dates[start_idx], "end": dates[end_idx], "size": size})
            in_gap = False
    if in_gap:
        end_idx = len(values) - 1
        size = end_idx - start_idx + 1
        gaps.append({"start": dates[start_idx], "end": dates[end_idx], "size": size})

    if gaps and gaps[0]["start"] == dates[0]:
        gaps = gaps[1:]

    return [gap for gap in gaps if gap["size"] >= min_gap_size]


def log_dataset_stats(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Dataset coverage summary")
    synced_start = synced.select(pl.col("date").min()).to_series().item()
    synced_end = synced.select(pl.col("date").max()).to_series().item()
    async_start = asynced.select(pl.col("date").min()).to_series().item()
    async_end = asynced.select(pl.col("date").max()).to_series().item()
    log_kv("Synced rows:", synced.height)
    log_kv("Synced symbols:", len(synced.columns) - 1)
    log_kv("Synced range:", f"{format_date(synced_start)} to {format_date(synced_end)}")
    log_kv("Asynced rows:", asynced.height)
    log_kv("Asynced symbols:", len(asynced.columns) - 1)
    log_kv("Asynced range:", f"{format_date(async_start)} to {format_date(async_end)}")


def validate_monthly_return_comparison(synced: pl.DataFrame, asynced: pl.DataFrame) -> pl.DataFrame:
    log_section("Monthly returns: TickHistory vs Datastream")
    synced_monthly = compute_monthly_returns(synced)
    asynced_monthly = compute_monthly_returns(asynced)
    asynced_monthly = drop_all_null_rows(asynced_monthly)
    comparison_pdf = VALIDATION_DIR / "sync_vs_async_monthly_ret_comparison.pdf"
    for line in compare_tick_vs(
        synced_monthly,
        asynced_monthly,
        freq="monthly",
        threshold=0.03,
        figure_path=comparison_pdf,
    ):
        logger.info(f"  {line}")


def validate_daily_return_comparison(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Daily returns: TickHistory vs Datastream")
    async_cols = [col for col in synced.columns if col != "date" and col in asynced.columns]
    tick_daily = synced.select(["date"] + async_cols)
    async_daily = asynced.select(["date"] + async_cols)
    async_daily = drop_all_null_rows(async_daily)

    comparison_pdf = VALIDATION_DIR / "sync_vs_async_daily_ret_comparison.pdf"
    for line in compare_tick_vs(
        tick_daily,
        async_daily,
        freq="daily",
        threshold=0.03,
        figure_path=comparison_pdf,
    ):
        logger.info(f"  {line}")


def validate_missing_symbols(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Missing symbols between datasets")
    synced_cols = [col for col in synced.columns if col != "date"]
    async_cols = [col for col in asynced.columns if col != "date"]
    missing_in_async = sorted([symbol for symbol in synced_cols if symbol not in async_cols])
    missing_in_synced = sorted([symbol for symbol in async_cols if symbol not in synced_cols])
    log_wrapped_list("Synced only", missing_in_async, indent=2)
    log_wrapped_list("Asynced only", missing_in_synced, indent=2)


def validate_daily_symbol_counts_plot(synced: pl.DataFrame, dataset_label: str = "synced", freq: str = "daily") -> None:
    if freq not in {"daily", "monthly"}:
        raise ValueError(f"Unsupported freq: {freq}")
    log_section(f"{freq.capitalize()} symbol counts plot for {dataset_label}")
    data_cols = [col for col in synced.columns if col != "date"]
    counts = synced.select(
        pl.sum_horizontal(*[is_present_expr(col) for col in data_cols]).alias("count")
    )["count"].to_list()
    dates = synced["date"].to_list()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, counts, linewidth=2)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Symbols")
    ax.set_title(f"Available Assets per {freq.capitalize()} ({dataset_label.capitalize()} Dataset)")
    fig.autofmt_xdate()
    counts_pdf = VALIDATION_DIR / f"{dataset_label}_symbol_counts.pdf"
    fig.savefig(counts_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  {freq.capitalize()} symbol counts plot saved to {counts_pdf.name}.")


def validate_availability_summary(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Earliest and latest observation summary")
    for col in [c for c in synced.columns if c != "date"]:
        synced_first = first_valid_date(synced, col)
        synced_last = last_valid_date(synced, col)
        if col in asynced.columns:
            async_first = first_valid_date(asynced, col)
            async_last = last_valid_date(asynced, col)
        else:
            async_first = async_last = None
        synced_span = f"{format_date(synced_first)} to {format_date(synced_last)}"
        async_span = f"{format_date(async_first)} to {format_date(async_last)}" if async_first else "N/A"
        logger.info(f"{col:<10} TickHistory {synced_span} | Datastream {async_span}")


def validate_gap_analysis(
    synced: pl.DataFrame,
    asynced: pl.DataFrame,
    min_gap_size: int = 5,
    *,
    freq: str = "daily",
) -> None:
    if freq not in {"daily", "monthly"}:
        raise ValueError(f"Unsupported freq: {freq}")

    gap_unit = "months" if freq == "monthly" else "trading days"
    log_section(f"Gap analysis ({freq} gaps >= {min_gap_size} {gap_unit})")

    if freq == "monthly":
        async_dates = asynced["date"].to_list()
        for col in [c for c in asynced.columns if c != "date"]:
            daily_gaps = find_gaps(asynced, col, min_gap_size, freq=freq, dates=async_dates)
            if not daily_gaps:
                continue
            logger.info(f"{col}:")
            for gap in daily_gaps:
                logger.info(
                    f"  Datastream gap: {format_date(gap['start'])} to {format_date(gap['end'])} ({gap['size']} {gap_unit})"
                )
        return

    async_dates = asynced["date"].to_list()
    synced_dates = synced["date"].to_list()
    for col in [c for c in synced.columns if c != "date"]:
        if col not in asynced.columns:
            continue
        daily_gaps = find_gaps(asynced, col, min_gap_size, freq=freq, dates=async_dates)
        synced_gaps = find_gaps(synced, col, min_gap_size, freq=freq, dates=synced_dates)
        if not daily_gaps and not synced_gaps:
            continue
        logger.info(f"{col}:")
        if daily_gaps:
            for gap in daily_gaps:
                logger.info(
                    f"  Datastream gap: {format_date(gap['start'])} to {format_date(gap['end'])} ({gap['size']} {gap_unit})"
                )
        else:
            logger.info("  Datastream gap: None")
        if synced_gaps:
            for gap in synced_gaps:
                logger.info(
                    f"  TickHistory gap: {format_date(gap['start'])} to {format_date(gap['end'])} ({gap['size']} {gap_unit})"
                )
        else:
            logger.info("  TickHistory gap: None")


def validate_missing_dates(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Missing dates between datasets")
    cutoff = date(1996, 1, 4)
    synced_dates = set(synced["date"].to_list())
    async_dates = set(asynced.filter(pl.col("date") >= pl.lit(cutoff))["date"].to_list())
    missing_dates_tick = sorted(async_dates - synced_dates)
    missing_dates_async = sorted(synced_dates - async_dates)
    logger.info(f"Datastream-only dates: {len(missing_dates_tick)}")
    log_wrapped_list(
        "Dates",
        [format_date(value) for value in missing_dates_tick],
        indent=2,
    )
    logger.info(f"TickHistory-only dates: {len(missing_dates_async)}")
    log_wrapped_list(
        "Dates",
        [format_date(value) for value in missing_dates_async],
        indent=2,
    )


def validate_missing_returns_by_symbol(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Missing returns by symbol")
    common_cols = [col for col in asynced.columns if col != "date" and col in synced.columns]
    if not common_cols:
        logger.info("No overlapping symbols.")
        return
    cutoff = date(1996, 1, 4)
    daily_common = asynced.filter(pl.col("date") >= pl.lit(cutoff))
    synced_common = synced.filter(pl.col("date") >= pl.lit(cutoff))

    for col in common_cols:
        aligned = full_join_on_date(
            synced_common.select(["date", col]),
            daily_common.select(["date", col]),
            date_col="date",
            suffix="_async",
        )
        missing_in_synced = aligned.filter(
            is_missing_expr(col) & is_present_expr(f"{col}_async")
        )["date"].to_list()
        missing_in_async = aligned.filter(
            is_present_expr(col) & is_missing_expr(f"{col}_async")
        )["date"].to_list()
        if not missing_in_synced and not missing_in_async:
            continue
        logger.info(f"{col}:")
        log_wrapped_list(
            "Datastream-only dates",
            [format_date(value) for value in missing_in_synced],
            indent=4,
        )
        log_wrapped_list(
            "TickHistory-only dates",
            [format_date(value) for value in missing_in_async],
            indent=4,
        )


def validate_extreme_returns(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Extreme returns check")
    for col in [c for c in synced.columns if c != "date"]:
        synced_extreme = synced.filter(pl.col(col).abs() > 0.2).select(["date", col])
        async_extreme = asynced.filter(pl.col(col).abs() > 0.2).select(["date", col])
        if synced_extreme.is_empty() and async_extreme.is_empty():
            logger.info(f"{col}: No extreme returns found.")
            continue
        logger.info(f"{col}:")
        if not synced_extreme.is_empty():
            logger.info(f"  Synced ({synced_extreme.height})")
            for row in synced_extreme.iter_rows():
                logger.info(f"    {format_date(row[0])}: {row[1]}")
        else:
            logger.info("  Synced: None")
        if not async_extreme.is_empty():
            logger.info(f"  Async ({async_extreme.height})")
            for row in async_extreme.iter_rows():
                logger.info(f"    {format_date(row[0])}: {row[1]}")
        else:
            logger.info("  Async: None")


def validate_column_alignment(synced: pl.DataFrame, asynced: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    synced_cols = sorted([col for col in synced.columns if col != "date"])
    async_cols = sorted([col for col in asynced.columns if col != "date"])
    assert synced_cols == async_cols, f"Daily columns: {async_cols} != synced columns: {synced_cols}"
    synced = synced.select(["date"] + synced_cols)
    asynced = asynced.select(["date"] + async_cols)
    return synced, asynced


def validate_weekends(synced: pl.DataFrame, asynced: pl.DataFrame) -> None:
    log_section("Weekends in datasets")
    synced_weekends = [d for d in synced["date"].to_list() if d.weekday() >= 5]
    async_weekends = [d for d in asynced["date"].to_list() if d.weekday() >= 5]
    if synced_weekends:
        logger.info(f"Synced dataset contains weekend dates ({len(synced_weekends)}).")
    if async_weekends:
        logger.info(f"Asynced dataset contains weekend dates ({len(async_weekends)}).")
    if not synced_weekends and not async_weekends:
        logger.info("No weekends in datasets.")

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


def load_vxo(rf: pl.DataFrame) -> pl.DataFrame:
    vxo = read_excel(DATA_ROOT / "misc" / "vxoarchive.xls", skip_rows=2)
    vxo_columns = vxo.columns
    if len(vxo_columns) < 5:
        raise ValueError(f"Unexpected VXO format: expected 5 columns, got {len(vxo_columns)}")
    vxo = vxo.rename(
        {
            vxo_columns[0]: "date",
            vxo_columns[1]: "Open",
            vxo_columns[2]: "High",
            vxo_columns[3]: "Low",
            vxo_columns[4]: "Close",
        }
    )
    vxo = ensure_date(vxo, "date").sort("date").join(rf, on="date", how="left")
    vxo = vxo.select(["date", "Close", "rf"]).with_columns(
        (pl.col("Close").cast(pl.Float64).pct_change() - pl.col("rf")).alias("vxo_ret_rf")
    )
    return vxo.filter(pl.col("date") <= pl.date(1990, 1, 4)).select(["date", "vxo_ret_rf"])


def load_vix(rf: pl.DataFrame) -> pl.DataFrame:
    vix = pl.read_csv(DATA_ROOT / "misc" / "VIX_History.csv", try_parse_dates=False).rename(
        {"DATE": "date", "OPEN": "vix_ret_rf_open", "CLOSE": "vix_ret_rf_close"}
    )
    vix = vix.with_columns(pl.col("date").str.strptime(pl.Date, "%m/%d/%Y", strict=False))
    vix = vix.sort("date").join(rf, on="date", how="left")
    vix = vix.select(["date", "vix_ret_rf_open", "vix_ret_rf_close", "rf"]).with_columns(
        (pl.col("vix_ret_rf_open").cast(pl.Float64).pct_change() - pl.col("rf")).alias("vix_ret_rf_open"),
        (pl.col("vix_ret_rf_close").cast(pl.Float64).pct_change() - pl.col("rf")).alias("vix_ret_rf_close"),
    )
    return vix.drop("rf")


def load_async_dataset(tier: int = 1) -> pl.DataFrame:
    daily_ct = read_csv(DATASETS_ROOT / f"tier{tier}" / "async" / "daily_ret_1_CT.csv")
    daily_cs = read_csv(DATASETS_ROOT / f"tier{tier}" / "async" / "daily_ret_1_CS.csv")
    return coerce_numeric_data(combine_first_on_date(daily_ct, daily_cs))


def add_vix_vxo_to_async(asynced: pl.DataFrame, vxo: pl.DataFrame, vix: pl.DataFrame) -> pl.DataFrame:
    asynced = asynced.join(vxo, on="date", how="left").join(
        vix.select(["date", "vix_ret_rf_close"]), on="date", how="left"
    )
    asynced = asynced.with_columns(
        pl.coalesce([pl.col("VX"), pl.col("vix_ret_rf_close"), pl.col("vxo_ret_rf")]).alias("VX")
    ).drop(["vix_ret_rf_close", "vxo_ret_rf"])
    return asynced


def load_sectors_async() -> pl.DataFrame:
    sectors_async = read_csv(
        DATA_ROOT / "misc" / "daily_ind_gics.csv",
        schema_overrides={"gics": pl.Utf8},
    ).sort("date")
    return coerce_numeric_data(sectors_async.pivot(values="ret_vw_cap", index="date", on="gics").sort("date"))


def load_synthetic_returns(rf: pl.DataFrame, equities: List[Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    synthetic_fx_returns = read_csv(FX_PATH / "synthetic_fx_returns.csv").select(
        ["date", "NOK", "SEK", "6N", "6A"]
    )
    synthetic_fx_returns = coerce_numeric_data(drop_all_null_rows(synthetic_fx_returns))

    spot_equity_returns = read_csv(EQUITIES_PATH / "spot_equity_returns.csv")
    spot_equity_returns = coerce_numeric_data(drop_all_null_rows(spot_equity_returns).join(rf, on="date", how="left"))

    # Splice equity index returns together
    for equity in equities:
        if equity.dsindexcode and len(equity.dsindexcode) > 1:
            inactive = equity.dsindexmnem[0]
            if inactive not in spot_equity_returns.columns or equity.symbol not in spot_equity_returns.columns:
                logger.warning(f"Missing {inactive} or {equity.symbol} in spot_equity_returns")
                continue
            spot_equity_returns = spot_equity_returns.with_columns(
                (
                    pl.coalesce(pl.col(equity.symbol), pl.col(inactive)) - pl.col("rf")
                ).alias(equity.symbol)
            ).drop(inactive)
        else:
            if equity.symbol not in spot_equity_returns.columns:
                logger.warning(f"Missing {equity.symbol} in spot_equity_returns")
                continue

    spot_equity_returns = coerce_numeric_data(spot_equity_returns.drop("rf"))
    synthetic_returns = coerce_numeric_data(
        outer_join_on_date(synthetic_fx_returns, spot_equity_returns).sort("date")
    )
    return synthetic_returns, synthetic_returns.clone()


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


def build_synced_dataset(vix_open: pl.DataFrame) -> pl.DataFrame:
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
        DATA_ROOT / "misc" / "daily_ind_gics_synced.csv",
        schema_overrides={"gics": pl.Utf8},
    ).sort("date")
    sectors = coerce_numeric_data(sectors.pivot(values="ret_vw_cap", index="date", on="gics").sort("date"))

    synced = bonds
    for df in [commodities, currencies, nonus, us, volatilities, stirs, sectors, vix_open]:
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


def filter_dataset_by_monthly_returns(dataset: pl.DataFrame) -> pl.DataFrame:
    dataset_monthly = compute_monthly_returns(dataset)
    dataset_monthly_cols = [col for col in dataset.columns if col not in {"date", "year_month"}]
    if dataset_monthly_cols:
        first_valid_months = dataset_monthly.select(
            [pl.col("year_month").filter(is_present_expr(col)).min().alias(col) for col in dataset_monthly_cols]
        ).row(0)
        cutoff_by_col = dict(zip(dataset_monthly_cols, first_valid_months))
        cutoff_exprs = []
        for col in dataset_monthly_cols:
            cutoff = cutoff_by_col[col]
            if cutoff is None:
                expr = pl.when(pl.lit(False)).then(pl.col(col)).otherwise(None).alias(col)
            else:
                expr = pl.when(pl.col("date") >= pl.lit(cutoff)).then(pl.col(col)).otherwise(None).alias(col)
            cutoff_exprs.append(expr)
        dataset = dataset.with_columns(cutoff_exprs)
    dataset_monthly = dataset_monthly.drop("year_month")
    return dataset, dataset_monthly


def main() -> None:
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    equities = [future for future in futures if future.dsindexcode is not None]
    rf = load_rf()
    synthetic_returns, synthetic_returns_synced = load_synthetic_returns(rf, equities)

    log_section("Dataset preparation")
    asynced = load_async_dataset(tier=1)
    asynced = keep_after_date(asynced, "FBTP", date(2009, 9, 14), inclusive=True)
    asynced_tier2 = load_async_dataset(tier=2)

    vxo = load_vxo(rf)
    vix = load_vix(rf)
    asynced = add_vix_vxo_to_async(asynced, vxo, vix)
    sectors_async = load_sectors_async()
    asynced = asynced.join(sectors_async, on="date", how="left")
    asynced = splice_synthetic_returns(asynced, synthetic_returns, "Datastream")
    asynced = fill_asynced_gaps_with_synthetic_returns(asynced, synthetic_returns)
    asynced = drop_all_null_rows(asynced).filter(pl.col("date") <= pl.date(2025, 12, 31))

    vix_open = vix.filter(pl.col("date") >= pl.date(1996, 1, 1)).select(["date", "vix_ret_rf_open"])
    synced = build_synced_dataset(vix_open)
    synced = splice_synthetic_returns(synced, synthetic_returns_synced, "TickHistory", skip_if_before=date(1996, 1, 4))
    synced = drop_all_null_rows(synced).filter(
        (pl.col("date") >= pl.date(1996, 1, 4)) & (pl.col("date") <= pl.date(2025, 12, 31))
    )

    asynced, asynced_monthly = filter_dataset_by_monthly_returns(asynced)

    log_dataset_stats(synced, asynced)
    validate_missing_symbols(synced, asynced)
    validate_daily_symbol_counts_plot(synced, "synced")
    validate_daily_symbol_counts_plot(asynced, "asynced")
    validate_daily_symbol_counts_plot(asynced_monthly, "asynced_monthly")
    validate_availability_summary(synced, asynced)
    validate_gap_analysis(synced, asynced, freq="daily")
    validate_gap_analysis(synced, asynced_monthly, min_gap_size=1, freq="monthly")
    validate_missing_dates(synced, asynced)
    validate_missing_returns_by_symbol(synced, asynced)
    validate_extreme_returns(synced, asynced)
    validate_weekends(synced, asynced)
    synced, asynced = validate_column_alignment(synced, asynced)
    validate_daily_return_comparison(synced, asynced)
    validate_monthly_return_comparison(synced, asynced)

    asynced_monthly.write_csv(DATASETS_ROOT / "tier1" / "async" / "async_monthly.csv")
    asynced.write_csv(DATASETS_ROOT / "tier1" / "async" / "async_daily.csv")
    synced.write_csv(DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv")


if __name__ == "__main__":
    VALIDATION_DIR = GLOBALMACRO_ROOT / "validation"
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
        DATASTREAM_PATH,
        FUTURES_PATH,
        EQUITIES_PATH,
        FX_PATH,
        ECONOMICS_PATH,
        COMMODITIES_PATH,
        COMPUSTAT_PATH,
        TICKHISTORY_PATH,
    ]
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)

    main()

    logger.removeHandler(handler)
    handler.close()
