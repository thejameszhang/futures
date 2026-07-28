# src/globalmacro/validation/synthetic.py
"""Shared machinery for the two synthetic-return exercises.

Both ask the same question: over the window where the REAL future exists, does the
synthetic series we splice in ahead of it actually track that future?

Two things are easy to get wrong and are pinned here:

1. The window. Below a symbol's cutoff the shipped panel IS the synthetic (that is what
   splicing means), so correlating there is tautological and inflates the result. The
   cutoff therefore comes from the PRE-SPLICE panel; taking it from the shipped panel
   yields the synthetic's own start date instead.

2. The monthly aggregation. Use build.compute_monthly_returns and nothing else. A
   hand-rolled `(x + 1).product() - 1` looks equivalent but is not: polars' product()
   ignores nulls and returns 1 for an all-null month, so a month with NO synthetic data
   silently becomes a ZERO RETURN. Where a synthetic starts long after its future does,
   that manufactures years of fake zeros -- it is what once made 6E look like 0.62 when
   it is really 0.998.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from math import isfinite

import polars as pl

from globalmacro.build import (
    build_synced_dataset,
    compute_monthly_returns,
    first_valid_date,
    keep_after_date,
    load_async_dataset,
)
from globalmacro.utils.paths import DATASETS_ROOT

_MIN_MONTHS = 24


@lru_cache(maxsize=4)
def pre_splice_panel(dataset: str) -> pl.DataFrame:
    """The futures panel as it exists immediately before splice_synthetic_returns.

    Built at tier 2, exactly as build.main() does; Tier 1 is a column subset of it.
    This is the ONLY correct source of a symbol's real-future cutoff.
    """
    if dataset == "async":
        asynced = load_async_dataset(tier=2)
        asynced = keep_after_date(asynced, "FBTP", date(2009, 9, 14), inclusive=True)
        asynced = keep_after_date(asynced, "PLN", date(2004, 8, 1), inclusive=True)
        asynced = keep_after_date(asynced, "6Z", date(1997, 5, 8), inclusive=True)
        return asynced
    if dataset == "sync":
        return build_synced_dataset(tier=2)
    raise ValueError(f"dataset must be 'async' or 'sync', got {dataset!r}")


def load_panel(path) -> pl.DataFrame:
    """Read a wide date-keyed CSV: parse the date, cast every other column to Float64.

    The one loader for every wide panel this package reads (shipped panels, the synthetic
    FX files, the spot FX panels). infer_schema_length=0 first, so a column that is empty
    for its first rows is not typed Utf8 and silently dropped by a later numeric filter.
    """
    df = pl.read_csv(str(path), infer_schema_length=0)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c != "date"]
    )


@lru_cache(maxsize=8)
def shipped_panel(dataset: str, tier: int = 1) -> pl.DataFrame:
    # Cached: validate calls these repeatedly, and rebuilding a tier-2 pre-splice panel from
    # raw is the single most expensive thing in the run. The frames are never mutated.
    return load_panel(DATASETS_ROOT / f"tier{tier}" / dataset / f"{dataset}_daily.csv")


def _valid(a: str, b: str) -> pl.Expr:
    """Both sides present AND finite.

    is_finite() is False for NaN and for +-inf, and the inf case is not hypothetical:
    tier2/async carries FKLI = inf on 2025-10-01 (a data defect upstream of this package).
    An inf that reaches pl.corr poisons it to NaN, grade() then drops NaN correlations, and
    the symbol would vanish from the median without a word. Screen it here instead.
    """
    return (
        pl.col(a).is_not_null()
        & pl.col(b).is_not_null()
        & pl.col(a).is_finite()
        & pl.col(b).is_finite()
    )


def _monthly_corr(window: pl.DataFrame) -> tuple[float | None, int]:
    """Correlate monthly returns. Each side aggregated independently by build's own rule
    (which nulls any month with < 15 observations), then joined on year_month."""
    mx = compute_monthly_returns(window.select("date", "x"))
    my = compute_monthly_returns(window.select("date", "y"))
    m = (
        mx.select("year_month", "x")
        .join(my.select("year_month", "y"), on="year_month", how="inner")
        .filter(_valid("x", "y"))
    )
    if m.height < _MIN_MONTHS:
        return None, m.height
    return m.select(pl.corr("x", "y")).item(), m.height


def daily_corr(window: pl.DataFrame, col: str = "x") -> float | None:
    w = window.filter(_valid(col, "y"))
    if w.height < 100:
        return None
    return w.select(pl.corr(col, "y")).item()


def _windowed(
    synth: pl.DataFrame, pre: pl.DataFrame, ship: pl.DataFrame, symbol: str
) -> tuple[date, pl.DataFrame] | None:
    """(cutoff, window_df[date, x=synth, y=real-future]) over date >= cutoff, or None if
    the symbol has no cutoff.

    THE single source of truth for the real-future window: synthetic_correlations (the
    grade) and synthetic_pairs (the comparison.pdf plot) both call this, so the plot can
    never drift from what was graded. See the module docstring for why the cutoff must
    come from `pre`, never `ship`.
    """
    if symbol not in ship.columns or symbol not in pre.columns:
        return None
    cutoff = first_valid_date(pre, symbol)
    if cutoff is None:
        return None
    joined = synth.select("date", pl.col(symbol).alias("x")).join(
        ship.select("date", pl.col(symbol).alias("y")), on="date", how="inner"
    )
    return cutoff, joined.filter(pl.col("date") >= cutoff)


def synthetic_correlations(
    synth: pl.DataFrame,
    pre: pl.DataFrame,
    ship: pl.DataFrame,
    alt: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per symbol present in both `synth` and `ship`.

    correlation   -- monthly, over the real-future window. This is what grade() reads.
    corr_daily    -- same window, daily. Reported, never graded.
    corr_daily_alt-- daily correlation of the ALTERNATIVE synthetic (`alt`): the other FX
                     source for (a), the other alignment for (b). The discriminating test.
    mean_gap_bp   -- mean(synth) - mean(future), bp/day. Correlation is blind to level drift.
    n_backfilled  -- synthetic observations below the cutoff that reached the shipped panel.
    used          -- n_backfilled > 0, i.e. the synthetic actually ships. Boolean.
    """
    rows = []
    for symbol in sorted(c for c in synth.columns if c != "date"):
        windowed = _windowed(synth, pre, ship, symbol)
        if windowed is None:
            continue
        cutoff, window = windowed

        if alt is not None and symbol in alt.columns:
            window = window.join(
                alt.select("date", pl.col(symbol).alias("z")), on="date", how="left"
            )
        else:
            window = window.with_columns(pl.lit(None, dtype=pl.Float64).alias("z"))

        # Backfill is counted against the SHIPPED panel: it starts 1996-01-04 for sync,
        # so a pre-splice grid row at 1996-01-03 cannot conjure a phantom backfill. Computed
        # below the cutoff, i.e. outside the window _windowed returns -- so it is rejoined
        # here rather than sourced from the window itself.
        below_cutoff = synth.select("date", pl.col(symbol).alias("x")).join(
            ship.select("date", pl.col(symbol).alias("y")), on="date", how="inner"
        )
        n_backfilled = below_cutoff.filter(
            (pl.col("date") < cutoff) & _valid("x", "y")
        ).height

        corr_m, n_months = _monthly_corr(window)
        # NaN is not None. A NaN correlation emitted with used=True would be filtered out
        # again by grade(), so the symbol would leave the median with nothing said about it.
        # Drop it here, where it is at least absent from correlations.csv too.
        if corr_m is None or not isfinite(corr_m):
            continue

        valid = window.filter(_valid("x", "y"))
        mean_gap_bp = (
            valid.select(pl.col("x").mean()).item() - valid.select(pl.col("y").mean()).item()
        ) * 1e4

        rows.append(
            {
                "instrument": symbol,
                "correlation": corr_m,
                "n_obs": n_months,
                "corr_daily": daily_corr(window, "x"),
                "corr_daily_alt": daily_corr(window, "z"),
                "mean_gap_bp": mean_gap_bp,
                "n_backfilled": n_backfilled,
                "used": n_backfilled > 0,
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "instrument": pl.Utf8, "correlation": pl.Float64, "n_obs": pl.Int64,
                "corr_daily": pl.Float64, "corr_daily_alt": pl.Float64,
                "mean_gap_bp": pl.Float64, "n_backfilled": pl.Int64, "used": pl.Boolean,
            }
        )
    return pl.DataFrame(rows).sort("correlation", descending=True)


_PAIRS_SCHEMA = {
    "instrument": pl.Utf8, "name": pl.Utf8, "month": pl.Date,
    "ours": pl.Float64, "theirs": pl.Float64,
}


def synthetic_pairs(
    synth: pl.DataFrame,
    pre: pl.DataFrame,
    ship: pl.DataFrame,
    name_of: dict[str, str] | None = None,
) -> pl.DataFrame:
    """Long-form [instrument, name, month, ours, theirs] for comparison.pdf.

    Uses the SAME `_windowed` cutoff+window as synthetic_correlations, so the plot shows
    exactly the real-future window that was graded -- never the tautological backfill
    region below the cutoff. Monthly aggregation is compute_monthly_returns, not a
    hand-rolled product (see the module docstring: an all-null month must stay null).
    """
    name_of = name_of or {}
    frames = []
    for symbol in sorted(c for c in synth.columns if c != "date"):
        windowed = _windowed(synth, pre, ship, symbol)
        if windowed is None:
            continue
        _, window = windowed

        mx = compute_monthly_returns(window.select("date", "x"))
        my = compute_monthly_returns(window.select("date", "y"))
        joined = (
            mx.select(pl.col("year_month").alias("month"), pl.col("x").alias("ours"))
            .join(
                my.select(pl.col("year_month").alias("month"), pl.col("y").alias("theirs")),
                on="month", how="inner",
            )
            .with_columns(
                pl.lit(symbol).alias("instrument"),
                pl.lit(name_of.get(symbol, symbol)).alias("name"),
            )
        )
        frames.append(joined.select("instrument", "name", "month", "ours", "theirs"))

    if not frames:
        return pl.DataFrame(schema=_PAIRS_SCHEMA)
    return pl.concat(frames).cast(_PAIRS_SCHEMA)
