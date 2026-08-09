# src/globalmacro/validation/consistency.py
"""Exercise 2: async (Datastream) vs sync (TickHistory) consistency.

Two independent construction paths for the same instrument should agree. This is
the graded core of build.py's old compare_tick_vs (freq="monthly" path), without
the plotting. IMPORTANT: the two paths are NOT synced intraday, so a *daily*
correlation is meaningless (empirically median ~0.64, which would FAIL the 0.95
gate); aggregating to monthly washes the timing mismatch out (build's existing
monthly comparison yields median ~0.979). So Exercise 2 grades MONTHLY returns:
BOTH the sync and async daily series are compounded to monthly via
compute_monthly_returns before correlating (median ~0.9789, verified). Compounding
only async and correlating it against DAILY sync is WRONG — it fans one monthly
value across ~21 daily rows and collapses to median ~0.19. Tier 2 is the graded
universe.
"""
from __future__ import annotations

import polars as pl

# Pure, side-effect-free transform (importing build runs no heavy code: build has
# no module-level work outside the __main__ guard). compute_monthly_returns is the
# exact monthly aggregation build uses, so Exercise 2 matches build's numbers.
from globalmacro.build import compute_monthly_returns
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import DATASETS_ROOT, PROJECT_ROOT
from globalmacro.validation.base import Check


def _present(col: str) -> pl.Expr:
    return pl.col(col).is_not_null() & pl.col(col).is_not_nan()


def consistency_correlations(
    synced: pl.DataFrame, asynced_monthly: pl.DataFrame, freq: str = "monthly"
) -> pl.DataFrame:
    """Per-instrument correlation of sync vs async returns (compare_tick_vs core)."""
    def val(df: pl.DataFrame) -> pl.DataFrame:
        """Cast non-date columns to Float64 safely."""
        return df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c != "date"]
        )
    tick, other = val(synced), val(asynced_monthly)
    if freq == "monthly":
        tick = tick.with_columns(pl.col("date").dt.truncate("1mo").alias("period")).drop("date")
        other = other.with_columns(pl.col("date").dt.truncate("1mo").alias("period")).drop("date")
        join_col = "period"
    else:
        join_col = "date"
    symbols = sorted((set(tick.columns) & set(other.columns)) - {join_col, "year_month"})
    rows = []
    for sym in symbols:
        t = tick.select([join_col, pl.col(sym).alias("t")])
        o = other.select([join_col, pl.col(sym).alias("o")])
        comp = o.join(t, on=join_col, how="left").sort(join_col)   # build's join direction
        valid = _present("t") & _present("o")
        n = comp.select(valid.sum()).to_series().item()
        if not n:
            continue
        corr = comp.filter(valid).select(pl.corr("t", "o")).to_series().item()
        rows.append({"instrument": sym, "correlation": corr, "n_obs": int(n)})
    if not rows:
        return pl.DataFrame(schema={"instrument": pl.Utf8, "correlation": pl.Float64, "n_obs": pl.Int64})
    return pl.DataFrame(rows).sort("correlation", descending=True)




def _consistency_pairs() -> pl.DataFrame:
    """Tier-1 sync vs async, monthly, long-form for comparison.pdf. Independent of
    the Tier-2 graded path above."""
    def _load(path):
        df = pl.read_csv(str(path), infer_schema_length=0)
        df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
        return df.with_columns(
            [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c != "date"]
        )
    sync_m = compute_monthly_returns(_load(DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv"))
    async_m = compute_monthly_returns(_load(DATASETS_ROOT / "tier1" / "async" / "async_daily.csv"))
    # tier1.yaml names, plus the 11 GICS Select-Sector tickers that appear in the
    # datasets but not tier1.yaml (else their panels read "XAB (XAB)"). Sector
    # labels are build.py:32-44's GICS-code -> ticker comments.
    _SECTOR_NAMES = {
        "XAE": "Energy", "XAB": "Materials", "XAI": "Industrials",
        "XAY": "Consumer Discretionary", "XAP": "Consumer Staples",
        "XAV": "Health Care", "XAF": "Financials",
        "XAK": "Information Technology", "XAZ": "Communication Services",
        "XAU": "Utilities", "XAR": "Real Estate",
    }
    name_of = {str(f.symbol): f.name for f in load_config(PROJECT_ROOT / "tier1.yaml")}
    name_of.update(_SECTOR_NAMES)
    syms = sorted((set(sync_m.columns) & set(async_m.columns)) - {"date", "year_month"})
    frames = []
    for sym in syms:
        # Join on the canonical month key (year_month, month-truncated), NOT the
        # per-month max trading date, which can differ across the two calendars.
        s = sync_m.select(pl.col("year_month").alias("month"), pl.col(sym).alias("ours"))
        a = async_m.select(pl.col("year_month").alias("month"), pl.col(sym).alias("theirs"))
        j = s.join(a, on="month", how="inner").with_columns(
            pl.lit(sym).alias("instrument"),
            pl.lit(name_of.get(sym, sym)).alias("name"),
        )
        frames.append(j.select("instrument", "name", "month", "ours", "theirs"))
    if not frames:
        return pl.DataFrame(schema={"instrument": pl.Utf8, "name": pl.Utf8,
                                    "month": pl.Date, "ours": pl.Float64, "theirs": pl.Float64})
    return pl.concat(frames)


def _consistency_correlations() -> pl.DataFrame:
    # Grade the shipped Tier-2 aggregates read from disk. BOTH sides are compounded
    # to monthly before correlating (verified median ~0.9789 → PASS). With both
    # already monthly, the dt.truncate inside consistency_correlations is a harmless
    # no-op and the period join is 1:1.
    def _load(path):
        df = pl.read_csv(path, infer_schema_length=0)
        return df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    synced_monthly = compute_monthly_returns(_load(DATASETS_ROOT / "tier2" / "sync" / "sync_daily.csv"))
    asynced_monthly = compute_monthly_returns(_load(DATASETS_ROOT / "tier2" / "async" / "async_daily.csv"))
    return consistency_correlations(synced_monthly, asynced_monthly, freq="monthly")


consistency_check = Check(
    name="Async vs sync consistency",
    slug="consistency",
    run=_consistency_correlations,          # unchanged: grades Tier-2
    pairs=_consistency_pairs,               # comparison.pdf: Tier-1
    series_labels=("sync", "async"),
    requires_sync=True,
)
