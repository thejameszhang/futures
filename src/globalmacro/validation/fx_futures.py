# src/globalmacro/validation/fx_futures.py
"""Futures-vs-Compustat-spot validation exercise (per currency x mode): daily correlation
table (graded) plus a single house-style comparison plot (diagnostic figure).

`daily_fx_vs_spot` is the canonical gap-aware alignment of a currency future's daily
return against the Compustat spot return over the SAME observation gap -- originally
written (and G10-daily-corr->0.95 validated) in `scripts/compare_4pm_london_fx.py` as
`compute_gap_aware_spot_return`; that script now imports this implementation instead of
carrying its own copy.

`fx_futures_vs_spot_check` (registered in `validation/run.py`) is the promoted exercise:
`run()` walks both modes ("et": the shipped 9:30-ET sync panel; "london": the Task-6
4pm-London-aligned panel) x every `SYMBOL_TO_CURCDD_MAPPING` currency, scoring each pair
with `fx_futures_correlation`. Only G10-london rows are graded (`used=True`) -- see
`G10_MAJORS` for why. `figures` is `plot_fx_vs_spot_comparison`, which writes ONE
`comparison.pdf` -- a small-multiples grid, one panel per currency, plotting every pair
(graded or not) for visual inspection; matplotlib itself lives in `validation/plots.py`.

The panels read the PRE-build `sync/currency_daily_returns.csv`, which still carries
pre-listing bad prints that `build` nulls out before shipping (CZK `2004-07-13`, PLN
`2004-07-12`). `_fx_vs_spot_panels` clips each currency's plotted cumulative lines to
its SHIPPED date window -- `first_non_null`/`last_non_null` of that symbol's column in
the post-build `tier2/sync/sync_daily.csv` (see `_shipped_windows`) -- so those artifacts
never reach the plot. This is a plotting-window change only: correlations (graded or
not) are still `fx_futures_correlation`'s numbers over the un-clipped series.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl

from globalmacro.pipeline.fx import SYMBOL_TO_CURCDD_MAPPING, save_compustat_fx_rates
from globalmacro.utils.paths import DATASETS_ROOT, VALIDATION_OUTPUT
from globalmacro.utils.sync_fx import G10_FUTURES, build_sync_fx_panel
from globalmacro.validation.base import Check, Invariant
from globalmacro.validation.plots import plot_fx_vs_spot_grid
from globalmacro.validation.synthetic import pre_splice_panel

# mode label -> the sync subdirectory it reads from. "et" (mode a) is the shipped ET sync
# panel; "london" (mode b) is the Task-6 4pm-London-aligned sync panel and may not exist
# yet -- see `_load_currency_panel`.
_MODES: tuple[tuple[str, str], ...] = (("et", "sync"), ("london", "sync_london"))

# Grading is restricted to G10-london rows. Grounded in data/comp/{london_4pm,et}_
# fx_futures_correlation.csv (Task 5/6 output): london-4pm G10 pearson runs 0.87-0.98
# (median of the 9 ~=0.98, >=0.95 -> PASS). ET is 0.80-0.87 for the SAME G10 currencies --
# structurally capped below 0.95 by the ~1.5h 9:30-ET-vs-4pm-London timing gap between the
# futures settle and the Compustat snapshot, not a data defect -- so ET is a visual
# spike-free-tracking diagnostic only, never graded. EM (KRW 0.02, PLN 0.34, CZK -0.01,
# CNY 0.20, ...) is peg/noise-dominated in both modes -- diagnostic only. Do not widen this
# set without new grounding data; see emfx-task-7-brief.md.
G10_MAJORS = frozenset({"6A", "6C", "6J", "6B", "6E", "6N", "6S", "NOK", "SEK"})
_MIN_OBS = 30


def daily_fx_vs_spot(
    fut_returns: pl.DataFrame, comp_levels: pl.DataFrame, symbol: str, ccy: str
) -> pl.DataFrame:
    """Daily futures return (`symbol`) vs Compustat spot return (`ccy`), gap-aware: each
    Compustat return spans exactly the same observation gap as the futures return it is
    paired with (a futures panel is not evenly sampled -- holidays, listing gaps, and
    per-symbol start dates all leave holes).

    Seeds the FX level from the last Compustat observation strictly BEFORE the futures
    panel starts, so the first in-panel day's return is a real return against a real prior
    level, instead of coming back null purely for lack of anything before it to divide by.
    Without this seed the naive gap-aware computation nulls out exactly the first
    observation of every symbol -- worth nothing on a multi-decade panel, but worth
    preserving since it was part of the validated (G10 daily corr > 0.95) logic.

    Returns `[date, r_fut, r_spot]`, one row per in-panel futures date (an empty
    typed frame if `symbol`/`ccy` has no data, so callers can `.drop_nulls()` without a
    schema check).
    """
    empty = pl.DataFrame(schema={"date": pl.Date, "r_fut": pl.Float64, "r_spot": pl.Float64})
    if ccy not in comp_levels.columns:
        return empty

    spot = (
        comp_levels.select("date", pl.col(ccy).cast(pl.Float64, strict=False))
        .rename({ccy: "__fx_level"})
    )
    fut = fut_returns.select("date", pl.col(symbol).cast(pl.Float64, strict=False))
    panel_start = fut.filter(pl.col(symbol).is_not_null()).select(pl.col("date").min()).item()
    if panel_start is None:
        return empty

    seed = spot.filter((pl.col("date") < panel_start) & pl.col("__fx_level").is_not_null()).tail(1)
    fut_filtered = fut.filter(pl.col("date") >= panel_start)
    if seed.height > 0:
        seed_date = seed["date"][0]
        seed_fut = pl.DataFrame(
            {"date": [seed_date], symbol: [None]}, schema={"date": pl.Date, symbol: pl.Float64}
        )
        fut_concat = pl.concat([seed_fut, fut_filtered])
    else:
        fut_concat = fut_filtered

    df = fut_concat.join(spot, on="date", how="left").sort("date")
    df = df.with_columns(pl.col("__fx_level").forward_fill())

    observed = pl.when(pl.col(symbol).is_not_null()).then(pl.col("__fx_level")).otherwise(None)
    df = df.with_columns(observed.alias("__observed_level"))

    prev_observation = pl.col("__observed_level").forward_fill().shift(1)
    denom = pl.coalesce([prev_observation, pl.col("__fx_level").shift(1)])
    df = df.with_columns((pl.col("__fx_level") / denom - 1.0).alias("r_spot"))

    df = df.filter(pl.col("date") >= panel_start)
    df = df.rename({symbol: "r_fut"})
    return df.select("date", "r_fut", "r_spot")


def _load_currency_panel(symbol: str, sync_dir: str) -> pl.DataFrame | None:
    """Read whichever tier's `<sync_dir>/currency_daily_returns.csv` carries `symbol`.

    Returns None (never raises) if neither tier has the file, or neither has the column --
    callers must skip missing panels rather than crash. `sync_london` in particular is a
    Task-6 DAG output and may not have run yet.
    """
    for tier in (1, 2):
        path = DATASETS_ROOT / f"tier{tier}" / sync_dir / "currency_daily_returns.csv"
        if not path.exists():
            continue
        df = pl.read_csv(path, schema_overrides={"date": pl.Date})
        if symbol not in df.columns:
            continue
        return df.select("date", pl.col(symbol).cast(pl.Float64, strict=False))
    return None


def _shipped_windows() -> dict[str, tuple[date, date]]:
    """One-time read of the post-build `datasets/tier2/sync/sync_daily.csv` -- the
    tier1|tier2 union superset -- yielding `{symbol: (first_non_null_date,
    last_non_null_date)}` for every currency-future column in that file.

    `build`'s `keep_after_date`/`set_null_on_date`/splice nulling has already been
    applied here, so a pre-listing bad print (CZK `2004-07-13`, PLN `2004-07-12`) is
    `null` in this file and therefore excluded from the window -- this is the SHIPPED
    window `_fx_vs_spot_panels` clips its plotted cumulative lines to.

    `infer_schema_length=None` (full-file dtype scan, mirroring
    `scripts/scan_currency_spikes.py`'s read of this same file): the default
    100-row sample misinfers several sparse currency columns as strings.

    A symbol absent from the file, or whose column is entirely null, is simply absent
    from the returned map -- `_fx_vs_spot_panels` falls back to no clip for it rather
    than dropping the panel.
    """
    path = DATASETS_ROOT / "tier2" / "sync" / "sync_daily.csv"
    if not path.exists():
        return {}
    df = pl.read_csv(path, infer_schema_length=None, schema_overrides={"date": pl.Date})
    windows: dict[str, tuple[date, date]] = {}
    for col in df.columns:
        if col == "date":
            continue
        non_null = df.select("date", pl.col(col).cast(pl.Float64, strict=False)).drop_nulls()
        if non_null.height == 0:
            continue
        first_nn: date = non_null.select(pl.col("date").min()).item()
        last_nn: date = non_null.select(pl.col("date").max()).item()
        windows[col] = (first_nn, last_nn)
    return windows


def _blend_corr_from_series(series: pl.DataFrame) -> float | None:
    """Pearson corr(r_fut, r_spot) over the both-finite support, or None if < _MIN_OBS."""
    valid = series.drop_nulls(["r_fut", "r_spot"])
    if valid.height < _MIN_OBS:
        return None
    return valid.select(pl.corr("r_fut", "r_spot")).item()


def _blend_series(comp_levels: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """For each G10 symbol: [date, r_fut, r_spot] where r_fut is the futures return the
    blend used (pre_splice_synced[symbol], DM-spliced for 6E) and r_spot is the blend
    level differenced over the same futures gaps. On futures-covered days r_spot == r_fut;
    they diverge only the day after a futures holiday -> corr ~0.99.

    Defensive (review C1): returns {} without touching the expensive pre_splice_panel
    build if `comp_levels` carries no G10 currency (toy panels, non-G10-only runs), and
    swallows build_sync_fx_panel's zero-override guard -- so this validation helper never
    aborts the run; a genuinely all-USD real run surfaces as an empty {} -> the invariant
    reports passed=False (a clean failure, not a traceback).
    """
    if not (set(G10_FUTURES.values()) & set(comp_levels.columns)):
        return {}
    pre = pre_splice_panel("sync")
    try:
        blended = build_sync_fx_panel(comp_levels, pre, SYMBOL_TO_CURCDD_MAPPING)
    except ValueError:
        return {}
    out: dict[str, pl.DataFrame] = {}
    for symbol, ccy in G10_FUTURES.items():
        if symbol not in pre.columns or ccy not in blended.columns:
            continue
        fut = pre.select("date", pl.col(symbol).cast(pl.Float64, strict=False))
        out[symbol] = daily_fx_vs_spot(fut, blended, symbol, ccy)
    return out


def _blend_invariant() -> list[Invariant]:
    """min over G10 of corr(blend, its futures) >= 0.99. Min, not median: a median of 9
    passes with up to 4 broken currencies; min fails on any single wrong-column/spot-leak."""
    comp_levels = save_compustat_fx_rates()
    corrs = {sym: _blend_corr_from_series(s) for sym, s in _blend_series(comp_levels).items()}
    scored = {sym: c for sym, c in corrs.items() if c is not None}
    if not scored:
        return [Invariant(check="Futures vs Compustat spot (daily)",
                          name="blend tracks its futures (min over G10)",
                          value="no G10 blend series", passed=False)]
    worst_sym = min(scored, key=lambda k: scored[k])
    worst = scored[worst_sym]
    return [Invariant(
        check="Futures vs Compustat spot (daily)",
        name="blend tracks its futures (min over G10 corr >= 0.99)",
        value=f"min {worst:.4f} @ {G10_FUTURES[worst_sym]}",
        passed=worst >= 0.99,
    )]


def _fx_vs_spot_panels(comp_levels: pl.DataFrame, blend: dict[str, pl.DataFrame] | None = None) -> list[dict]:
    """Build one per-currency panel record for `plot_fx_vs_spot_grid`, for every currency
    in `SYMBOL_TO_CURCDD_MAPPING` present in `comp_levels`.

    Per currency: outer-join the ET futures return, the London futures return, and a
    single reference spot return (London-mode `r_spot` -- the 4pm apples-to-apples spot
    -- falling back to ET-mode `r_spot` on any date London has no observation) on `date`,
    then clip to the window where at least one futures series has a non-null return,
    then clip AGAIN to the currency's SHIPPED window (`_shipped_windows()`, read once
    up front -- not per currency) so pre-listing bad prints that `build` nulls out of
    the shipped `sync_daily.csv` (CZK `2004-07-13`, PLN `2004-07-12`) never reach the
    plot, even though they are still present in this function's PRE-build source panels.
    A currency absent from `_shipped_windows()` falls back to no clip -- the panel is
    still built, just over its full pre-build window. Cumulative log returns
    (`cumsum(log1p(r))`, missing daily return -> 0 so a gap shows as a flat segment) are
    computed AFTER both clips, here in polars, so the cumulative lines start at 0 at the
    first SHIPPED date -- `plot_fx_vs_spot_grid` only draws already-cumulative arrays,
    no matplotlib-side computation.

    Correlations annotated on the plot are `fx_futures_correlation`'s pearson for that
    exact (symbol, mode) pair -- the SAME scorer `_correlations()` grades on, called with
    the same arguments over the UN-clipped panel -- so the panel's `r_et`/`r_lon` MUST
    equal `correlations.csv`'s `correlation` column for that (symbol, mode) row exactly;
    this function never recomputes a correlation itself, and the shipped-window clip
    (which only affects the plotted cumulative lines) never touches it.

    A mode absent for a currency (file/column missing, or no in-panel data at all)
    contributes neither a line nor an annotation for that currency (`cum_et`/`cum_lon`/
    `r_et`/`r_lon` = None) rather than a misleading flat-zero line. A currency is skipped
    entirely only if BOTH modes are absent, or if the shipped-window clip leaves nothing.
    """
    panels: list[dict] = []
    windows = _shipped_windows()
    blend = blend or {}   # symbol -> [date, r_fut, r_spot]; empty in the direct-call unit tests
    for symbol, ccy in SYMBOL_TO_CURCDD_MAPPING.items():
        if ccy not in comp_levels.columns:
            continue

        mode_frames: dict[str, pl.DataFrame] = {}
        mode_corr: dict[str, float | None] = {}
        for mode_label, sync_dir in _MODES:
            fut = _load_currency_panel(symbol, sync_dir)
            if fut is None:
                continue
            mode_corr[mode_label] = fx_futures_correlation(
                fut, comp_levels, symbol, ccy, mode_label
            )["correlation"]
            result = daily_fx_vs_spot(fut, comp_levels, symbol, ccy)
            if result.height > 0:
                mode_frames[mode_label] = result.select(
                    "date",
                    pl.col("r_fut").alias(f"r_{mode_label}"),
                    pl.col("r_spot").alias(f"r_spot_{mode_label}"),
                )

        if not mode_frames:
            continue

        if "et" in mode_frames and "london" in mode_frames:
            merged = mode_frames["et"].join(
                mode_frames["london"], on="date", how="full", coalesce=True
            )
        elif "et" in mode_frames:
            merged = mode_frames["et"].with_columns(
                r_london=pl.lit(None, dtype=pl.Float64),
                r_spot_london=pl.lit(None, dtype=pl.Float64),
            )
        else:
            merged = mode_frames["london"].with_columns(
                r_et=pl.lit(None, dtype=pl.Float64),
                r_spot_et=pl.lit(None, dtype=pl.Float64),
            )

        merged = merged.sort("date").with_columns(
            pl.coalesce(["r_spot_london", "r_spot_et"]).alias("r_spot")
        )
        merged = merged.filter(pl.col("r_et").is_not_null() | pl.col("r_london").is_not_null())
        if merged.height == 0:
            continue

        first_shipped, last_shipped = windows.get(symbol, (None, None))
        if first_shipped is not None:
            merged = merged.filter(pl.col("date") >= first_shipped)
        if last_shipped is not None:
            merged = merged.filter(pl.col("date") <= last_shipped)
        if merged.height == 0:
            continue

        merged = merged.with_columns(
            pl.col("r_et").fill_null(0.0).log1p().cum_sum().alias("cum_et"),
            pl.col("r_london").fill_null(0.0).log1p().cum_sum().alias("cum_lon"),
            pl.col("r_spot").fill_null(0.0).log1p().cum_sum().alias("cum_spot"),
        )

        blend_df = blend.get(symbol)
        r_blend = _blend_corr_from_series(blend_df) if blend_df is not None else None
        cum_blend = None
        if blend_df is not None and blend_df.height:
            cb = (
                merged.select("date")
                .join(blend_df.select("date", pl.col("r_spot")), on="date", how="left")
                .with_columns(pl.col("r_spot").fill_null(0.0).log1p().cum_sum().alias("cum_blend"))
            )
            cum_blend = cb.get_column("cum_blend").to_numpy()

        panels.append({
            "ccy": ccy,
            "symbol": symbol,
            "dates": merged.get_column("date").to_numpy(),
            "cum_et": merged.get_column("cum_et").to_numpy() if "et" in mode_frames else None,
            "cum_lon": merged.get_column("cum_lon").to_numpy() if "london" in mode_frames else None,
            "cum_spot": merged.get_column("cum_spot").to_numpy(),
            "cum_blend": cum_blend,
            "r_et": mode_corr.get("et"),
            "r_lon": mode_corr.get("london"),
            "r_blend": r_blend,
        })
    return panels


def plot_fx_vs_spot_comparison(out_dir: str | Path) -> None:
    """Write ONE `out_dir/comparison.pdf`: a small-multiples grid, one panel per
    currency, plotting ET futures / London futures / Compustat spot cumulative daily log
    returns with both correlations annotated (see `_fx_vs_spot_panels`). Replaces the
    old per-run 36-PDF `<sym>_et.pdf`/`<sym>_london.pdf` output.
    """
    out_dir = Path(out_dir)
    comp_levels = save_compustat_fx_rates()
    panels = _fx_vs_spot_panels(comp_levels, blend=_blend_series(comp_levels))
    plot_fx_vs_spot_grid(
        panels,
        "Futures vs Compustat spot: cumulative daily log returns, per currency",
        out_dir / "comparison.pdf",
    )


_CORR_SCHEMA = pl.Schema(
    {
        "instrument": pl.Utf8,
        "correlation": pl.Float64,
        "spearman": pl.Float64,
        "n_obs": pl.Int64,
        "used": pl.Boolean,
    }
)


def fx_futures_correlation(
    fut: pl.DataFrame, comp_levels: pl.DataFrame, symbol: str, ccy: str, mode: str
) -> dict:
    """Pure per-(symbol, mode) scorer: given in-memory futures-return and Compustat-level
    frames, align them via `daily_fx_vs_spot` and score the daily Pearson/Spearman
    correlation plus the `used` grading flag. No filesystem I/O -- this is what tests
    exercise directly to assert `used` gating without needing on-disk panels, and what
    `_correlations()` below calls once per (symbol, mode) x currency pair.

    Guard: fewer than `_MIN_OBS` valid daily pairs is not enough to trust a correlation --
    `correlation`/`spearman` come back None and `used` is forced False regardless of the
    grading rule.
    """
    result = daily_fx_vs_spot(fut, comp_levels, symbol, ccy)
    valid = result.drop_nulls(["r_fut", "r_spot"])
    n_obs = valid.height
    if n_obs < _MIN_OBS:
        pearson: float | None = None
        spearman: float | None = None
    else:
        pearson = valid.select(pl.corr("r_fut", "r_spot")).item()
        spearman = valid.select(pl.corr("r_fut", "r_spot", method="spearman")).item()
    used = symbol in G10_MAJORS and mode == "london" and pearson is not None and n_obs >= _MIN_OBS
    return {
        "instrument": f"{symbol} ({mode})",
        "correlation": pearson,
        "spearman": spearman,
        "n_obs": n_obs,
        "used": used,
    }


def _correlations() -> pl.DataFrame:
    """`run()` for `fx_futures_vs_spot_check`: every (symbol, mode) pair in
    `SYMBOL_TO_CURCDD_MAPPING` x `_MODES`, scored by `fx_futures_correlation`. Skips pairs
    whose currency isn't in the Compustat panel or whose futures panel file/column is
    absent (see `_load_currency_panel`) rather than emitting a row for them -- an absent
    panel is not a diagnostic-worthy zero, it is nothing to report.

    `save_compustat_fx_rates()` is called ONCE, not per row -- it re-derives and rewrites
    `compustat_fx_rates.csv` and is not free.
    """
    comp_levels = save_compustat_fx_rates()
    rows = []
    for symbol, ccy in SYMBOL_TO_CURCDD_MAPPING.items():
        if ccy not in comp_levels.columns:
            continue
        for mode_label, sync_dir in _MODES:
            fut = _load_currency_panel(symbol, sync_dir)
            if fut is None:
                continue
            rows.append(fx_futures_correlation(fut, comp_levels, symbol, ccy, mode_label))
    if not rows:
        # Explicit schema: an empty pl.DataFrame([]) has NO columns, so run.py's
        # filter(pl.col("used")) would raise ColumnNotFoundError instead of reporting 0/0.
        return pl.DataFrame(schema=_CORR_SCHEMA)
    return pl.DataFrame(rows, schema=_CORR_SCHEMA).sort("correlation", descending=True, nulls_last=True)


fx_futures_vs_spot_check = Check(
    name="Futures vs Compustat spot (daily)",
    slug="fx_futures_vs_spot",
    run=_correlations,
    invariants=_blend_invariant,
    figures=plot_fx_vs_spot_comparison,
    requires_sync=True,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m globalmacro.validation.fx_futures")
    parser.add_argument("--out-dir", default=str(VALIDATION_OUTPUT / "fx_futures"))
    args = parser.parse_args(argv)
    plot_fx_vs_spot_comparison(args.out_dir)


if __name__ == "__main__":
    main()
