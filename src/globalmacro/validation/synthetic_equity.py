# src/globalmacro/validation/synthetic_equity.py
"""Exercise (b): spot index returns spliced in ahead of each index future.

The invariant is what proves the Part-A lag and keeps it from regressing. The sync panel
samples at 09:31 ET, so its day-t window holds an Americas cash session from day t-1: the
lagged synthetic must beat the same-day one for exactly {SXF, YM, NQ, RTY}, and lose for
every other symbol with a signal to speak of (Europe's session is 77-94% inside the window,
Asia's 100%). AMERICAS_CASH_INDICES' fifth member, IPC, is indeterminate -- neither
candidate clears MIN_ALIGNMENT_SIGNAL -- so it is reported, not graded.

The sync synthetic is NOT a file on disk -- spot_equity_returns.csv is unlagged and the lag
is applied inside load_synthetic_returns. Reading the CSV directly would silently grade the
PRE-FIX alignment, so we call the function.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import polars as pl

from globalmacro.build import (
    AMERICAS_CASH_INDICES,
    first_valid_date,
    lag_one_session,
    load_rf,
    load_symbols,
    load_synthetic_returns,
)
from globalmacro.utils.capabilities import sync_panels_ready
from globalmacro.validation.base import Check, Invariant
from globalmacro.validation.mode import current_mode
from globalmacro.validation.synthetic import (
    daily_corr,
    pre_splice_panel,
    shipped_panel,
    synthetic_correlations,
    synthetic_pairs,
)

# The graded median below is computed on the ASYNC panel; the alignment invariant is what
# speaks for sync. Name the graded panel, so "PASS" cannot be read as a verdict on sync.
NAME = "Synthetic equity returns (async)"
SLUG = "synthetic_equity"

# load_synthetic_returns joins the FX synthetics onto the equity ones; these four columns
# are the FX side and are exercise (a)'s business, not ours.
_FX_SYNTHETIC_COLS = ("NOK", "SEK", "6N", "6A")

# Below this, the BEST of the two daily correlations is indistinguishable from noise, so
# "which alignment does the data prefer" has no answer: both candidates are noise, and
# whichever happens to be larger is a coin flip, not evidence. Such a symbol is reported
# indeterminate and excluded from the verdict rather than graded on a coin flip. IPC is
# one (0.003 same-day vs 0.122 lag-1): its lag CANNOT be validated from data, and saying
# so is the honest report. Its membership in AMERICAS_CASH_INDICES rests on the cash
# index's exchange, a human judgement -- see build.AMERICAS_CASH_INDICES.
MIN_ALIGNMENT_SIGNAL = 0.30

# Shared between _invariants()/_figures() and synthetic_equity_check's dropped_*
# fields below, so the names/filename async-only prints under "## Skipped" cannot
# drift from what full mode would actually emit for the same invariant/figure (see
# synthetic_fx._invariant_name's docstring for the same rationale; base.py:36-41 --
# a shared constant, not a call into the check, so this does not run the sync-only
# code async-only mode exists to avoid).
_ALIGNMENT_INVARIANT = "shipped alignment is the one the data prefers, per symbol"
_COVERAGE_INVARIANT = "every Americas symbol is actually under test"
_ALIGNMENT_PDF = "equity_alignment.pdf"


@lru_cache(maxsize=1)
def _synth_panels() -> tuple[pl.DataFrame, pl.DataFrame]:
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    equities = [f for f in (t1f + t2f) if f.dsindexcode is not None]
    return load_synthetic_returns(load_rf(), equities)


def _equity_only(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        ["date"] + [c for c in df.columns if c != "date" and c not in _FX_SYNTHETIC_COLS]
    )


def _synthetic_base() -> pl.DataFrame:
    """The UNLAGGED synthetic equity panel, exactly as build derives it.

    This is the async panel, NOT spot_equity_returns.csv. load_synthetic_returns does real
    work on the raw CSV -- it coalesces each multi-dsindexcode index with its historical
    predecessor and subtracts rf (build.py:340-342) -- so the raw file differs from what
    actually ships for AP, FIE, FXS30 and Z. Comparing against the raw CSV makes those four
    match NEITHER alignment candidate, and the invariant then reports them as misaligned
    when they are fine.
    """
    async_synth, _ = _synth_panels()
    return _equity_only(async_synth)


def _sync_synthetic() -> pl.DataFrame:
    """The synthetic as build actually splices it into sync -- i.e. WITH the Americas lag."""
    _, sync_synth = _synth_panels()
    return _equity_only(sync_synth)


def _correlations() -> pl.DataFrame:
    """Graded on async (the longer history and the larger universe)."""
    return synthetic_correlations(
        synth=_synthetic_base(),
        pre=pre_splice_panel("async"),
        ship=shipped_panel("async", tier=1),
    )


def _name_of() -> dict[str, str]:
    """symbol -> index name, for the equities in EITHER tier (mirrors _synth_panels'
    own universe: tier1 + tier2 futures carrying a dsindexcode)."""
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    return {f.symbol: f.name for f in (t1f + t2f) if f.dsindexcode is not None}


def _pairs() -> pl.DataFrame:
    """comparison.pdf. Same synth/pre/ship as _correlations() -- IDENTICAL window -- so
    the plot shows exactly what was graded, never the sync (lagged) panel or the
    pre-cutoff backfill."""
    return synthetic_pairs(
        synth=_synthetic_base(),
        pre=pre_splice_panel("async"),
        ship=shipped_panel("async", tier=1),
        name_of=_name_of(),
    )


def _matches(a: pl.DataFrame, b: pl.DataFrame, symbol: str) -> bool:
    """Do two frames carry the same series for `symbol` on their shared dates?"""
    j = (
        a.select("date", pl.col(symbol).alias("p"))
        .join(b.select("date", pl.col(symbol).alias("q")), on="date", how="inner")
        .filter(pl.col("p").is_not_null() | pl.col("q").is_not_null())
    )
    if j.height == 0:
        return False
    differs = j.filter(
        pl.col("p").is_null()
        | pl.col("q").is_null()
        | ((pl.col("p") - pl.col("q")).abs() > 1e-12)
    ).height
    return differs == 0


def alignment() -> pl.DataFrame:
    """Per symbol: which alignment SHIPPED, and which one the data actually prefers.

    Deliberately NOT keyed on AMERICAS_CASH_INDICES for the expectation. An earlier draft
    derived both the counterfactual and the expectation from that constant, so zeroing it
    moved both in lockstep and the invariant trivially passed -- an invariant that cannot
    fail is worthless. Here:

      shipped  = what load_synthetic_returns actually put in the sync panel, discovered by
                 comparing it against both candidates.
      expected = whichever candidate correlates better with the real sync future. Pure data.

    The invariant then asserts shipped == expected. Remove the lag and SXF/YM/NQ/RTY ship
    the same-day series while the data prefers lag-1, and it fails -- as it must.

    A symbol whose BEST daily correlation is below MIN_ALIGNMENT_SIGNAL gets no verdict:
    `expected` is "indeterminate" and the invariant excludes it. Forcing a preference out
    of two noise correlations would be a coin flip dressed up as a check.

    TIER 2, not tier 1. build.main() splices the synthetic into the tier-2 panel and tier 1
    is a column SUBSET of it, so grading tier 1 silently drops every tier-2-only index --
    IPC among them, which ships 867 lagged rows and would be validated by nothing.
    """
    base = _synthetic_base()                        # unlagged, exactly as build derives it
    shipped_synth = _sync_synthetic()
    equity_symbols = [c for c in base.columns if c != "date"]

    same_day = base
    lagged = lag_one_session(base, equity_symbols)   # EVERY symbol, not just the Americas

    pre, ship = pre_splice_panel("sync"), shipped_panel("sync", tier=2)
    rows = []
    for symbol in sorted(equity_symbols):
        if symbol not in ship.columns or symbol not in pre.columns:
            continue
        if symbol not in shipped_synth.columns:
            continue
        cutoff = first_valid_date(pre, symbol)
        if cutoff is None:
            continue
        joined = shipped_synth.select("date", pl.col(symbol).alias("shipped")).join(
            ship.select("date", pl.col(symbol).alias("y")), on="date", how="inner"
        )
        n_backfilled = joined.filter(
            (pl.col("date") < cutoff)
            & pl.col("shipped").is_not_null()
            & pl.col("y").is_not_null()
        ).height
        if n_backfilled == 0:
            continue

        ref = ship.select("date", pl.col(symbol).alias("y")).filter(pl.col("date") >= cutoff)
        sd = ref.join(same_day.select("date", pl.col(symbol).alias("x")), on="date", how="inner")
        lg = ref.join(lagged.select("date", pl.col(symbol).alias("x")), on="date", how="inner")
        c_same, c_lag = daily_corr(sd, "x"), daily_corr(lg, "x")
        if c_same is None or c_lag is None:
            continue

        # What did we actually ship? Read it off the frame, don't assume it from a constant.
        if _matches(shipped_synth, lagged, symbol):
            shipped_alignment = "lag_1"
        elif _matches(shipped_synth, same_day, symbol):
            shipped_alignment = "same_day"
        else:
            shipped_alignment = "unknown"

        if max(c_same, c_lag) < MIN_ALIGNMENT_SIGNAL:
            expected = "indeterminate"          # both candidates are noise; no verdict
        else:
            expected = "lag_1" if c_lag > c_same else "same_day"   # data decides

        rows.append(
            {
                "instrument": symbol,
                "same_day": c_same,
                "lag_1": c_lag,
                "shipped": shipped_alignment,
                "expected": expected,
                "americas": symbol in AMERICAS_CASH_INDICES,            # reported only
                "n_backfilled": n_backfilled,
            }
        )
    if not rows:
        # Explicit schema: pl.DataFrame([]) has NO columns, so _invariants()'s
        # filter(pl.col("shipped")) would raise ColumnNotFoundError instead of reporting 0/0.
        return pl.DataFrame(
            schema={
                "instrument": pl.Utf8, "same_day": pl.Float64, "lag_1": pl.Float64,
                "shipped": pl.Utf8, "expected": pl.Utf8, "americas": pl.Boolean,
                "n_backfilled": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def _grade_sync() -> bool:
    """Mirrors synthetic_fx._grade_sync: the sync half needs BOTH the sync panels present
    (sync_panels_ready) AND the resolved mode actually "full" (validation.mode.
    current_mode) -- an explicit --async-only must suppress it even when the panels are
    on disk and fresh. Data-free, so unit-testable by monkeypatching sync_panels_ready
    and setting the mode via validation.mode.validation_mode."""
    return current_mode() == "full" and sync_panels_ready().ready


def _invariants() -> list[Invariant]:
    if not _grade_sync():
        return []          # both invariants derive from alignment(), which is sync-only
    a = alignment()

    # Grade only the symbols the data can actually speak about. An indeterminate symbol is
    # NOT a pass and NOT a failure -- it is a symbol whose alignment this exercise cannot
    # test -- so it leaves the numerator AND the denominator, and is named in the value.
    determinate = a.filter(pl.col("expected") != "indeterminate")
    indeterminate = sorted(a.filter(pl.col("expected") == "indeterminate")["instrument"].to_list())
    correct = determinate.filter(pl.col("shipped") == pl.col("expected")).height
    total = determinate.height
    wrong = sorted(
        determinate.filter(pl.col("shipped") != pl.col("expected"))["instrument"].to_list()
    )
    detail = f"{correct}/{total}"
    if wrong:
        detail += f" (misaligned: {wrong})"
    if indeterminate:
        detail += f" ({len(indeterminate)} indeterminate: {indeterminate})"

    # Pin the denominator, and take it from the CONSTANT. alignment() drops any symbol with
    # no backfill, and RTY ships exactly ONE backfilled observation (its sync cutoff is one
    # day after the panel floor). If that cutoff ever slips by a day RTY silently leaves the
    # frame and the verdict above reports a clean n-1/n-1 PASS. Deriving the denominator
    # from the frame under test, or from the shipped panel's columns, means the check cannot
    # notice its own blind spot -- that is exactly how a tier-1 lookup here reported "4/4"
    # for a five-element constant while IPC was absent. Every member must be IN the frame;
    # being indeterminate is an acceptable state, being absent is not.
    in_frame = set(a.get_column("instrument").to_list()) if a.height else set()
    expected_americas = set(AMERICAS_CASH_INDICES)
    missing = sorted(expected_americas - in_frame)

    return [
        Invariant(
            check=NAME,
            name=_ALIGNMENT_INVARIANT,
            value=detail,
            passed=total > 0 and correct == total,
        ),
        Invariant(
            check=NAME,
            name=_COVERAGE_INVARIANT,
            value=f"{len(expected_americas) - len(missing)}/{len(expected_americas)}"
                  + (f" (MISSING: {missing})" if missing else ""),
            passed=not missing,
        ),
    ]


def _figures(out_dir: Path) -> None:
    if not _grade_sync():
        return          # equity_alignment.pdf is a sync-only comparison; nothing to draw
    from globalmacro.validation.plots import plot_paired_bars

    a = alignment().with_columns(
        pl.when(pl.col("americas")).then(pl.lit("Americas (lag 1)"))
        .otherwise(pl.lit("Europe / Asia (same day)")).alias("region")
    )
    plot_paired_bars(
        a,
        group_col="region",
        label_col="instrument",
        left_col="same_day",
        right_col="lag_1",
        series_labels=("same-day spot return", "lag-1 spot return"),
        title="Daily correlation of the synthetic equity return with the sync future\n"
              "The 09:31 ET window holds the PREVIOUS session for Americas indices only",
        ylabel="daily correlation",
        path=out_dir / _ALIGNMENT_PDF,
    )


synthetic_equity_check = Check(
    name=NAME,
    slug=SLUG,
    run=_correlations,
    pairs=_pairs,
    series_labels=("spot synthetic", "real future"),
    invariants=_invariants,
    figures=_figures,
    dropped_invariants=(_ALIGNMENT_INVARIANT, _COVERAGE_INVARIANT),
    dropped_figures=(_ALIGNMENT_PDF,),
)
