# src/globalmacro/validation/synthetic_fx.py
"""Exercise (a): CIP-implied synthetic FX futures vs the real FX futures.

Graded on the synthetics that actually ship. The invariant is the load-bearing part:
each dataset's synthetic is built from its OWN FX source (Datastream EOD for async,
Compustat/WM-Reuters 4pm London for sync), and each must win on its own futures panel.
Measured 16/16 with wide margins. A swapped FX source inverts every one of them.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from globalmacro.build import first_valid_date
from globalmacro.pipeline.fx import SYMBOL_TO_CURCDD_MAPPING
from globalmacro.utils.paths import FX_PATH
from globalmacro.validation.base import Check, Invariant
from globalmacro.validation.synthetic import (
    daily_corr,
    load_panel,
    pre_splice_panel,
    shipped_panel,
    synthetic_correlations,
    synthetic_pairs,
)

# The graded median below is computed on the ASYNC panel. Say so in the name: a reader of
# VALIDATION_SUMMARY.md must not read "PASS" as a verdict on a panel that was not graded.
NAME = "Synthetic FX returns (async)"
SLUG = "synthetic_fx"


def _synthetics() -> tuple[pl.DataFrame, pl.DataFrame]:
    """(Datastream-derived, Compustat-derived).

    Read from disk by filename, so source_diagonal() proves only that the FILES are what
    they claim to be. That the pipeline then splices the right file into the right panel is
    a separate claim, pinned by exact equality in tests/test_usd_identity.py -- swapping the
    two inside load_synthetic_returns leaves this diagonal byte-identical.
    """
    return (
        load_panel(FX_PATH / "synthetic_fx_returns_async.csv"),
        load_panel(FX_PATH / "synthetic_fx_returns_sync.csv"),
    )


def _correlations() -> pl.DataFrame:
    """Graded on async (the longer history); the sync panel is covered by the invariant."""
    datastream, compustat = _synthetics()
    return synthetic_correlations(
        synth=datastream,
        pre=pre_splice_panel("async"),
        ship=shipped_panel("async", tier=1),
        alt=compustat,          # corr_daily_alt = the WRONG source, for contrast
    )


def _pairs() -> pl.DataFrame:
    """comparison.pdf. Same synth/pre/ship as _correlations() -- IDENTICAL window -- so
    the plot shows exactly what was graded, never the sync panel or the pre-cutoff
    backfill."""
    datastream, _compustat = _synthetics()
    return synthetic_pairs(
        synth=datastream,
        pre=pre_splice_panel("async"),
        ship=shipped_panel("async", tier=1),
        name_of=SYMBOL_TO_CURCDD_MAPPING,
    )


def source_diagonal() -> pl.DataFrame:
    """Per currency: each synthetic's daily correlation against each futures panel."""
    datastream, compustat = _synthetics()
    rows = []
    for dataset in ("sync", "async"):
        pre, ship = pre_splice_panel(dataset), shipped_panel(dataset, tier=1)
        for symbol in sorted(c for c in datastream.columns if c != "date"):
            if symbol not in ship.columns or symbol not in pre.columns:
                continue
            cutoff = first_valid_date(pre, symbol)
            if cutoff is None:
                continue
            base = ship.select("date", pl.col(symbol).alias("y")).filter(
                pl.col("date") >= cutoff
            )
            ds = base.join(
                datastream.select("date", pl.col(symbol).alias("x")), on="date", how="inner"
            )
            cp = base.join(
                compustat.select("date", pl.col(symbol).alias("x")), on="date", how="inner"
            )
            c_ds, c_cp = daily_corr(ds, "x"), daily_corr(cp, "x")
            if c_ds is None or c_cp is None:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "instrument": symbol,
                    "datastream": c_ds,
                    "compustat": c_cp,
                    "expected": "compustat" if dataset == "sync" else "datastream",
                    "winner": "compustat" if c_cp > c_ds else "datastream",
                }
            )
    return pl.DataFrame(rows)


def _invariants() -> list[Invariant]:
    d = source_diagonal()
    out = []
    for dataset, source in (("sync", "Compustat"), ("async", "Datastream")):
        sub = d.filter(pl.col("dataset") == dataset)
        wins = sub.filter(pl.col("winner") == pl.col("expected")).height
        total = sub.height
        out.append(
            Invariant(
                check=NAME,
                name=f"{source} synthetic beats the other on the {dataset} futures",
                value=f"{wins}/{total}",
                passed=total > 0 and wins == total,
            )
        )
    return out


def _figures(out_dir: Path) -> None:
    from globalmacro.validation.plots import plot_paired_bars

    d = source_diagonal()
    plot_paired_bars(
        d,
        group_col="dataset",
        label_col="instrument",
        left_col="datastream",
        right_col="compustat",
        series_labels=("Datastream synthetic", "Compustat synthetic"),
        title="Daily correlation of each synthetic FX future with the real future\n"
              "Compustat wins on sync (09:31 ET); Datastream wins on async (settlement)",
        ylabel="daily correlation",
        path=out_dir / "fx_source_diagonal.pdf",
    )


synthetic_fx_check = Check(
    name=NAME,
    slug=SLUG,
    run=_correlations,
    pairs=_pairs,
    series_labels=("CIP synthetic", "real future"),
    invariants=_invariants,
    figures=_figures,
)
