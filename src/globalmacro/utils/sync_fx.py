# src/globalmacro/utils/sync_fx.py
"""Blend the TickHistory FX-futures return into the Compustat sync FX level panel.

For each G10 currency, rebuild its USD-per-currency level: Compustat spot verbatim below
the currency's real-future cutoff, then single weekday-step futures returns chained onto
the last pre-cutoff Compustat level. The result replaces fx_sync in save_usd_datasets;
usd_panel then differences the level over each asset's own calendar.

I/O-free: it receives the Compustat panel and the pre-splice sync panel. The pre-splice
panel (build_synced_dataset output, BEFORE splice_synthetic_returns) is real-splice-aware
(6E carries the DM future) and synthetic-blind (below the cutoff the future column is null,
so the blend falls to Compustat spot, never the CIP synthetic).
"""
from __future__ import annotations

import logging

import polars as pl

from globalmacro.utils.panels import first_finite_date

logger = logging.getLogger(__name__)

# The nine non-USD currencies with a liquid G10 currency future. Canonical here; a test
# pins this equal to validation.fx_futures.G10_MAJORS (never import validation from utils).
G10_FUTURES: dict[str, str] = {
    "6A": "AUD", "6C": "CAD", "6J": "JPY", "6B": "GBP", "6E": "EUR",
    "6N": "NZD", "6S": "CHF", "NOK": "NOK", "SEK": "SEK",
}


def _blend_currency_level(w: pl.DataFrame, cutoff) -> pl.DataFrame:
    """`w` is weekday-sorted with columns [date, fx_level, fut_ret]. Return [date, blend]:
    fx_level verbatim below `cutoff`; from `cutoff` a level chained from single weekday
    steps (the future where observed at both endpoints, else the Compustat step)."""
    observed = pl.col("fut_ret").is_finite()
    comp_step = pl.col("fx_level") / pl.col("fx_level").shift(1) - 1.0
    r_step = (
        pl.when(observed & observed.shift(1))
        .then(pl.col("fut_ret"))
        .otherwise(comp_step)
        .fill_null(0.0)          # first row has no prior; its step is unused (anchor seeds it)
    )
    w = w.sort("date").with_columns(r_step.alias("__r_step"))

    below = w.filter(pl.col("date") < cutoff).with_columns(pl.col("fx_level").alias("blend"))
    above = w.filter(pl.col("date") >= cutoff)
    # Anchor: the last Compustat level strictly before the cutoff (the full fx_sync domain
    # always has one for a G10 currency). If the cutoff is the very first row, anchor on
    # its own Compustat level so blend[cutoff] == fx_level[cutoff] (unconditional chaining).
    if below.height:
        anchor = below.get_column("fx_level").tail(1).item()
    else:
        anchor = above.get_column("fx_level").head(1).item()
        above = above.with_columns(
            pl.when(pl.int_range(pl.len()) == 0).then(0.0).otherwise(pl.col("__r_step")).alias("__r_step")
        )
    above = above.with_columns((anchor * (1.0 + pl.col("__r_step")).cum_prod()).alias("blend"))
    return (
        pl.concat([below.select("date", "blend"), above.select("date", "blend")])
        .sort("date")
    )


def build_sync_fx_panel(
    fx_sync: pl.DataFrame, pre_splice_synced: pl.DataFrame, symbol_to_curcdd: dict[str, str]
) -> pl.DataFrame:
    """Rewrite each G10 currency column of `fx_sync` with its futures-blended level.

    `symbol_to_curcdd` must be fx.SYMBOL_TO_CURCDD_MAPPING (6E->EUR, ...). Passing
    build_currency_map's all-'USD' map overrides zero currencies and RAISES (a silent
    no-op guard). Non-G10 columns and any G10 currency without a futures column pass
    through unchanged.
    """
    fx = fx_sync.sort("date")
    # Mon..Fri only (polars weekday: Mon=1 .. Sun=7). The calendar-dense grid's row before
    # any Monday is Sunday (no future), so a dense single-step would route every Monday to
    # Compustat; the weekday sub-grid makes Fri->Mon one step.
    weekday_dates = fx.select("date").filter(pl.col("date").dt.weekday() <= 5)

    overridden: list[str] = []
    for symbol, expected_ccy in G10_FUTURES.items():
        if symbol_to_curcdd.get(symbol) != expected_ccy:
            continue  # the map disagrees (e.g. all-'USD' build_currency_map) -> never blend
        ccy = expected_ccy
        if ccy not in fx.columns or symbol not in pre_splice_synced.columns:
            logger.warning("sync_fx: skip %s/%s (missing column)", symbol, ccy)
            continue
        cutoff = first_finite_date(pre_splice_synced, symbol)
        if cutoff is None:
            logger.warning("sync_fx: skip %s/%s (no finite futures return)", symbol, ccy)
            continue
        w = (
            weekday_dates
            .join(fx.select("date", pl.col(ccy).cast(pl.Float64, strict=False).alias("fx_level")),
                  on="date", how="left")
            .join(pre_splice_synced.select("date", pl.col(symbol).cast(pl.Float64, strict=False).alias("fut_ret")),
                  on="date", how="left")
            .sort("date")
        )
        blended = _blend_currency_level(w, cutoff)
        full = (
            fx.select("date").join(blended, on="date", how="left")
            .with_columns(pl.col("blend").forward_fill())
        )
        # Below the cutoff the panel is pure Compustat spot VERBATIM (incl. distinct weekend
        # quotes that are real prints, not forward-fills); only from the cutoff onward does the
        # column take the futures-blended (weekend-ffilled) level.
        fx = fx.with_columns(
            pl.when(pl.col("date") >= cutoff)
            .then(full.get_column("blend"))
            .otherwise(pl.col(ccy))
            .alias(ccy)
        )
        overridden.append(ccy)

    if not overridden:
        raise ValueError(
            "build_sync_fx_panel overrode zero currencies -- the symbol_to_curcdd map "
            "disagrees with every G10 future (a build_currency_map all-'USD' map?)."
        )
    logger.info("sync_fx: blended G10 currencies %s", overridden)
    return fx
