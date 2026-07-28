# src/globalmacro/validation/plots.py
"""Rendering for the validation deliverables. The only matplotlib in the
package; Agg backend (headless HPC). Renderers:
  - plot_comparison: small-multiples cumulative monthly log returns, ours vs
    theirs, one panel per Tier-1 instrument (title = name (ticker) + r).
  - plot_fx_vs_spot_grid: sibling of plot_comparison -- small-multiples
    cumulative DAILY log returns, one panel per currency, THREE lines (ET
    futures / London futures / Compustat spot) + two correlation annotations.
  - plot_symbol_counts: instruments present per date (data-completeness).
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
from matplotlib.lines import Line2D

_C_OURS, _C_THEIRS = "#1f77b4", "#ff7f0e"
_MIN_MONTHS = 12
_NCOLS = 6


def _cum_and_corr(ours: np.ndarray, theirs: np.ndarray):
    """Cumulative log returns of each series + their Pearson r (nan if <2 pts
    or zero variance)."""
    cum_o = np.cumsum(np.log1p(ours))
    cum_t = np.cumsum(np.log1p(theirs))
    r = float(np.corrcoef(ours, theirs)[0, 1]) if len(ours) > 1 else float("nan")
    return cum_o, cum_t, r


def plot_comparison(pairs: pl.DataFrame, series_labels, title: str, path) -> None:
    panels = []
    for key, grp in pairs.sort("month").group_by(["instrument"], maintain_order=True):
        inst = key[0]
        g = grp.drop_nulls(["ours", "theirs"])
        if g.height < _MIN_MONTHS:
            continue
        name = grp.get_column("name")[0]
        months = g.get_column("month").to_numpy()
        o = g.get_column("ours").to_numpy().astype(float)
        t = g.get_column("theirs").to_numpy().astype(float)
        cum_o, cum_t, r = _cum_and_corr(o, t)
        panels.append((str(inst), str(name), r, months, cum_o, cum_t))
    panels.sort(key=lambda x: (x[1] or "").lower())

    n = len(panels)
    nrows = max(1, math.ceil(n / _NCOLS))
    fig, axes = plt.subplots(nrows, _NCOLS, figsize=(_NCOLS * 2.7, nrows * 1.85), squeeze=False)
    axes = axes.flatten()
    for ax, (inst, name, r, months, cum_o, cum_t) in zip(axes, panels, strict=False):
        ax.plot(months, cum_o, color=_C_OURS, lw=0.9)
        ax.plot(months, cum_t, color=_C_THEIRS, lw=0.9)
        ax.set_title(f"{name} ({inst})", fontsize=6.2, pad=2)
        rtxt = "r = n/a" if math.isnan(r) else f"r = {r:.2f}"
        ax.text(0.035, 0.93, rtxt, transform=ax.transAxes, fontsize=5.8, va="top",
                color="#c0392b" if (not math.isnan(r) and r < 0.8) else "#444")
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=4.5, length=2)
        ax.xaxis.set_major_locator(YearLocator(10))
        ax.xaxis.set_major_formatter(DateFormatter("%y"))
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend([Line2D([], [], color=_C_OURS, lw=1.5),
                Line2D([], [], color=_C_THEIRS, lw=1.5)],
               list(series_labels), loc="upper right", fontsize=8, frameon=False,
               ncol=2, bbox_to_anchor=(0.995, 0.997))
    fig.suptitle(title, fontsize=12, x=0.02, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


_C_ET, _C_LONDON, _C_SPOT = "#1f77b4", "#ff7f0e", "#444444"


def _corr_label(prefix: str, r: float | None) -> tuple[str, str]:
    """`"et r=0.83"` (or `"et r=n/a"`) + its color -- red if <0.8, matching
    `plot_comparison`'s single-annotation convention."""
    if r is None or (isinstance(r, float) and math.isnan(r)):
        return f"{prefix} r=n/a", "#444"
    return f"{prefix} r={r:.2f}", ("#c0392b" if r < 0.8 else "#444")


def plot_fx_vs_spot_grid(panels: list[dict], title: str, path) -> None:
    """Small-multiples grid, one panel per currency, THREE cumulative-daily-log-return
    lines (ET futures / London futures / Compustat spot) with both correlations
    annotated. A sibling of `plot_comparison` -- same grid/sizing/font/legend
    conventions -- but 3 lines/panel and 2 annotations instead of 1.

    `panels` is a list of per-currency records (built by
    `fx_futures._fx_vs_spot_panels`):
    `{"ccy", "symbol", "dates", "cum_et", "cum_lon", "cum_spot", "r_et", "r_lon"}`.
    `cum_et`/`cum_lon` (and `r_et`/`r_lon`) are `None` when that mode has no panel for
    the currency -- the line and its annotation are then simply skipped, rather than
    drawing a misleading flat-zero line. `cum_spot` is always present.
    """
    panels = sorted(panels, key=lambda p: p["ccy"])
    n = len(panels)
    nrows = max(1, math.ceil(n / _NCOLS))
    fig, axes = plt.subplots(nrows, _NCOLS, figsize=(_NCOLS * 2.7, nrows * 1.85), squeeze=False)
    axes = axes.flatten()
    for ax, p in zip(axes, panels, strict=False):
        dates = p["dates"]
        if p["cum_et"] is not None:
            ax.plot(dates, p["cum_et"], color=_C_ET, lw=0.9)
        if p["cum_lon"] is not None:
            ax.plot(dates, p["cum_lon"], color=_C_LONDON, lw=0.9)
        ax.plot(dates, p["cum_spot"], color=_C_SPOT, lw=1.0)
        ax.set_title(f"{p['ccy']} ({p['symbol']})", fontsize=6.2, pad=2)
        for i, (prefix, r) in enumerate((("et", p["r_et"]), ("lon", p["r_lon"]))):
            txt, color = _corr_label(prefix, r)
            ax.text(0.035, 0.93 - i * 0.17, txt, transform=ax.transAxes, fontsize=5.6,
                    va="top", color=color)
        ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=4.5, length=2)
        ax.xaxis.set_major_locator(YearLocator(5))
        ax.xaxis.set_major_formatter(DateFormatter("%y"))
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    for ax in axes[n:]:
        ax.axis("off")
    fig.legend([Line2D([], [], color=_C_ET, lw=1.5),
                Line2D([], [], color=_C_LONDON, lw=1.5),
                Line2D([], [], color=_C_SPOT, lw=1.5)],
               ["ET futures", "London futures", "Compustat spot"],
               loc="upper right", fontsize=8, frameon=False,
               ncol=3, bbox_to_anchor=(0.995, 0.997))
    fig.suptitle(title, fontsize=12, x=0.02, ha="left", y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def plot_symbol_counts(daily: pl.DataFrame, title: str, path) -> None:
    cols = [c for c in daily.columns if c != "date"]
    present = daily.select(
        pl.col("date"),
        pl.sum_horizontal(
            [(pl.col(c).cast(pl.Float64, strict=False).is_not_null()
              & pl.col(c).cast(pl.Float64, strict=False).is_not_nan()).cast(pl.Int32)
             for c in cols]
        ).alias("n"),
    ).sort("date")
    dates = present.get_column("date").to_numpy()
    counts = present.get_column("n").to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(dates, counts, step="pre", color=_C_OURS, alpha=0.25)
    ax.plot(dates, counts, color=_C_OURS, lw=1.0)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("# instruments")
    ax.set_xlabel("date")
    ax.set_ylim(0, float(counts.max()) * 1.08 if len(counts) else 1.0)
    ax.grid(alpha=0.3, lw=0.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path))
    plt.close(fig)


def plot_paired_bars(
    df: pl.DataFrame,
    *,
    group_col: str,
    label_col: str,
    left_col: str,
    right_col: str,
    series_labels,
    title: str,
    ylabel: str,
    path,
) -> None:
    """Side-by-side bars per label, one subplot per group.

    Used for the two justification figures, where the whole argument is 'the right bar is
    taller than the left one, for every symbol, in this panel and the other way round in
    that one'. A bar chart makes that legible at a glance; a line chart would not.
    """
    groups = list(dict.fromkeys(df.get_column(group_col).to_list()))
    fig, axes = plt.subplots(
        1, len(groups), figsize=(7.5 * len(groups), 4.6), squeeze=False
    )
    for ax, group in zip(axes[0], groups, strict=False):
        sub = df.filter(pl.col(group_col) == group)
        labels = sub.get_column(label_col).to_list()
        left = [0.0 if v is None else float(v) for v in sub.get_column(left_col).to_list()]
        right = [0.0 if v is None else float(v) for v in sub.get_column(right_col).to_list()]
        x = np.arange(len(labels))
        width = 0.38
        ax.bar(x - width / 2, left, width, label=series_labels[0], color=_C_OURS)
        ax.bar(x + width / 2, right, width, label=series_labels[1], color=_C_THEIRS)
        ax.axhline(0.0, color="0.4", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_title(str(group), fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
