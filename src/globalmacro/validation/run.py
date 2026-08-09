# src/globalmacro/validation/run.py
"""globalmacro validate: run every available data-QA check against the built
datasets, grade each identically, render its comparison.pdf, write per-dataset
symbol-count PDFs, and write one VALIDATION_SUMMARY.md."""
import argparse
import os

import polars as pl

from globalmacro.utils.capabilities import (
    resolve_mode,
    sync_panels_fresh,
    sync_panels_ready,
)
from globalmacro.utils.paths import DATASETS_ROOT, VALIDATION_OUTPUT
from globalmacro.validation.base import Invariant, grade, write_summary
from globalmacro.validation.consistency import consistency_check
from globalmacro.validation.datastream_comparison import datastream_check
from globalmacro.validation.fx_futures import fx_futures_vs_spot_check
from globalmacro.validation.plots import plot_comparison, plot_symbol_counts
from globalmacro.validation.spot_fx import spot_fx_check
from globalmacro.validation.synthetic_equity import synthetic_equity_check
from globalmacro.validation.synthetic_fx import synthetic_fx_check


def _available_checks(mode: str = "full"):
    checks = [datastream_check, consistency_check, synthetic_fx_check,
              synthetic_equity_check, spot_fx_check, fx_futures_vs_spot_check]
    try:
        from globalmacro.validation.external_comparison import external_check
        checks.append(external_check)          # optional, gitignored, local-only
    except ImportError:
        pass
    if mode == "async-only":
        return [c for c in checks if not c.requires_sync]
    return checks


def _skipped_checks(mode: str) -> list[str]:
    if mode != "async-only":
        return []
    kept = {c.slug for c in _available_checks("async-only")}
    return [c.name for c in _available_checks("full") if c.slug not in kept]


def _load_wide(path):
    df = pl.read_csv(str(path), infer_schema_length=0)
    return df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))


_SYMBOL_COUNT_SOURCES = [
    ("tier1_sync_daily",   DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv",
     "Tier-1 sync daily — number of instruments present per date"),
    ("tier1_async_daily",  DATASETS_ROOT / "tier1" / "async" / "async_daily.csv",
     "Tier-1 async daily — number of instruments present per date"),
    ("tier2_async_monthly", DATASETS_ROOT / "tier2" / "async" / "async_monthly.csv",
     "Tier-2 async monthly — number of instruments present per month"),
]


def _symbol_count_sources(mode: str = "full"):
    # Match the parent directory NAME, not the stem or a path substring. A stem filter
    # (e.g. "sync" in "tier1_async_daily") would silently drop every source, since
    # "sync" is a substring of "async". A path-substring filter ("/sync/" in the full
    # path) has the same failure mode one level up: DATASETS_ROOT is env-configurable
    # (paths.py, FUTURES_DATASETS_ROOT) and can itself contain a "sync" path component
    # (e.g. a relocated data directory under .../sync/datasets) -- every source's path
    # would then contain "/sync/" and every source would be silently dropped.
    if mode == "async-only":
        return [s for s in _SYMBOL_COUNT_SOURCES if s[1].parent.name != "sync"]
    return _SYMBOL_COUNT_SOURCES


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="globalmacro validate")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--async-only", dest="mode", action="store_const", const="async-only",
                   help="run only the checks that don't require the sync panels")
    g.add_argument("--full", dest="mode", action="store_const", const="full",
                   help="require the sync panels; fail rather than silently degrade")
    p.set_defaults(mode=None)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    mode = resolve_mode(args.mode, sync_panels_ready(), "validate")
    if mode == "full":
        # sync_panels_ready() only checked existence. async-only builds rewrite the
        # async panels on every run but leave any previously-built sync panels
        # untouched, so a machine that ran full mode once and async-only since would
        # otherwise grade fresh async panels against stale sync ones and report a
        # confident, wrong verdict. Refuse rather than risk that -- see
        # capabilities.sync_panels_fresh for the mtime rule.
        freshness = sync_panels_fresh()
        if not freshness.ready:
            raise SystemExit(f"globalmacro validate: {freshness.message}")
    os.makedirs(VALIDATION_OUTPUT, exist_ok=True)
    results = []
    invariants: list[Invariant] = []
    for check in _available_checks(mode):
        correlations = check.run()
        out_dir = VALIDATION_OUTPUT / check.slug
        os.makedirs(out_dir, exist_ok=True)
        correlations.write_csv(str(out_dir / "correlations.csv"))   # full universe
        # Grade only what ships. Diagnostic rows (used=false) stay in the CSV but must
        # never move the median -- grading a series no reader receives is meaningless.
        graded = (
            correlations.filter(pl.col("used"))
            if "used" in correlations.columns
            else correlations
        )
        r = grade(check.name, check.slug, graded)
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:32s} n={r.n:4d} median={r.median:.4f} "
              f"mean={r.mean:.4f} min={r.minimum:.4f} (<0.80: {r.n_below})")

        if check.invariants is not None:
            for inv in check.invariants():
                invariants.append(inv)
                tag = "PASS" if inv.passed else "FAIL"
                print(f"        [{tag}] {inv.name}: {inv.value}")

        if check.pairs is not None:
            # Plotting is a secondary deliverable — never let a render failure
            # abort grading or the summary write.
            try:
                plot_comparison(
                    check.pairs(), check.series_labels,
                    f"{check.name}: cumulative monthly log returns, per Tier-1 asset",
                    out_dir / "comparison.pdf",
                )
                print(f"        wrote {out_dir / 'comparison.pdf'}")
            except Exception as e:
                print(f"        WARN: comparison.pdf failed for {check.slug}: {e}")

        if check.figures is not None:
            try:
                check.figures(out_dir)
            except Exception as e:
                print(f"        WARN: figures failed for {check.slug}: {e}")

    sc_dir = VALIDATION_OUTPUT / "symbol_counts"
    os.makedirs(sc_dir, exist_ok=True)
    for stem, src, title in _symbol_count_sources(mode):
        try:
            plot_symbol_counts(_load_wide(src), title, sc_dir / f"{stem}.pdf")
            print(f"        wrote {sc_dir / stem}.pdf")
        except Exception as e:
            print(f"        WARN: symbol_counts {stem} failed: {e}")

    skipped = _skipped_checks(mode)
    for name in skipped:
        print(f"[SKIP] {name:32s} SKIPPED (async-only run)")

    write_summary(results, invariants, VALIDATION_OUTPUT / "VALIDATION_SUMMARY.md", skipped)
    print(f"\nSummary written to {VALIDATION_OUTPUT / 'VALIDATION_SUMMARY.md'}")
    ok = all(r.passed for r in results) and all(i.passed for i in invariants)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
