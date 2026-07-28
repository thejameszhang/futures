# src/globalmacro/validation/run.py
"""globalmacro validate: run every available data-QA check against the built
datasets, grade each identically, render its comparison.pdf, write per-dataset
symbol-count PDFs, and write one VALIDATION_SUMMARY.md."""
import argparse
import os

import polars as pl

from globalmacro.utils.paths import DATASETS_ROOT, VALIDATION_OUTPUT
from globalmacro.validation.base import Invariant, grade, write_summary
from globalmacro.validation.consistency import consistency_check
from globalmacro.validation.datastream_comparison import datastream_check
from globalmacro.validation.plots import plot_comparison, plot_symbol_counts
from globalmacro.validation.spot_fx import spot_fx_check
from globalmacro.validation.synthetic_equity import synthetic_equity_check
from globalmacro.validation.synthetic_fx import synthetic_fx_check


def _available_checks():
    checks = [datastream_check, consistency_check, synthetic_fx_check,
              synthetic_equity_check, spot_fx_check]
    try:
        from globalmacro.validation.external_comparison import external_check
        checks.append(external_check)          # optional, gitignored, local-only
    except ImportError:
        pass
    return checks


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


def main(argv=None):
    argparse.ArgumentParser(prog="globalmacro validate").parse_args(argv)
    os.makedirs(VALIDATION_OUTPUT, exist_ok=True)
    results = []
    invariants: list[Invariant] = []
    for check in _available_checks():
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
    for stem, src, title in _SYMBOL_COUNT_SOURCES:
        try:
            plot_symbol_counts(_load_wide(src), title, sc_dir / f"{stem}.pdf")
            print(f"        wrote {sc_dir / stem}.pdf")
        except Exception as e:
            print(f"        WARN: symbol_counts {stem} failed: {e}")

    write_summary(results, invariants, VALIDATION_OUTPUT / "VALIDATION_SUMMARY.md")
    print(f"\nSummary written to {VALIDATION_OUTPUT / 'VALIDATION_SUMMARY.md'}")
    ok = all(r.passed for r in results) and all(i.passed for i in invariants)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
