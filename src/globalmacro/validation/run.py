# src/globalmacro/validation/run.py
"""globalmacro validate: run every available data-QA check against the built
datasets, grade each identically, render its comparison.pdf, write per-dataset
symbol-count PDFs, and write one VALIDATION_SUMMARY.md."""
import argparse
import dataclasses
import os
from pathlib import Path

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
from globalmacro.validation.mode import validation_mode
from globalmacro.validation.plots import plot_comparison, plot_symbol_counts
from globalmacro.validation.spot_fx import spot_fx_check
from globalmacro.validation.synthetic_equity import synthetic_equity_check
from globalmacro.validation.synthetic_fx import synthetic_fx_check

# The literal filename check.pairs's render (via plot_comparison below) writes to,
# and the convention every check.figures implementation in this codebase also
# follows for its primary comparison figure (see fx_futures.plot_fx_vs_spot_comparison).
# Named once here so F9's stale-figure cleanup references the SAME string the write
# site uses, rather than a second, independently-typed literal that could drift.
_COMPARISON_PDF = "comparison.pdf"


def _available_checks(mode: str = "full"):
    checks = [datastream_check, consistency_check, synthetic_fx_check,
              synthetic_equity_check, spot_fx_check, fx_futures_vs_spot_check]
    try:
        from globalmacro.validation.external_comparison import external_check
        # F4: external_comparison.py is gitignored and untracked (confidential-
        # adjacent, .gitignore:34) -- it CANNOT carry requires_sync=True itself and
        # have that travel with the merge; a local edit to a gitignored file is
        # invisible to `git status` and to review, and any clone/worktree with a
        # different (or default requires_sync=False) copy would run this check in
        # async-only mode and try to read tier1/sync/sync_daily.csv, which doesn't
        # exist there. State the requirement here instead, in tracked code -- the
        # only place it can actually travel with the branch. Check is a plain,
        # non-frozen dataclass, so dataclasses.replace() works.
        checks.append(dataclasses.replace(external_check, requires_sync=True))
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


def _dropped_invariants(mode: str) -> list[str]:
    """Invariant names that the checks still RUNNING in this mode omit -- distinct from
    _skipped_checks, which names checks that don't run at all. Static per-Check data
    (Check.dropped_invariants), not derived by re-running anything in full mode."""
    if mode != "async-only":
        return []
    return [name for c in _available_checks(mode) for name in c.dropped_invariants]


def _dropped_figures(mode: str) -> list[str]:
    """Sibling of _dropped_invariants, for figure filenames."""
    if mode != "async-only":
        return []
    return [name for c in _available_checks(mode) for name in c.dropped_figures]


def _stale_figure_paths(mode: str) -> list[Path]:
    """F9: after a full run, an async-only run's own VALIDATION_SUMMARY.md calls a
    figure SKIPPED while it may still be sitting on disk from that earlier full run
    -- the summary says one thing, the tree says another. Names the on-disk paths
    that must not survive an async-only run, from two sources ONLY (both already
    tracked, static, per-Check data -- nothing here is derived by running anything):

      * a check dropped WHOLE (requires_sync=True, named under _skipped_checks) --
        its comparison.pdf, written either via check.pairs (this module's own
        plot_comparison(..., out_dir / _COMPARISON_PDF) call above) or a custom
        check.figures whose output filename this codebase's convention also names
        _COMPARISON_PDF (fx_futures.plot_fx_vs_spot_comparison).
      * a check still RUNNING in this mode but dropping some of its own figures --
        each name in that check's own declared Check.dropped_figures (see
        _dropped_figures above).

    Full mode returns []: nothing here may ever touch a file when nothing was
    actually skipped.
    """
    if mode != "async-only":
        return []
    kept_slugs = {c.slug for c in _available_checks(mode)}
    paths: list[Path] = []
    for check in _available_checks("full"):
        out_dir = VALIDATION_OUTPUT / check.slug
        if check.slug not in kept_slugs:
            if check.pairs is not None or check.figures is not None:
                paths.append(out_dir / _COMPARISON_PDF)
        else:
            for fig_name in check.dropped_figures:
                paths.append(out_dir / fig_name)
    return paths


def _remove_stale_figures(mode: str) -> list[Path]:
    """Unlink exactly _stale_figure_paths(mode) (missing_ok=True: a clean researcher
    tree with no prior full run has nothing to remove). Guarded hard, independent of
    the mode check already inside _stale_figure_paths: every path removed must
    resolve to a bare filename directly inside VALIDATION_OUTPUT/<check-slug>/ --
    never a nested path, a path with a separator in the filename, or anything that
    could resolve outside VALIDATION_OUTPUT. Returns what it removed, for logging."""
    if mode != "async-only":
        return []
    removed = []
    for p in _stale_figure_paths(mode):
        # p must be exactly VALIDATION_OUTPUT/<check-slug>/<bare filename> -- refuse
        # anything else rather than unlink it. _stale_figure_paths only ever builds
        # paths this shape from tracked string literals, so this should never fire;
        # it exists as a second, independent line of defense against a future edit
        # to that function (or to a Check's slug/dropped_figures) introducing a
        # separator or a "..".
        if p.parent.parent != VALIDATION_OUTPUT:
            continue
        if not p.name or "/" in p.name or p.name in (".", ".."):
            continue
        existed = p.exists()
        p.unlink(missing_ok=True)   # tolerant of a genuinely clean tree (no prior full run)
        if existed:
            removed.append(p)
    return removed


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
    # F5a: record which mode produced this run's output, as the first line of
    # stdout. Stdout-only -- does not alter any graded number or write to a file.
    print(f"mode: {mode}")
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
    # Scoped for exactly this loop: the individual checks' _invariants()/_figures() take
    # no arguments (base.Check), so this is how they learn the resolved mode -- an
    # explicit --async-only must suppress their sync-derived output even when the sync
    # panels are present and fresh (see synthetic_fx._grade_sync / synthetic_equity.
    # _grade_sync). The context manager resets validation.mode's module state on the way
    # out, exception or not, so a second main() call in the same process never inherits it.
    with validation_mode(mode):
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
                # F3b: sync_panels_ready()/sync_stage_outputs_ready() (F3a) still don't
                # cover EVERY file this can transitively read -- build_synced_dataset()
                # also reads a hand-placed MANUAL prerequisite (see
                # sync_stage_outputs_ready()'s docstring) that is deliberately excluded
                # from both predicates, so a full-mode resolution can still reach here
                # with a file missing. Unlike `pairs`/`figures` below, this call sits
                # outside any try -- without this, the researcher gets a raw
                # FileNotFoundError traceback, no remedy, and no mention of
                # --async-only. Only FileNotFoundError is caught: a genuine invariant
                # FAILURE is not an exception (it's a False Invariant.passed, handled
                # by the loop below and the final `ok` check) and any OTHER exception
                # is a real bug that must still fail the run loudly.
                try:
                    computed_invariants = check.invariants()
                except FileNotFoundError as e:
                    raise SystemExit(
                        f"globalmacro validate: {check.name} needs a file that is "
                        f"missing ({e}). Either rerun `globalmacro build --full` to "
                        "rebuild the sync half, or run `globalmacro validate "
                        "--async-only` to skip the checks that need it."
                    ) from e
                for inv in computed_invariants:
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
                        out_dir / _COMPARISON_PDF,
                    )
                    print(f"        wrote {out_dir / _COMPARISON_PDF}")
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

    dropped_invariants = _dropped_invariants(mode)
    dropped_figures = _dropped_figures(mode)
    for name in dropped_invariants:
        print(f"[SKIP] invariant: {name} SKIPPED (async-only run)")
    for name in dropped_figures:
        print(f"[SKIP] figure: {name} SKIPPED (async-only run)")

    # F9: an async-only run must not leave stale PDFs from an earlier full run on
    # disk while its own summary calls them SKIPPED -- that is the output actively
    # lying, not merely omitting. No-op in full mode (see _remove_stale_figures).
    for p in _remove_stale_figures(mode):
        print(f"        removed stale {p} (async-only run; from an earlier full run)")

    write_summary(
        results, invariants, VALIDATION_OUTPUT / "VALIDATION_SUMMARY.md",
        skipped, dropped_invariants, dropped_figures,
    )
    print(f"\nSummary written to {VALIDATION_OUTPUT / 'VALIDATION_SUMMARY.md'}")
    ok = all(r.passed for r in results) and all(i.passed for i in invariants)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
