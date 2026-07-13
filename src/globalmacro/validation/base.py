# src/globalmacro/validation/base.py
"""Shared machinery for the standardized validation checks.

Every validation exercise is the same operation: correlate our per-instrument
return series against a reference series, then grade by the distribution of
per-instrument Pearson correlations. A check returns a frame with columns
[instrument, correlation, n_obs]; grade() reduces it to a CheckResult and
write_summary() renders all results into VALIDATION_SUMMARY.md.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import polars as pl

PASS_THRESHOLD = 0.95   # PASS iff median per-instrument correlation >= this
FLAG_THRESHOLD = 0.80   # diagnostic only: count instruments below this


@dataclass
class CheckResult:
    name: str
    slug: str
    n: int
    mean: float
    median: float
    minimum: float
    n_below: int
    passed: bool


@dataclass
class Invariant:
    """A pass/fail assertion that is NOT a correlation.

    Correlation is the wrong instrument for some of the sharpest statements a check
    can make -- "Compustat beats Datastream on the sync futures, 8/8" or "the two FX
    panels agree in level to 0.19%". Those are invariants: they hold or they don't.
    A failing invariant fails the run.
    """
    check: str      # display name of the owning Check; write_summary groups on this
    name: str       # e.g. "Compustat beats Datastream on sync futures"
    value: str      # pre-rendered, e.g. "8/8" or "max 0.19%"
    passed: bool


@dataclass
class Check:
    name: str                       # display name, e.g. "Datastream comparison"
    slug: str                       # output subdir, e.g. "datastream"
    run: Callable[[], pl.DataFrame]  # returns columns: instrument, correlation, n_obs
    # Optional Tier-1 comparison-plot support: pairs() returns long-form
    # [instrument, name, month, ours, theirs]; series_labels names the two lines.
    pairs: "Callable[[], pl.DataFrame] | None" = None
    series_labels: tuple[str, str] = ("ours", "reference")
    # Optional non-correlation assertions; a failure fails the run.
    invariants: "Callable[[], list[Invariant]] | None" = None
    # Optional one-off justification figures; called with the check's out_dir.
    figures: "Callable[[Path], None] | None" = None


def grade(name: str, slug: str, correlations: pl.DataFrame) -> CheckResult:
    c = correlations.get_column("correlation").cast(pl.Float64, strict=False)
    c = c.filter(c.is_not_null() & c.is_not_nan())
    n = c.len()
    if n == 0:
        return CheckResult(name, slug, 0, float("nan"), float("nan"),
                           float("nan"), 0, False)
    median = float(c.median())
    return CheckResult(
        name=name, slug=slug, n=n,
        mean=float(c.mean()), median=median, minimum=float(c.min()),
        n_below=int((c < FLAG_THRESHOLD).sum()),
        passed=median >= PASS_THRESHOLD,
    )


def write_summary(results: list[CheckResult], invariants: list[Invariant], path: Path) -> None:
    lines = [
        "# Validation Summary",
        "",
        f"Grade: PASS iff median per-instrument correlation ≥ {PASS_THRESHOLD:.2f}.",
        "",
        "| Exercise | n | mean | median | min | # < 0.80 | Result |",
        "|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in results:
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(
            f"| {r.name} | {r.n} | {r.mean:.4f} | {r.median:.4f} | "
            f"{r.minimum:.4f} | {r.n_below} | {status} |"
        )
    if invariants:
        lines += [
            "",
            "## Invariants",
            "",
            "Assertions that are not correlations. A failure fails the run.",
            "",
            "| Exercise | Invariant | Value | Result |",
            "|---|---|---:|:--:|",
        ]
        for i in invariants:
            status = "✅ PASS" if i.passed else "❌ FAIL"
            lines.append(f"| {i.check} | {i.name} | {i.value} | {status} |")
    lines.append("")
    path.write_text("\n".join(lines))
