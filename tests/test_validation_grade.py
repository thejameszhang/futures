from datetime import date
from pathlib import Path

import polars as pl

from globalmacro.validation.base import (
    PASS_THRESHOLD,
    Check,
    Invariant,
    grade,
    write_summary,
)
from globalmacro.validation.consistency import consistency_correlations


def _df(corrs):
    return pl.DataFrame({"instrument": [f"X{i}" for i in range(len(corrs))],
                         "correlation": corrs,
                         "n_obs": [100] * len(corrs)})

def test_grade_passes_when_median_at_or_above_threshold():
    r = grade("Demo", "demo", _df([0.99, 0.96, 0.95, 0.80]))
    assert r.n == 4
    assert abs(r.median - 0.955) < 1e-9
    assert r.n_below == 0          # flag threshold is 0.80, exclusive
    assert r.passed is True

def test_grade_fails_when_median_below_threshold_and_counts_flags():
    r = grade("Demo", "demo", _df([0.99, 0.90, 0.70, 0.60]))
    assert r.passed is False       # median 0.80 < 0.95
    assert r.n_below == 2          # 0.70, 0.60 below 0.80

def test_grade_ignores_null_and_nan_correlations():
    r = grade("Demo", "demo", _df([0.99, None, float("nan"), 0.97]))
    assert r.n == 2
    assert r.passed is True

def test_grade_passes_exactly_at_threshold():
    # pins the >= boundary: median exactly PASS_THRESHOLD must PASS (a `>` bug would fail this)
    r = grade("Demo", "demo", _df([PASS_THRESHOLD, PASS_THRESHOLD, PASS_THRESHOLD]))
    assert abs(r.median - PASS_THRESHOLD) < 1e-12
    assert r.passed is True



def test_consistency_correlations_matches_shared_symbols():
    import polars as pl
    dates = [date(2020,1,d) for d in range(1,6)]
    synced = pl.DataFrame({"date": dates, "AA": [0.01,-0.02,0.03,-0.01,0.02], "BB":[0.0,0.0,0.0,0.0,0.0]})
    asynced = pl.DataFrame({"date": dates, "AA": [0.01,-0.02,0.03,-0.01,0.02], "CC":[0.1]*5})
    # freq="daily" exercises the correlation core directly on identical series
    out = consistency_correlations(synced, asynced, freq="daily")
    assert out.filter(pl.col("instrument")=="AA")["correlation"].item() > 0.999
    assert set(out["instrument"].to_list()) == {"AA"}   # only shared symbol

def test_consistency_correlations_monthly_path():
    import polars as pl
    months = [date(2020, m, 1) for m in range(1, 6)]     # 5 month-start dates
    vals = [0.03, -0.01, 0.05, 0.02, -0.04]
    a = pl.DataFrame({"date": months, "AA": vals})
    b = pl.DataFrame({"date": months, "AA": vals})
    # freq="monthly": dt.truncate is a no-op on month-starts, period join is 1:1
    out = consistency_correlations(a, b, freq="monthly")
    assert out.filter(pl.col("instrument")=="AA")["correlation"].item() > 0.999




def test_check_defaults_leave_existing_checks_untouched():
    c = Check(name="Demo", slug="demo", run=lambda: _df([0.99]))
    assert c.invariants is None
    assert c.figures is None


def test_write_summary_renders_invariants_table(tmp_path):
    results = [grade("Demo", "demo", _df([0.99, 0.97]))]
    invariants = [
        Invariant(check="Demo", name="diagonal holds", value="8/8", passed=True),
        Invariant(check="Demo", name="level agreement", value="max 0.19%", passed=False),
    ]
    path = Path(tmp_path) / "SUMMARY.md"
    write_summary(results, invariants, path)
    text = path.read_text()
    assert "## Invariants" in text
    assert "diagonal holds" in text and "8/8" in text
    assert "level agreement" in text
    assert "❌" in text and "✅" in text


def test_write_summary_omits_invariants_table_when_none(tmp_path):
    path = Path(tmp_path) / "SUMMARY.md"
    write_summary([grade("Demo", "demo", _df([0.99]))], [], path)
    assert "## Invariants" not in path.read_text()


def test_write_summary_default_skipped_matches_three_positional_call(tmp_path):
    """Task 5 appends `skipped` with a default -- the pre-existing three-positional
    call above must keep working unchanged."""
    path = Path(tmp_path) / "SUMMARY.md"
    write_summary([grade("Demo", "demo", _df([0.99]))], [], path)
    assert "## Skipped" not in path.read_text()


def test_write_summary_renders_skipped_section(tmp_path):
    path = Path(tmp_path) / "SUMMARY.md"
    write_summary(
        [grade("Demo", "demo", _df([0.99]))], [], path,
        ["Async vs sync consistency", "External ground-truth cross-check"],
    )
    text = path.read_text()
    assert "## Skipped" in text
    assert "Async vs sync consistency" in text
    assert "External ground-truth cross-check" in text
    assert text.count("SKIPPED (async-only run)") == 2
