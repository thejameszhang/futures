from pathlib import Path

import numpy as np
import polars as pl
import pytest
from globalmacro.validation.plots import _cum_and_corr, plot_comparison, plot_symbol_counts, plot_paired_bars


def test_cum_and_corr_identical_series_is_perfectly_correlated():
    o = np.array([0.10, -0.05, 0.02, 0.08])
    t = o.copy()
    cum_o, cum_t, r = _cum_and_corr(o, t)
    assert r == pytest.approx(1.0)
    assert cum_o[0] == pytest.approx(np.log1p(0.10))
    assert cum_o[-1] == pytest.approx(np.cumsum(np.log1p(o))[-1])
    np.testing.assert_allclose(cum_o, cum_t)


def test_plot_comparison_writes_nontrivial_pdf(tmp_path):
    months = pl.date_range(pl.date(2000, 1, 1), pl.date(2001, 6, 1), interval="1mo", eager=True)
    rows = []
    for inst, name in [("AA", "Alpha"), ("BB", "Beta")]:
        for i, m in enumerate(months):
            rows.append({"instrument": inst, "name": name, "month": m,
                         "ours": 0.01 * (i % 3 - 1), "theirs": 0.01 * (i % 3 - 1)})
    pairs = pl.DataFrame(rows)
    out = tmp_path / "comparison.pdf"
    plot_comparison(pairs, ("ours", "reference"), "Test", out)
    assert out.exists() and out.stat().st_size > 1000


def test_plot_symbol_counts_writes_nontrivial_pdf(tmp_path):
    dates = pl.date_range(pl.date(2000, 1, 1), pl.date(2000, 1, 10), interval="1d", eager=True)
    daily = pl.DataFrame({"date": dates, "AA": [1.0] * 10, "BB": [None] + [2.0] * 9})
    out = tmp_path / "counts.pdf"
    plot_symbol_counts(daily, "Test counts", out)
    assert out.exists() and out.stat().st_size > 1000


def test_datastream_pairs_shape_and_grade_unchanged():
    from globalmacro.validation.datastream_comparison import _datastream_pairs, _datastream_correlations
    pairs = _datastream_pairs()
    assert set(pairs.columns) == {"instrument", "name", "month", "ours", "theirs"}
    assert pairs.height > 0
    corr = _datastream_correlations()
    import numpy as np
    med = float(np.nanmedian(corr.get_column("correlation").to_numpy()))
    assert med == pytest.approx(0.9997, abs=5e-4)   # graded median must not move


def test_consistency_pairs_tier1_and_grade_unchanged():
    from globalmacro.validation.consistency import _consistency_pairs, _consistency_correlations
    pairs = _consistency_pairs()
    assert set(pairs.columns) == {"instrument", "name", "month", "ours", "theirs"}
    assert pairs.height > 0
    import numpy as np
    med = float(np.nanmedian(_consistency_correlations().get_column("correlation").to_numpy()))
    # Tier-2 grade unchanged by this task (run() untouched). Tolerance is wide
    # because the pipeline's async stage is non-deterministic run-to-run, so this
    # median legitimately drifts ~0.9778–0.9789 independent of any code here.
    assert med == pytest.approx(0.9783, abs=3e-3)





def test_plot_paired_bars_writes_a_pdf(tmp_path):
    df = pl.DataFrame({
        "grp": ["sync", "sync", "async", "async"],
        "sym": ["6A", "6B", "6A", "6B"],
        "l": [0.64, 0.67, 0.91, 0.90],
        "r": [0.87, 0.85, 0.81, 0.82],
    })
    path = Path(tmp_path) / "bars.pdf"
    plot_paired_bars(df, group_col="grp", label_col="sym", left_col="l", right_col="r",
                     series_labels=("left", "right"), title="t", ylabel="corr", path=path)
    assert path.exists() and path.stat().st_size > 0


def test_plot_paired_bars_tolerates_a_single_group(tmp_path):
    df = pl.DataFrame({"grp": ["only"], "sym": ["X"], "l": [0.3], "r": [0.6]})
    path = Path(tmp_path) / "one.pdf"
    plot_paired_bars(df, group_col="grp", label_col="sym", left_col="l", right_col="r",
                     series_labels=("a", "b"), title="t", ylabel="c", path=path)
    assert path.exists()
