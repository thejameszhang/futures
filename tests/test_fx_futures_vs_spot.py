from datetime import date, timedelta

import polars as pl
import pytest

import globalmacro.validation.fx_futures as fx_futures
from globalmacro.validation.fx_futures import (
    G10_MAJORS,
    daily_fx_vs_spot,
    fx_futures_correlation,
    fx_futures_vs_spot_check,
    plot_fx_vs_spot_comparison,
)


def test_daily_fx_vs_spot_aligns_returns():
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 6)]
    fut = pl.DataFrame({"date": dates, "6E": [0.01, -0.02, 0.0]})
    comp = pl.DataFrame({"date": dates, "EUR": [1.10, 1.10 * 1.01, 1.10 * 1.01 * 0.98]})
    out = daily_fx_vs_spot(fut, comp, "6E", "EUR")
    assert set(out.columns) == {"date", "r_fut", "r_spot"}
    row = out.filter(pl.col("date") == date(2020, 1, 3))
    assert abs(row["r_fut"][0] - (-0.02)) < 1e-9
    assert abs(row["r_spot"][0] - 0.01) < 1e-9


def _perfect_pair(symbol: str, ccy: str, n: int = 35, seed_level: float = 1.20):
    """A synthetic (fut, comp_levels) pair whose Compustat spot level is built by
    compounding the SAME daily returns as the futures panel, one-to-one, with no gaps.
    `daily_fx_vs_spot` therefore recovers r_spot == r_fut for every panel date (up to
    floating-point rounding in the compounding), so pearson/spearman come back ~1.0
    regardless of `symbol`/`ccy`/mode -- isolating the `used` gate (which depends only on
    symbol membership + mode, not on how well the series actually track) from correlation
    magnitude.
    """
    base_returns = [0.0020, -0.0015, 0.0010, -0.0008, 0.0022, -0.0011, 0.0009]
    returns = (base_returns * ((n // len(base_returns)) + 1))[:n]
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n + 1)]  # seed date + n panel dates
    levels = [seed_level]
    for r in returns:
        levels.append(levels[-1] * (1 + r))
    fut = pl.DataFrame({"date": dates[1:], symbol: returns})
    comp = pl.DataFrame({"date": dates, ccy: levels})
    return fut, comp


def test_fx_futures_correlation_g10_london_is_used():
    fut, comp = _perfect_pair("6E", "EUR")
    row = fx_futures_correlation(fut, comp, "6E", "EUR", "london")
    assert row["instrument"] == "6E (london)"
    assert row["n_obs"] == 35
    assert row["correlation"] is not None
    assert row["correlation"] > 0.999
    assert row["used"] is True


def test_fx_futures_correlation_g10_et_is_not_used():
    # Same symbol, same near-perfect correlation as the london case above -- only `mode`
    # differs. ET is never graded, no matter how well it tracks.
    fut, comp = _perfect_pair("6E", "EUR")
    row = fx_futures_correlation(fut, comp, "6E", "EUR", "et")
    assert row["instrument"] == "6E (et)"
    assert row["correlation"] > 0.999
    assert row["used"] is False


def test_fx_futures_correlation_em_symbol_is_not_used_in_either_mode():
    # KRW is not in G10_MAJORS -- even a near-perfect synthetic correlation must not be
    # graded, in london OR et mode.
    assert "KRW" not in G10_MAJORS
    fut, comp = _perfect_pair("KRW", "KRW")
    for mode in ("et", "london"):
        row = fx_futures_correlation(fut, comp, "KRW", "KRW", mode)
        assert row["correlation"] > 0.999
        assert row["used"] is False


def test_fx_futures_correlation_below_min_obs_is_not_used():
    fut, comp = _perfect_pair("6E", "EUR", n=10)   # below the n_obs >= 30 floor
    row = fx_futures_correlation(fut, comp, "6E", "EUR", "london")
    assert row["n_obs"] == 10
    assert row["correlation"] is None
    assert row["spearman"] is None
    assert row["used"] is False


def test_fx_futures_correlation_row_has_expected_columns():
    fut, comp = _perfect_pair("6E", "EUR")
    row = fx_futures_correlation(fut, comp, "6E", "EUR", "london")
    assert set(row.keys()) == {"instrument", "correlation", "spearman", "n_obs", "used"}


def test_fx_futures_vs_spot_check_is_registered_correctly():
    assert fx_futures_vs_spot_check.slug == "fx_futures_vs_spot"
    assert fx_futures_vs_spot_check.name == "Futures vs Compustat spot (daily)"
    assert callable(fx_futures_vs_spot_check.run)
    assert fx_futures_vs_spot_check.figures is plot_fx_vs_spot_comparison


def _synthetic_comp_and_futures(symbol="6E", ccy="EUR", n=40, seed_level=1.10):
    """A synthetic (comp_levels, fut_et, fut_london) triple: `comp_levels` compounds a
    base daily-return series; `fut_et` and `fut_london` are two DIFFERENT (but
    correlated) return series over the same date range, so the panel-building path
    exercises two genuinely distinct futures lines against one spot line, not clones
    of the same numbers.
    """
    base = [0.0021, -0.0014, 0.0009, -0.0007, 0.0018, -0.0011, 0.0006]
    returns = (base * ((n // len(base)) + 1))[:n]
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n + 1)]
    levels = [seed_level]
    for r in returns:
        levels.append(levels[-1] * (1 + r))
    comp = pl.DataFrame({"date": dates, ccy: levels})
    fut_et = pl.DataFrame({"date": dates[1:], symbol: returns})
    # Same underlying moves with a small jitter -- a genuinely different (but still
    # well-correlated) line, not a byte-identical clone of the ET series.
    fut_london = pl.DataFrame({"date": dates[1:], symbol: [r * 1.02 for r in returns]})
    return comp, fut_et, fut_london


def test_plot_fx_vs_spot_comparison_writes_a_single_pdf(tmp_path, monkeypatch):
    comp, fut_et, fut_london = _synthetic_comp_and_futures()
    monkeypatch.setattr(fx_futures, "SYMBOL_TO_CURCDD_MAPPING", {"6E": "EUR"})
    monkeypatch.setattr(fx_futures, "save_compustat_fx_rates", lambda: comp)
    monkeypatch.setattr(fx_futures, "_blend_series", lambda comp: {})

    def fake_loader(symbol, sync_dir):
        if sync_dir == "sync":
            return fut_et
        if sync_dir == "sync_london":
            return fut_london
        return None

    monkeypatch.setattr(fx_futures, "_load_currency_panel", fake_loader)

    plot_fx_vs_spot_comparison(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["comparison.pdf"]


def test_plot_fx_vs_spot_comparison_handles_a_missing_london_panel(tmp_path, monkeypatch):
    """`sync_london` may not exist for a currency -- the hook must still write ONE pdf,
    using ET only (no crash, no per-asset file)."""
    comp, fut_et, _ = _synthetic_comp_and_futures()
    monkeypatch.setattr(fx_futures, "SYMBOL_TO_CURCDD_MAPPING", {"6E": "EUR"})
    monkeypatch.setattr(fx_futures, "save_compustat_fx_rates", lambda: comp)
    monkeypatch.setattr(fx_futures, "_blend_series", lambda comp: {})

    def fake_loader(symbol, sync_dir):
        return fut_et if sync_dir == "sync" else None

    monkeypatch.setattr(fx_futures, "_load_currency_panel", fake_loader)

    plot_fx_vs_spot_comparison(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["comparison.pdf"]


def test_fx_vs_spot_panels_correlations_match_fx_futures_correlation(monkeypatch):
    """The et/lon correlations annotated on the plot must be EXACTLY the same numbers
    `correlations.csv` reports for the same (symbol, mode) -- both come from
    `fx_futures_correlation`, never a fresh computation."""
    comp, fut_et, fut_london = _synthetic_comp_and_futures()
    monkeypatch.setattr(fx_futures, "SYMBOL_TO_CURCDD_MAPPING", {"6E": "EUR"})

    def fake_loader(symbol, sync_dir):
        if sync_dir == "sync":
            return fut_et
        if sync_dir == "sync_london":
            return fut_london
        return None

    monkeypatch.setattr(fx_futures, "_load_currency_panel", fake_loader)

    panels = fx_futures._fx_vs_spot_panels(comp)
    assert len(panels) == 1
    panel = panels[0]
    assert panel["ccy"] == "EUR"
    assert panel["symbol"] == "6E"

    expected_et = fx_futures_correlation(fut_et, comp, "6E", "EUR", "et")["correlation"]
    expected_lon = fx_futures_correlation(fut_london, comp, "6E", "EUR", "london")["correlation"]
    assert panel["r_et"] == expected_et
    assert panel["r_lon"] == expected_lon
    assert panel["r_et"] is not None and panel["r_et"] > 0.9
    assert panel["r_lon"] is not None and panel["r_lon"] > 0.9

    # Cumulative arrays: sane shape, all three series share one date axis.
    assert (
        len(panel["dates"])
        == len(panel["cum_et"])
        == len(panel["cum_lon"])
        == len(panel["cum_spot"])
    )
    assert panel["cum_et"][-1] != 0.0   # a real cumulative move, not a degenerate flat line


def test_fx_vs_spot_panels_clips_to_shipped_window(monkeypatch):
    """A pre-listing bad print (e.g. CZK 2004-07-13 = +97939x) sits in the PRE-build
    et/london sync panels this function reads, before the currency's shipped window
    even starts. `_fx_vs_spot_panels` must clip the plotted cumulative series to the
    shipped window (`_shipped_windows()`) so that leading artifact never reaches the
    plot -- the first plotted date must equal the shipped-window start, and the
    cumulative path must not contain the blow-out."""
    comp, fut_et, fut_london = _synthetic_comp_and_futures(symbol="CZK", ccy="CZK")
    monkeypatch.setattr(fx_futures, "SYMBOL_TO_CURCDD_MAPPING", {"CZK": "CZK"})

    shipped_start = fut_et["date"][0]
    shipped_end = fut_et["date"][-1]
    artifact_date = shipped_start - timedelta(days=7)   # strictly before the shipped window
    huge_ret = 979.39   # +97939%, CZK 2004-07-13's actual pre-build magnitude

    fut_et_with_artifact = pl.concat(
        [pl.DataFrame({"date": [artifact_date], "CZK": [huge_ret]}), fut_et]
    )
    fut_london_with_artifact = pl.concat(
        [pl.DataFrame({"date": [artifact_date], "CZK": [huge_ret]}), fut_london]
    )
    comp_with_artifact = pl.concat(
        [pl.DataFrame({"date": [artifact_date], "CZK": [comp["CZK"][0]]}), comp]
    )

    def fake_loader(symbol, sync_dir):
        if sync_dir == "sync":
            return fut_et_with_artifact
        if sync_dir == "sync_london":
            return fut_london_with_artifact
        return None

    monkeypatch.setattr(fx_futures, "_load_currency_panel", fake_loader)
    # Mirrors how other tests inject panels via `_load_currency_panel`/
    # `save_compustat_fx_rates`: monkeypatch the sync_daily-derived shipped-window
    # lookup directly, so the test has no filesystem coupling to the real
    # tier2/sync/sync_daily.csv.
    monkeypatch.setattr(
        fx_futures, "_shipped_windows", lambda: {"CZK": (shipped_start, shipped_end)}
    )

    panels = fx_futures._fx_vs_spot_panels(comp_with_artifact)
    assert len(panels) == 1
    panel = panels[0]

    assert panel["dates"][0] == shipped_start
    assert artifact_date not in list(panel["dates"])
    # The blow-out (~979.39 daily return, log1p ~= 6.9 cumulative) never enters the
    # clipped series -- the surviving cumulative path stays near the synthetic panel's
    # ordinary few-percent moves.
    assert max(abs(x) for x in panel["cum_et"]) < 1.0
    assert max(abs(x) for x in panel["cum_lon"]) < 1.0


def test_fx_vs_spot_panels_falls_back_to_no_clip_when_currency_absent_from_windows(monkeypatch):
    """A currency missing from `_shipped_windows()`'s map (e.g. sync_daily.csv doesn't
    have it, which shouldn't happen for mapped currencies but must not crash) keeps its
    full pre-build panel rather than being dropped."""
    comp, fut_et, fut_london = _synthetic_comp_and_futures(symbol="6E", ccy="EUR")
    monkeypatch.setattr(fx_futures, "SYMBOL_TO_CURCDD_MAPPING", {"6E": "EUR"})

    def fake_loader(symbol, sync_dir):
        if sync_dir == "sync":
            return fut_et
        if sync_dir == "sync_london":
            return fut_london
        return None

    monkeypatch.setattr(fx_futures, "_load_currency_panel", fake_loader)
    monkeypatch.setattr(fx_futures, "_shipped_windows", lambda: {})

    panels = fx_futures._fx_vs_spot_panels(comp)
    assert len(panels) == 1
    panel = panels[0]
    assert panel["dates"][0] == fut_et["date"][0]
    assert panel["dates"][-1] == fut_et["date"][-1]


def test_blend_series_tracks_the_future_on_a_toy_panel():
    # A clean toy blend (future finite every day) must be ~1.0 correlated with the future.
    # >= _MIN_OBS (30) rows, or _blend_corr_from_series returns None by design.
    from globalmacro.validation.fx_futures import _blend_corr_from_series
    n = 40
    dates = [date(2020, 1, 6) + timedelta(days=i) for i in range(n)]
    r = [0.0] + [0.01 if i % 2 else -0.008 for i in range(n - 1)]
    df = pl.DataFrame({"date": dates, "r_fut": r, "r_spot": r})  # blend == future
    assert _blend_corr_from_series(df) == pytest.approx(1.0, abs=1e-9)


def test_blend_invariant_passes_on_real_data():
    # DATA-COUPLED integration test: calls save_compustat_fx_rates() + pre_splice_panel("sync")
    # (a full tier-2 build from raw), so it SKIPS (not errors) where the raw dataset is absent,
    # and it is slow. min over G10 of corr(blend, its futures) must be >= 0.99; empirical min
    # ~0.9966 (NOK) -> ~0.007 headroom (thin by design -- min, not median).
    from globalmacro.utils.paths import COMPUSTAT_PATH, DATASETS_ROOT
    if not ((COMPUSTAT_PATH / "exrt_dly.csv").exists()
            and (DATASETS_ROOT / "tier1" / "sync" / "currency_daily_returns.csv").exists()):
        pytest.skip("raw sync CSVs absent")
    from globalmacro.validation.fx_futures import _blend_invariant
    invs = _blend_invariant()
    assert len(invs) == 1
    assert invs[0].passed, invs[0].value
