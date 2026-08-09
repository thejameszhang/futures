import polars as pl
import pytest

from globalmacro.utils.capabilities import Capability
from globalmacro.utils.paths import DATASETS_ROOT, FX_PATH
from globalmacro.validation import synthetic_equity, synthetic_fx
from globalmacro.validation.mode import current_mode, validation_mode


def _force_async(monkeypatch):
    absent = Capability(False, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: absent)
    monkeypatch.setattr(synthetic_equity, "sync_panels_ready", lambda: absent)


def _has_fx_synthetic_data() -> bool:
    """True iff the file synthetic_fx._synthetics() reads first is on disk. The
    data-coupled tests below skip (not error) where it is absent -- the established
    pattern, see tests/test_fx_futures_vs_spot.py's real-data test and
    tests/test_sync_fx_blend.py's _has_raw_sync_data."""
    return (FX_PATH / "synthetic_fx_returns_async.csv").exists()


def _has_sync_and_async_panels() -> bool:
    """Real data needed for a full-mode figures() call: fx_source_diagonal.pdf and
    equity_alignment.pdf each read both the sync AND async tier-1/tier-2 daily panels."""
    return (
        (DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv").exists()
        and (DATASETS_ROOT / "tier1" / "async" / "async_daily.csv").exists()
        and (DATASETS_ROOT / "tier2" / "sync" / "sync_daily.csv").exists()
        and (DATASETS_ROOT / "tier2" / "async" / "async_daily.csv").exists()
    )


def test_synthetic_equity_returns_no_invariants_in_async_only(monkeypatch):
    _force_async(monkeypatch)
    assert synthetic_equity._invariants() == []


def test_synthetic_fx_keeps_only_the_async_invariant(monkeypatch):
    """Names contain the dataset, and 'async' contains 'sync' -- assert on the source."""
    _force_async(monkeypatch)
    if not _has_fx_synthetic_data():
        pytest.skip("no on-disk synthetic FX returns to test against")
    invs = synthetic_fx._invariants()
    assert len(invs) == 1
    assert invs[0].name.startswith("Datastream")


def test_skipping_never_manufactures_a_failing_invariant(monkeypatch):
    """The trap: an empty sync subset makes `total > 0 and ...` False, so a skipped
    invariant would report FAIL rather than being absent. Assert on the SHAPE of the
    list -- not on `.passed`, which would require built datasets to evaluate."""
    _force_async(monkeypatch)
    if not _has_fx_synthetic_data():
        pytest.skip("no on-disk synthetic FX returns to test against")
    invs = synthetic_fx._invariants() + synthetic_equity._invariants()
    assert not any(i.value.startswith("0/0") for i in invs)
    # DEVIATION from the original brief: its snippet checked `"sync futures" not in
    # i.name`, which is itself the "async contains sync" trap the whole plan warns
    # about -- "async futures" CONTAINS the substring "sync futures" (a-SYNC futures),
    # so that check would spuriously fail even on a correct implementation. Assert the
    # exact expected name set instead: the only invariant that survives skipping is the
    # async-side one, named for its source, never the sync-side "Compustat ... sync
    # futures" name.
    assert {i.name for i in invs} == {
        "Datastream synthetic beats the other on the async futures",
    }


# --- F1: the resolved MODE, not the sync-panel capability, must gate the sync half --

def test_explicit_async_only_suppresses_sync_even_when_panels_are_ready(monkeypatch):
    """--async-only must win over sync_panels_ready() being True: the capability says
    the panels are on disk, but the resolved mode says don't grade them (capabilities.
    resolve_mode: an explicit flag wins regardless of what's on disk). Data-free --
    _grade_sync() never touches a file."""
    ready = Capability(True, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: ready)
    monkeypatch.setattr(synthetic_equity, "sync_panels_ready", lambda: ready)
    with validation_mode("async-only"):
        assert synthetic_fx._grade_sync() is False
        assert synthetic_equity._grade_sync() is False


def test_full_mode_keeps_sync_when_panels_are_ready(monkeypatch):
    ready = Capability(True, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: ready)
    monkeypatch.setattr(synthetic_equity, "sync_panels_ready", lambda: ready)
    with validation_mode("full"):
        assert synthetic_fx._grade_sync() is True
        assert synthetic_equity._grade_sync() is True


def test_grade_sync_defaults_to_the_capability_when_mode_is_unset(monkeypatch):
    """Nothing has called validation_mode() here -> current_mode() is "full" by
    default, so _grade_sync() reduces to sync_panels_ready().ready alone -- the
    pre-Task-6 guard. This is what keeps _force_async's tests above correct without
    them ever touching validation.mode."""
    assert current_mode() == "full"
    absent = Capability(False, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: absent)
    assert synthetic_fx._grade_sync() is False
    ready = Capability(True, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: ready)
    assert synthetic_fx._grade_sync() is True


# --- F4: the figure branching gets its own guard test, and its own coverage ---------

def test_synthetic_fx_figures_writes_nothing_under_async_only(tmp_path, monkeypatch):
    """Forced async-only, with the guard not bypassable via a ready capability: zero
    files land in out_dir. Data-free -- _grade_sync() short-circuits before any read,
    so this runs unconditionally, unlike the full-mode figures test below."""
    ready = Capability(True, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: ready)
    with validation_mode("async-only"):
        synthetic_fx._figures(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_synthetic_equity_figures_writes_nothing_under_async_only(tmp_path, monkeypatch):
    ready = Capability(True, None)
    monkeypatch.setattr(synthetic_equity, "sync_panels_ready", lambda: ready)
    with validation_mode("async-only"):
        synthetic_equity._figures(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_synthetic_fx_figures_writes_the_pdf_in_full_mode(tmp_path):
    if not _has_sync_and_async_panels():
        pytest.skip("no on-disk sync/async panels to test against")
    with validation_mode("full"):
        synthetic_fx._figures(tmp_path)
    assert (tmp_path / "fx_source_diagonal.pdf").exists()


def test_synthetic_equity_figures_writes_the_pdf_in_full_mode(tmp_path):
    if not _has_sync_and_async_panels():
        pytest.skip("no on-disk sync/async panels to test against")
    with validation_mode("full"):
        synthetic_equity._figures(tmp_path)
    assert (tmp_path / "equity_alignment.pdf").exists()


# --- F2: dropped_invariants/dropped_figures must not drift from what _invariants()/
# _figures() actually produce in full mode. Entirely data-free (source_diagonal/
# alignment/plot_paired_bars stubbed), so -- unlike the real-data tests above -- these
# run on a clean clone / empty data roots too, where a stale static literal would
# otherwise go uncaught (the reviewer's D3 mutation: "6 passed, 4 skipped").

def test_synthetic_fx_full_mode_invariants_order_and_count(monkeypatch):
    """Pins the full-mode invariant NAMES/order/count synthetic_fx._invariants() emits
    when both datasets are graded -- nothing did before (the older open finding F2
    also closes). The names are literals, not re-derived from _SOURCES, so a rename of
    _SOURCES (the reviewer's D1 mutation) is caught here directly, independent of the
    vrun-level dropped_invariants pin."""
    monkeypatch.setattr(synthetic_fx, "_grade_sync", lambda: True)
    stub = pl.DataFrame({
        "dataset": ["sync", "async"],
        "instrument": ["X", "X"],
        "datastream": [0.1, 0.9],
        "compustat": [0.9, 0.1],
        "expected": ["compustat", "datastream"],
        "winner": ["compustat", "datastream"],
    })
    monkeypatch.setattr(synthetic_fx, "source_diagonal", lambda *a, **k: stub)
    invs = synthetic_fx._invariants()
    assert [i.name for i in invs] == [
        "Compustat synthetic beats the other on the sync futures",
        "Datastream synthetic beats the other on the async futures",
    ]
    assert [i.passed for i in invs] == [True, True]
    # And that the sync-side name is the SAME string synthetic_fx_check.dropped_invariants
    # names -- the two are constructed from one shared expression (_invariant_name), not
    # two independently-typed literals.
    assert invs[0].name == synthetic_fx.synthetic_fx_check.dropped_invariants[0]


def test_synthetic_equity_full_mode_invariants_names_and_values(monkeypatch):
    """Equity counterpart: pins the full-mode invariant names/values, data-free
    (alignment() stubbed), and confirms they are the SAME literals
    synthetic_equity_check.dropped_invariants names."""
    monkeypatch.setattr(synthetic_equity, "_grade_sync", lambda: True)
    stub = pl.DataFrame({
        "instrument": ["SXF", "YM", "NQ", "RTY", "IPC"],
        "same_day": [0.1, 0.1, 0.1, 0.1, 0.01],
        "lag_1": [0.9, 0.9, 0.9, 0.9, 0.02],
        "shipped": ["lag_1"] * 5,
        "expected": ["lag_1", "lag_1", "lag_1", "lag_1", "indeterminate"],
        "americas": [True] * 5,
        "n_backfilled": [10] * 5,
    })
    monkeypatch.setattr(synthetic_equity, "alignment", lambda: stub)
    invs = synthetic_equity._invariants()
    assert [i.name for i in invs] == [
        "shipped alignment is the one the data prefers, per symbol",
        "every Americas symbol is actually under test",
    ]
    assert [i.passed for i in invs] == [True, True]
    assert [i.name for i in invs] == list(synthetic_equity.synthetic_equity_check.dropped_invariants)


def test_synthetic_fx_figure_path_matches_dropped_figures_data_free(monkeypatch, tmp_path):
    """F2 proof for D3 (a renamed produced-figure filename): ties the ACTUAL path
    _figures() hands to plot_paired_bars against Check.dropped_figures, entirely
    data-free -- so a drift is caught even under empty data roots, where
    test_synthetic_fx_figures_writes_the_pdf_in_full_mode (real-data) skips."""
    monkeypatch.setattr(synthetic_fx, "_grade_sync", lambda: True)
    monkeypatch.setattr(synthetic_fx, "source_diagonal", lambda *a, **k: pl.DataFrame())
    captured: dict = {}
    monkeypatch.setattr(
        "globalmacro.validation.plots.plot_paired_bars",
        lambda *a, **k: captured.update(path=k["path"]),
    )
    synthetic_fx._figures(tmp_path)
    assert captured["path"] == tmp_path / synthetic_fx.synthetic_fx_check.dropped_figures[0]


def test_synthetic_equity_figure_path_matches_dropped_figures_data_free(monkeypatch, tmp_path):
    """Equity counterpart of the D3 proof above, data-free (alignment() stubbed)."""
    monkeypatch.setattr(synthetic_equity, "_grade_sync", lambda: True)
    stub = pl.DataFrame(
        {"instrument": ["X"], "same_day": [0.1], "lag_1": [0.2], "americas": [True]}
    )
    monkeypatch.setattr(synthetic_equity, "alignment", lambda: stub)
    captured: dict = {}
    monkeypatch.setattr(
        "globalmacro.validation.plots.plot_paired_bars",
        lambda *a, **k: captured.update(path=k["path"]),
    )
    synthetic_equity._figures(tmp_path)
    assert captured["path"] == tmp_path / synthetic_equity.synthetic_equity_check.dropped_figures[0]
