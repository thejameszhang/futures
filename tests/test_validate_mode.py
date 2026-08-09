import os
import sys
import types
from pathlib import Path

import polars as pl
import pytest

from globalmacro.utils import capabilities as cap
from globalmacro.validation import run as vrun
from globalmacro.validation.base import Check, Invariant
from globalmacro.validation.mode import current_mode, validation_mode


def test_async_only_filters_exactly_the_full_sync_checks():
    full = {c.slug for c in vrun._available_checks("full")}
    async_only = {c.slug for c in vrun._available_checks("async-only")}
    dropped = full - async_only
    assert dropped <= {"consistency", "fx_futures_vs_spot", "external"}
    assert {"consistency", "fx_futures_vs_spot"} <= dropped
    assert {"datastream", "synthetic_fx", "synthetic_equity", "spot_fx"} <= async_only


def test_symbol_count_sources_drop_sync_but_keep_async():
    """Guards the 'async contains sync' trap: a stem-substring filter drops everything."""
    stems = {s for s, _, _ in vrun._symbol_count_sources("async-only")}
    assert "tier1_sync_daily" not in stems
    assert "tier1_async_daily" in stems
    assert "tier2_async_monthly" in stems


def test_symbol_count_sources_survives_sync_in_root_path(monkeypatch):
    """Guards a second 'async contains sync' trap one level up: DATASETS_ROOT is
    env-configurable (FUTURES_DATASETS_ROOT) and can itself contain a "sync" path
    component (e.g. a relocated data directory under .../sync/datasets). A path-
    substring filter ("/sync/" in the full path) would then match every source's path
    and silently drop all of them -- exactly the failure the reviewer demonstrated."""
    root = Path("/data/sync/datasets")
    sources = [
        ("tier1_sync_daily", root / "tier1" / "sync" / "sync_daily.csv", "x"),
        ("tier1_async_daily", root / "tier1" / "async" / "async_daily.csv", "x"),
        ("tier2_async_monthly", root / "tier2" / "async" / "async_monthly.csv", "x"),
    ]
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", sources)
    kept = vrun._symbol_count_sources("async-only")
    assert {s for s, _, _ in kept} == {"tier1_async_daily", "tier2_async_monthly"}


def test_validate_parses_modes():
    assert vrun._parse_args(["--async-only"]).mode == "async-only"
    assert vrun._parse_args(["--full"]).mode == "full"
    assert vrun._parse_args([]).mode is None


def test_available_checks_full_mode_matches_pre_task5_shape():
    """Pins that mode="full" (the default) returns the identical checks, in the
    identical order, that the pre-Task-5 zero-argument _available_checks() returned --
    the filter must be inert in full mode.

    "external" is deliberately NOT asserted as present: external_comparison.py is
    gitignored (confidential, local-only) and does not exist on a clean clone, so
    _available_checks("full") legitimately omits it there. Assert the six checks every
    clone has exactly, and let the optional seventh be present-or-absent."""
    checks = vrun._available_checks("full")
    slugs = [c.slug for c in checks]
    assert slugs[:6] == [
        "datastream", "consistency", "synthetic_fx",
        "synthetic_equity", "spot_fx", "fx_futures_vs_spot",
    ]
    assert set(slugs[6:]) <= {"external"}
    assert vrun._available_checks() == checks       # default argument matches explicit "full"


def test_skipped_checks_empty_outside_async_only():
    assert vrun._skipped_checks("full") == []


def test_skipped_checks_names_the_dropped_full_mode_checks():
    """"external" is only asserted when present -- see the docstring on
    test_available_checks_full_mode_matches_pre_task5_shape for why it may legitimately
    be absent on a clean clone. The expected count is derived from requires_sync rather
    than hard-coded, so it tracks whichever checks are actually importable."""
    skipped = vrun._skipped_checks("async-only")
    full_by_slug = {c.slug: c.name for c in vrun._available_checks("full")}
    assert full_by_slug["consistency"] in skipped
    assert full_by_slug["fx_futures_vs_spot"] in skipped
    if "external" in full_by_slug:
        assert full_by_slug["external"] in skipped
    expected = [c for c in vrun._available_checks("full") if c.requires_sync]
    assert len(skipped) == len(expected)


def test_available_checks_degrades_gracefully_without_external_module(monkeypatch):
    """F1 proof: simulates a clean clone, where external_comparison.py (gitignored,
    confidential, local-only) does not exist on disk at all. Evicts any cached import
    and installs a meta_path finder that makes the module unimportable -- the same
    mechanism a genuinely absent file produces -- rather than relying on it merely
    being present-but-unused on this machine."""
    name = "globalmacro.validation.external_comparison"
    monkeypatch.delitem(sys.modules, name, raising=False)

    class _BlockExternalComparison:
        def find_spec(self, fullname, path, target=None):
            if fullname == name:
                raise ModuleNotFoundError(f"simulated absence of {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockExternalComparison(), *sys.meta_path])

    checks = vrun._available_checks("full")
    slugs = [c.slug for c in checks]
    assert slugs == [
        "datastream", "consistency", "synthetic_fx",
        "synthetic_equity", "spot_fx", "fx_futures_vs_spot",
    ]
    assert "external" not in slugs

    skipped = vrun._skipped_checks("async-only")
    full_by_slug = {c.slug: c.name for c in vrun._available_checks("full")}
    assert "external" not in full_by_slug
    assert full_by_slug["consistency"] in skipped
    assert full_by_slug["fx_futures_vs_spot"] in skipped
    expected = [c for c in vrun._available_checks("full") if c.requires_sync]
    assert len(skipped) == len(expected)


# --- F4: requires_sync=True for the external check must live in TRACKED code -------


def test_external_check_requires_sync_is_forced_in_tracked_code(monkeypatch):
    """F4 (Reviewer A, proved). external_comparison.py is gitignored and untracked
    (.gitignore:34) -- a local requires_sync=True there is invisible to `git status`
    and does not travel with the merge; anywhere with a different copy would run
    this check in async-only mode and try to read tier1/sync/sync_daily.csv, which
    doesn't exist there. _available_checks must force requires_sync=True in tracked
    code instead, via dataclasses.replace.

    Injects a stub module (same sys.modules-substitution mechanism as
    test_connect_report.py's _fake_wrds_success) whose external_check leaves
    requires_sync at its dataclass default (False) -- exactly the state a
    differently-configured clone/worktree would have -- so this proves the forcing
    rather than depending on this machine's own (possibly already-correct) untracked
    copy. Does not require the real gitignored module to exist, so it also passes on
    a clean clone where the file is absent."""
    name = "globalmacro.validation.external_comparison"
    stub_check = Check(
        name="External ground-truth cross-check", slug="external", run=_stub_correlations,
    )
    assert stub_check.requires_sync is False          # the dataclass default
    stub_module = types.ModuleType(name)
    stub_module.external_check = stub_check
    monkeypatch.setitem(sys.modules, name, stub_module)

    full = vrun._available_checks("full")
    async_only = vrun._available_checks("async-only")

    injected_full = next(c for c in full if c.slug == "external")
    assert injected_full.requires_sync is True
    assert "external" not in {c.slug for c in async_only}
    # dataclasses.replace() returns a new instance -- the injected stub itself, and
    # therefore any other reference to it, must be untouched.
    assert stub_check.requires_sync is False


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("date\n")
    os.utime(path, (mtime, mtime))


def test_sync_panels_fresh_when_written_alongside_async(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    for tier in ("tier1", "tier2"):
        _touch(tmp_path / tier / "sync" / "sync_daily.csv", 1_000_000)
        _touch(tmp_path / tier / "async" / "async_daily.csv", 1_000_000)
    c = cap.sync_panels_fresh()
    assert c.ready is True
    assert c.message is None


def test_sync_panels_stale_after_async_only_rebuild(tmp_path, monkeypatch):
    """The exact scenario the reviewer flagged: a full build wrote both halves
    together, then a later async-only build rewrote only the async panel."""
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _touch(tmp_path / "tier1" / "sync" / "sync_daily.csv", 1_000_000)
    _touch(tmp_path / "tier1" / "async" / "async_daily.csv", 1_000_000)
    _touch(tmp_path / "tier2" / "sync" / "sync_daily.csv", 1_000_000)
    _touch(tmp_path / "tier2" / "async" / "async_daily.csv", 1_000_000)
    # async-only rebuild: only the async panels move forward in time
    _touch(tmp_path / "tier1" / "async" / "async_daily.csv", 2_000_000)
    _touch(tmp_path / "tier2" / "async" / "async_daily.csv", 2_000_000)

    c = cap.sync_panels_fresh()
    assert c.ready is False
    assert "tier1" in c.message and "sync_daily.csv" in c.message
    assert "tier2" in c.message


# --- F5a: the resolved mode is printed as the first line of validate's stdout -----


def test_f5a_mode_is_the_first_line_of_stdout_full(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    vrun.main(["--full"])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "mode: full"


def test_f5a_mode_is_the_first_line_of_stdout_async_only(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(False, None))

    vrun.main(["--async-only"])
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "mode: async-only"


# --- F3: validate can resolve full and then die with a raw traceback --------------


def _make_sync_stage_outputs(root, pairs):
    for tier, cls in pairs:
        d = root / tier / "sync"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cls}_daily_returns.csv").write_text("date\n")


def _make_sync_panels(root, present):
    all_paths = {
        "tier1_daily": root / "tier1" / "sync" / "sync_daily.csv",
        "tier2_daily": root / "tier2" / "sync" / "sync_daily.csv",
        "tier2_currency": root / "tier2" / "sync" / "currency_daily_returns.csv",
    }
    for key in present:
        p = all_paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("date\n")


def test_f3a_disagreement_state_now_resolves_async_only(tmp_path, monkeypatch):
    """Reviewer B's exact scratch-root scenario: a researcher reclaims disk after a
    full run -- the 3 aggregate sync panels remain, but the raw tickhistory stage
    outputs are gone. Before F3a, sync_panels_ready() (validate's own predicate)
    only checked those 3 files and reported ready=True, so an unflagged `globalmacro
    validate` resolved "full" while `globalmacro build` (which checks
    sync_stage_outputs_ready() directly) resolved "async-only" -- a silent
    disagreement. F3a makes sync_panels_ready() consult sync_stage_outputs_ready()
    first, so validate's own auto-resolution now agrees with build's: async-only."""
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_panels(tmp_path, ["tier1_daily", "tier2_daily", "tier2_currency"])
    assert cap.sync_stage_outputs_ready().ready is False
    resolved = cap.resolve_mode(None, cap.sync_panels_ready(), "validate")
    assert resolved == "async-only"


def test_f3b_residual_filenotfounderror_from_invariants_is_actionable(monkeypatch, tmp_path):
    """F3b covers the gap F3a deliberately leaves open: even with all 10 tickhistory
    stage outputs AND all 3 sync panels present (so resolve_mode cleanly resolves
    "full", never reaching its own not-ready SystemExit), build_synced_dataset()
    still reads one file neither predicate covers -- the hand-placed MANUAL JKP
    prerequisite (sync_stage_outputs_ready()'s own docstring: 'a missing manual
    prerequisite should crash loudly in full mode, not silently downgrade'). Before
    F3b, that FileNotFoundError propagated as a raw traceback with no remedy and no
    mention of --async-only; check.invariants() sits outside any try, unlike
    pairs/figures. Hermetic: a stub Check reproduces the raise directly rather than
    exercising the real synthetic_fx chain, following this file's own established
    stub-Check pattern (see test_validation_mode_does_not_leak_between_two_runs_in_one_process)."""
    def _boom():
        raise FileNotFoundError("data/jkp/updated_daily_ind_gics_synced.csv")

    stub_check = Check(name="stub", slug="stub", run=_stub_correlations, invariants=_boom)
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [stub_check])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    with pytest.raises(SystemExit) as exc_info:
        vrun.main(["--full"])
    msg = str(exc_info.value)
    assert "updated_daily_ind_gics_synced.csv" in msg
    assert "--full" in msg
    assert "--async-only" in msg


def test_f3b_genuine_invariant_failure_still_fails_the_run(monkeypatch, tmp_path):
    """F3b must not blanket-catch: a genuine invariant FAILURE (Invariant.passed is
    False, not an exception) is untouched by the new try/except and still fails the
    run through the normal ok=False path."""
    def _fails():
        return [Invariant(check="stub", name="x beats y", value="no", passed=False)]

    stub_check = Check(name="stub", slug="stub", run=_stub_correlations, invariants=_fails)
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [stub_check])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    assert vrun.main(["--full"]) == 1


def test_f3b_non_filenotfound_exceptions_propagate_unconverted(monkeypatch, tmp_path):
    """F3b's except clause is FileNotFoundError ONLY -- any other exception is a real
    bug and must still surface as itself, not be swallowed or relabeled."""
    def _boom():
        raise ValueError("genuine bug, not a missing file")

    stub_check = Check(name="stub", slug="stub", run=_stub_correlations, invariants=_boom)
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [stub_check])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    with pytest.raises(ValueError, match="genuine bug"):
        vrun.main(["--full"])


def test_sync_panels_fresh_ignores_missing_files(tmp_path, monkeypatch):
    """Nothing to compare (e.g. async side never built) is not this predicate's
    concern -- sync_panels_ready()/the async build itself own that failure mode."""
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    c = cap.sync_panels_fresh()
    assert c.ready is True
    assert c.message is None


def test_full_mode_refuses_when_sync_panels_are_stale(monkeypatch, tmp_path):
    """Wires sync_panels_fresh() into main(): a stale sync panel must abort a
    full-mode validate run before it touches any real check or dataset.

    VALIDATION_OUTPUT is redirected to tmp_path and _available_checks is stubbed to
    return no checks -- not because either is expected to matter, but so that if the
    gate itself were ever deleted, this test would fail on "DID NOT RAISE" instead of
    silently running datastream_check/consistency_check/synthetic_fx_check (the last
    of which rebuilds the tier-2 sync panel from raw) against the real DATASETS_ROOT
    and overwriting the real VALIDATION_SUMMARY.md. No unit test may write to the real
    DATASETS_ROOT or VALIDATION_OUTPUT; this asserts the gate fires rather than
    depending on it."""
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(
        vrun, "sync_panels_fresh",
        lambda: cap.Capability(False, "sync panels predate their async counterparts"),
    )
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [])
    with pytest.raises(SystemExit, match="sync panels predate their async counterparts"):
        vrun.main(["--full"])


# --- F2: dropped invariants/figures within a still-running check are named, not silent

def test_dropped_invariants_empty_in_full_mode():
    assert vrun._dropped_invariants("full") == []


def test_dropped_figures_empty_in_full_mode():
    assert vrun._dropped_figures("full") == []


def test_dropped_invariants_names_the_three_synthetic_check_invariants():
    """synthetic_fx drops one (the Compustat/sync-side invariant); synthetic_equity
    drops both of its own (both derive from alignment(), which is sync-only) -- three
    total, matching what the reviewer counted."""
    dropped = vrun._dropped_invariants("async-only")
    assert dropped == [
        "Compustat synthetic beats the other on the sync futures",
        "shipped alignment is the one the data prefers, per symbol",
        "every Americas symbol is actually under test",
    ]


def test_dropped_figures_names_both_synthetic_check_figures():
    dropped = vrun._dropped_figures("async-only")
    assert dropped == ["fx_source_diagonal.pdf", "equity_alignment.pdf"]


# --- F9: async-only must not leave stale PDFs the summary calls SKIPPED -----------


def _f9_stub_run():
    return pl.DataFrame(
        {"instrument": [], "correlation": [], "n_obs": []},
        schema={"instrument": pl.Utf8, "correlation": pl.Float64, "n_obs": pl.Int64},
    )


def _f9_stub_pairs():
    return pl.DataFrame(
        {"instrument": [], "name": [], "month": [], "ours": [], "theirs": []},
        schema={"instrument": pl.Utf8, "name": pl.Utf8, "month": pl.Utf8,
                "ours": pl.Float64, "theirs": pl.Float64},
    )


# A check dropped WHOLE in async-only (requires_sync=True) with a pairs-rendered
# comparison.pdf -- e.g. consistency_check.
_F9_SKIPPED_CHECK = Check(
    name="Skipped stub", slug="skipped_stub", run=_f9_stub_run,
    pairs=_f9_stub_pairs, requires_sync=True,
)
# A check that still RUNS in async-only but drops one of its own figures --
# e.g. synthetic_fx_check.
_F9_RUNNING_CHECK = Check(
    name="Running stub", slug="running_stub", run=_f9_stub_run,
    dropped_figures=("dropped_stub.pdf",),
)
# A check untouched by any of this -- no pairs, no figures, always kept.
_F9_KEPT_CHECK = Check(name="Kept stub", slug="kept_stub", run=_f9_stub_run)


def _f9_full_checks():
    return [_F9_SKIPPED_CHECK, _F9_RUNNING_CHECK, _F9_KEPT_CHECK]


def _f9_available_checks(mode):
    if mode == "async-only":
        return [c for c in _f9_full_checks() if not c.requires_sync]
    return _f9_full_checks()


# F16: a private, test-owned _SYMBOL_COUNT_SOURCES shape (NOT the production one),
# so these tests stay independent of what globalmacro.validation.run's real source
# list contains -- deterministic, and unaffected if the production list ever grows.
_F9_SYMBOL_COUNT_SOURCES = [
    ("tier1_sync_daily", Path("/fake-data/tier1/sync/sync_daily.csv"), "x"),
    ("tier1_async_daily", Path("/fake-data/tier1/async/async_daily.csv"), "x"),
]


def test_dropped_symbol_count_stems_matches_symbol_count_sources_filter(monkeypatch):
    """F16: derived from _symbol_count_sources's own filter, not a fresh literal."""
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", _F9_SYMBOL_COUNT_SOURCES)
    assert vrun._dropped_symbol_count_stems("async-only") == ["tier1_sync_daily"]
    assert vrun._dropped_symbol_count_stems("full") == []


def test_stale_figure_paths_names_skipped_comparison_pdf_and_dropped_figures(monkeypatch):
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", _F9_SYMBOL_COUNT_SOURCES)
    paths = vrun._stale_figure_paths("async-only")
    assert vrun.VALIDATION_OUTPUT / "skipped_stub" / "comparison.pdf" in paths
    # F16: correlations.csv is the machine-readable half of the SAME dropped-check
    # artifact -- written unconditionally by main()'s loop, so it must be cleaned up
    # alongside comparison.pdf, not just the plot.
    assert vrun.VALIDATION_OUTPUT / "skipped_stub" / "correlations.csv" in paths
    assert vrun.VALIDATION_OUTPUT / "running_stub" / "dropped_stub.pdf" in paths
    # F16: the sync symbol-count PDF -- not a Check at all, so it was invisible to
    # F9's original two sources entirely.
    assert vrun.VALIDATION_OUTPUT / "symbol_counts" / "tier1_sync_daily.pdf" in paths
    assert len(paths) == 4


def test_stale_figure_paths_empty_in_full_mode(monkeypatch):
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", _F9_SYMBOL_COUNT_SOURCES)
    assert vrun._stale_figure_paths("full") == []


def test_remove_stale_figures_deletes_exactly_the_dropped_ones(monkeypatch, tmp_path):
    """Plant dummy PDFs/CSVs, run async-only cleanup, assert exactly the dropped ones
    are gone and every other file (including siblings in the SAME check/symbol_counts
    directories) survives."""
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", _F9_SYMBOL_COUNT_SOURCES)

    stale1 = tmp_path / "skipped_stub" / "comparison.pdf"
    stale2 = tmp_path / "running_stub" / "dropped_stub.pdf"
    stale3 = tmp_path / "skipped_stub" / "correlations.csv"          # F16
    stale4 = tmp_path / "symbol_counts" / "tier1_sync_daily.pdf"     # F16
    # Survivors: a plain file in a fully-kept check's dir, the RUNNING check's own
    # (non-dropped) comparison.pdf, a differently-named sibling in the skipped
    # check's dir (proves this is not a whole-directory wipe), and the symbol-count
    # PDF _symbol_count_sources keeps in async-only mode.
    survivor_kept = tmp_path / "kept_stub" / "comparison.pdf"
    survivor_running_comparison = tmp_path / "running_stub" / "comparison.pdf"
    survivor_notes = tmp_path / "skipped_stub" / "notes.txt"
    survivor_symbol_count = tmp_path / "symbol_counts" / "tier1_async_daily.pdf"
    for p in (stale1, stale2, stale3, stale4, survivor_kept,
              survivor_running_comparison, survivor_notes, survivor_symbol_count):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy")

    removed = vrun._remove_stale_figures("async-only")

    assert set(removed) == {stale1, stale2, stale3, stale4}
    assert not stale1.exists()
    assert not stale2.exists()
    assert not stale3.exists()
    assert not stale4.exists()
    assert survivor_kept.exists()
    assert survivor_running_comparison.exists()
    assert survivor_notes.exists()
    assert survivor_symbol_count.exists()


def test_remove_stale_figures_deletes_nothing_in_full_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    stale = tmp_path / "skipped_stub" / "comparison.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"dummy")

    removed = vrun._remove_stale_figures("full")
    assert removed == []
    assert stale.exists()


def test_remove_stale_figures_missing_ok_on_clean_tree(monkeypatch, tmp_path):
    """A genuinely clean async-only run (no prior full run, nothing on disk) must
    not raise -- unlink(missing_ok=True)."""
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    assert vrun._remove_stale_figures("async-only") == []


def test_remove_stale_figures_guard_rejects_path_traversal(monkeypatch, tmp_path):
    """N5 (Reviewer B). _remove_stale_figures's path-traversal guard (p.parent.parent
    != VALIDATION_OUTPUT, and no separator/".."/"" in p.name) had NO test -- a future
    simplification could delete it with the suite green, on deletion code. Craft the
    two shapes the guard exists for and prove both are refused:

    1. A crafted SLUG ("../..") on a whole-dropped check: out_dir = VALIDATION_OUTPUT
       / "../..", so its comparison.pdf lands two directories above VALIDATION_OUTPUT
       on the real filesystem (the OS resolves ".." at each path component even
       though pathlib's own .parent does not -- which is exactly why the guard
       compares the raw, unresolved p.parent.parent rather than a resolved path).
       VALIDATION_OUTPUT is nested two levels deep here so that escape target still
       lands inside tmp_path, not a real shared directory.
    2. A crafted dropped_figures ENTRY ("<tmp_path>/escape/x.pdf", an absolute path)
       on a still-running check: pathlib's / operator discards everything to the left
       of an absolute right operand, so out_dir / abs_path == abs_path verbatim,
       escaping VALIDATION_OUTPUT entirely if unguarded.

    Both planted files must survive, and _remove_stale_figures must remove nothing.
    """
    validation_output = tmp_path / "a" / "b" / "validation"
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", validation_output)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])

    absolute_figure_target = tmp_path / "escape" / "x.pdf"
    traversal_check = Check(
        name="Traversal stub", slug="../..", run=_f9_stub_run,
        pairs=_f9_stub_pairs, requires_sync=True,
    )
    leaky_check = Check(
        name="Leaky stub", slug="leaky_stub", run=_f9_stub_run,
        dropped_figures=(str(absolute_figure_target),),
    )

    def _checks(mode):
        if mode == "async-only":
            return [leaky_check]
        return [traversal_check, leaky_check]

    monkeypatch.setattr(vrun, "_available_checks", _checks)

    # Sanity-check the fixture is shaped as intended before trusting the guard result:
    # _stale_figure_paths must actually PRODUCE the malicious candidates, or this test
    # would pass for the wrong reason (nothing to guard against). The traversal slug
    # produces TWO candidates (comparison.pdf AND correlations.csv, F16), both sharing
    # the same malicious out_dir.
    candidates = vrun._stale_figure_paths("async-only")
    assert len(candidates) == 3
    slug_candidate = next(p for p in candidates if p.name == "comparison.pdf")
    assert slug_candidate.parent.parent != validation_output
    assert absolute_figure_target in candidates

    # The real filesystem location the slug candidate resolves to once the OS
    # processes "..": two directories above VALIDATION_OUTPUT == tmp_path/"a". The
    # OS only resolves ".." while descending through REAL directories -- unlike
    # unguarded removal, it will not walk through a "validation" that doesn't exist
    # on disk -- so VALIDATION_OUTPUT itself must actually exist here, or this half
    # of the test would "pass" even with the guard fully deleted (ENOENT on the
    # nonexistent intermediate, not the guard, would be doing the protecting).
    validation_output.mkdir(parents=True, exist_ok=True)
    escape_via_slug = tmp_path / "a" / "comparison.pdf"
    escape_via_slug.parent.mkdir(parents=True, exist_ok=True)
    escape_via_slug.write_bytes(b"dummy")
    absolute_figure_target.parent.mkdir(parents=True, exist_ok=True)
    absolute_figure_target.write_bytes(b"dummy")

    removed = vrun._remove_stale_figures("async-only")

    assert removed == []
    assert escape_via_slug.exists()
    assert absolute_figure_target.exists()


def test_main_wires_stale_figure_removal_for_async_only(monkeypatch, tmp_path):
    """Proves the cleanup is actually reachable from main(), not just a standalone
    function -- end to end via the repo's established hermetic stubbing pattern.
    EXPLICIT --async-only (see the auto-detected counterpart below, F13)."""
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(False, None))

    stale = tmp_path / "skipped_stub" / "comparison.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"dummy")

    vrun.main(["--async-only"])
    assert not stale.exists()


def test_main_does_not_remove_stale_figures_when_async_only_is_auto_detected(monkeypatch, tmp_path):
    """F13 (Reviewer A, PROVED). F9 originally keyed on the RESOLVED mode alone, so it
    fired on an auto-detected downgrade the researcher never asked for -- deleting five
    figures that exist in the owner's pristine baseline. Take the exact state A proved:
    no explicit flag, sync inputs not ready (auto-resolves async-only) -- deletion must
    require EXPLICIT --async-only; an auto-detected downgrade must delete NOTHING and
    disclose the risk in the summary instead."""
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(False, None))

    stale = tmp_path / "skipped_stub" / "comparison.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"dummy")

    vrun.main([])   # no flag -> auto-detected async-only, NOT explicit
    assert stale.exists(), "auto-detected async-only must not delete anything"
    summary = (tmp_path / "VALIDATION_SUMMARY.md").read_text()
    assert "may still be sitting on disk" in summary
    assert "--async-only explicitly" in summary


def test_main_does_not_remove_anything_in_full_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", _f9_available_checks)
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    stale = tmp_path / "skipped_stub" / "comparison.pdf"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"dummy")

    vrun.main(["--full"])
    assert stale.exists()


# --- F1: validation_mode() must reject anything but the two canonical literals ------

@pytest.mark.parametrize("bad_mode", ["Full", "FULL", "aysnc-only", "", None])
def test_validation_mode_rejects_invalid_modes(bad_mode):
    """The same defect class build._validate_mode was written to close: a typo (or a
    caller reaching this shim some way other than run.py's --async-only/--full CLI
    path, which is the only constrained caller today) must raise loudly rather than
    silently take the async-only branch -- `current_mode() == "full"` is False for
    anything but the exact string "full", so an unvalidated mode degrades a run to
    async-only with no error and no warning."""
    assert current_mode() == "full"          # nothing has run yet
    with pytest.raises(ValueError, match="unrecognized mode"), validation_mode(bad_mode):
        pass
    assert current_mode() == "full"          # the rejected value never took effect


def test_validation_mode_accepts_the_two_canonical_literals():
    with validation_mode("async-only"):
        assert current_mode() == "async-only"
    assert current_mode() == "full"
    with validation_mode("full"):
        assert current_mode() == "full"
    assert current_mode() == "full"


# --- N4: the docstring promises reset "including on an exception" -- prove it -------

def test_validation_mode_resets_even_when_the_block_raises():
    assert current_mode() == "full"
    with pytest.raises(RuntimeError, match="boom"), validation_mode("async-only"):
        assert current_mode() == "async-only"
        raise RuntimeError("boom")
    assert current_mode() == "full"          # reset even though the block raised


# --- F1: the mode shim must not leak between two main() calls in one process --------

def _stub_correlations():
    return pl.DataFrame(
        {"instrument": [], "correlation": [], "n_obs": []},
        schema={"instrument": pl.Utf8, "correlation": pl.Float64, "n_obs": pl.Int64},
    )


def test_validation_mode_does_not_leak_between_two_runs_in_one_process(monkeypatch, tmp_path):
    """F1's mode shim (validation.mode) is scoped to a single main() call via a context
    manager. Prove it: run async-only then full in the same process, with a stub check
    whose invariants() callable records validation.mode.current_mode() -- and confirm
    neither run sees the other's mode, and module state is back to the default ("full")
    both between and after the two calls.

    Entirely data-free: VALIDATION_OUTPUT is redirected to tmp_path, the real checks and
    symbol-count sources are replaced with a stub/empty list, and the sync-panel
    capability checks are stubbed -- no real DATASETS_ROOT or VALIDATION_OUTPUT is ever
    touched, and main() is never called with real paths (binding constraint)."""
    seen: list[str] = []

    def _record_invariants():
        seen.append(current_mode())
        return []

    stub_check = Check(
        name="stub", slug="stub", run=_stub_correlations, invariants=_record_invariants,
    )

    monkeypatch.setattr(vrun, "VALIDATION_OUTPUT", tmp_path)
    monkeypatch.setattr(vrun, "_available_checks", lambda mode: [stub_check])
    monkeypatch.setattr(vrun, "_SYMBOL_COUNT_SOURCES", [])
    monkeypatch.setattr(vrun, "sync_panels_ready", lambda: cap.Capability(True, None))
    monkeypatch.setattr(vrun, "sync_panels_fresh", lambda: cap.Capability(True, None))

    assert current_mode() == "full"          # nothing has run yet

    vrun.main(["--async-only"])
    assert seen == ["async-only"]
    assert current_mode() == "full"          # reset after the async-only run

    vrun.main(["--full"])
    assert seen == ["async-only", "full"]
    assert current_mode() == "full"          # reset after the full run too
