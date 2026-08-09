import os
import sys
from pathlib import Path

import pytest

from globalmacro.utils import capabilities as cap
from globalmacro.validation import run as vrun


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
