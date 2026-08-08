from datetime import date

import polars as pl
import pytest

from globalmacro import build


@pytest.fixture
def datasets_root(tmp_path, monkeypatch):
    for tier in ("tier1", "tier2"):
        for fam in ("sync", "async"):
            (tmp_path / tier / fam).mkdir(parents=True)
    monkeypatch.setattr(build, "DATASETS_ROOT", tmp_path)
    # NOTE: save_usd_datasets's `out_root` parameter defaults to DATASETS_ROOT at
    # *def* time (build.py), so this monkeypatch does NOT redirect it -- only
    # save_datasets (which reads the module attribute at call time) is covered.
    # Every save_usd_datasets call below MUST pass `out_root=datasets_root`
    # explicitly, or it silently writes into the real repo datasets/ tree.
    return tmp_path


def _frame(cols):
    # datetime.date, NOT pl.date() -- the latter is an expression and yields an
    # Object column that write_csv rejects. See tests/test_usd_build.py:44.
    return pl.DataFrame({"date": [date(2020, 1, 1)], **{c: [0.01] for c in cols}})


def test_save_datasets_writes_no_sync_file_when_synced_is_none(datasets_root):
    # Path, not filename: "async_daily.csv" is written under BOTH tier1/async/ and
    # tier2/async/, so a bare-filename set would collapse the two and could not
    # detect a tier-2 write silently dropped (e.g. moved inside the wrong guard).
    a = _frame(["ES", "CL"])
    build.save_datasets(None, a, a, ["ES"], ["CL"])
    written = {p.relative_to(datasets_root).as_posix() for p in datasets_root.rglob("*.csv")}
    expected_async = {
        "tier1/async/async_daily.csv",
        "tier1/async/async_monthly.csv",
        "tier2/async/async_daily.csv",
        "tier2/async/async_monthly.csv",
    }
    assert expected_async <= written, sorted(expected_async - written)
    # Path-component check, not substring: "async" contains "sync" as a substring,
    # so `"sync" in p` would misfire on every async path.
    sync_paths = [p for p in written if p.split("/")[1] == "sync"]
    assert sync_paths == []


def test_save_datasets_returns_none_for_the_sync_frames(datasets_root):
    a = _frame(["ES", "CL"])
    t1_sync, t1_async, t1_m, t2_sync, t2_async, t2_m = build.save_datasets(
        None, a, a, ["ES"], ["CL"])
    assert t1_sync is None and t2_sync is None
    assert t1_async is not None and t2_async is not None


def test_save_usd_datasets_writes_no_sync_usd_when_fx_sync_is_none(datasets_root):
    a = _frame(["ES"])
    # out_root= is REQUIRED here -- see the note on the `datasets_root` fixture
    # above; the monkeypatch does not reach this function's default.
    build.save_usd_datasets(None, a, None, a, {"ES": "USD"},
                            _frame(["USD"]), None, out_root=datasets_root)
    written = {p.name for p in datasets_root.rglob("*_usd.csv")}
    assert not any(n.startswith("sync") for n in written)


def test_save_usd_datasets_writes_no_sync_usd_when_synced_frames_are_none(datasets_root):
    """Pins the `tier1_synced is not None` / `tier2_synced is not None` operands of
    the guards at build.py independently of `fx_sync is not None`: fx_sync here is
    a REAL frame, so a guard mutated down to bare `if fx_sync is not None:` (the
    anti-pattern the brief forbids) would try to convert a `None` sync panel and
    must be caught -- either by writing a sync _usd file (wrong) or by crashing
    inside save_usd_datasets, both of which fail this test."""
    a = _frame(["ES"])
    # out_root= is REQUIRED here -- see the note on the `datasets_root` fixture
    # above; the monkeypatch does not reach this function's default.
    build.save_usd_datasets(None, a, None, a, {"ES": "USD"},
                            _frame(["USD"]), _frame(["USD"]), out_root=datasets_root)
    written = {p.name for p in datasets_root.rglob("*_usd.csv")}
    assert not any(n.startswith("sync") for n in written)
