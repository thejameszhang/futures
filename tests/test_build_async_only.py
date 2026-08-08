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
    return tmp_path


def _frame(cols):
    # datetime.date, NOT pl.date() -- the latter is an expression and yields an
    # Object column that write_csv rejects. See tests/test_usd_build.py:44.
    return pl.DataFrame({"date": [date(2020, 1, 1)], **{c: [0.01] for c in cols}})


def test_save_datasets_writes_no_sync_file_when_synced_is_none(datasets_root):
    a = _frame(["ES", "CL"])
    build.save_datasets(None, a, a, ["ES"], ["CL"])
    written = {p.name for p in datasets_root.rglob("*.csv")}
    assert "async_daily.csv" in written and "async_monthly.csv" in written
    assert "sync_daily.csv" not in written


def test_save_datasets_returns_none_for_the_sync_frames(datasets_root):
    a = _frame(["ES", "CL"])
    t1_sync, t1_async, t1_m, t2_sync, t2_async, t2_m = build.save_datasets(
        None, a, a, ["ES"], ["CL"])
    assert t1_sync is None and t2_sync is None
    assert t1_async is not None and t2_async is not None


def test_save_usd_datasets_writes_no_sync_usd_when_fx_sync_is_none(datasets_root):
    a = _frame(["ES"])
    build.save_usd_datasets(None, a, None, a, {"ES": "USD"},
                            _frame(["USD"]), None, out_root=datasets_root)
    written = {p.name for p in datasets_root.rglob("*_usd.csv")}
    assert not any(n.startswith("sync") for n in written)
