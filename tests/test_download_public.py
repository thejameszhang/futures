import io
import os

import polars as pl
import pytest

from globalmacro.pipeline import download_public as dp


def test_constants_lock_the_owner_settings():
    # FRED series -> lowercase columns, ded3 the only consumed one.
    assert dp.FRED_SERIES == ["DED1", "DED3", "DED6", "EFFR", "DFF", "SOFR"]
    # OECD: 3-month interbank (IR3TIB), monthly (M), all countries (empty REF_AREA) + labels.
    assert "DSD_KEI@DF_KEI" in dp.OECD_URL
    assert "/.M.IR3TIB.....?" in dp.OECD_URL
    assert "format=csvfilewithlabels" in dp.OECD_URL
    assert dp.OECD_ACCEPT == "application/vnd.sdmx.data+csv"
    assert "F-F_Research_Data_Factors_daily_CSV.zip" in dp.FAMA_FRENCH_URL
    assert dp.FRED_URL.format(series="DED3").endswith("fredgraph.csv?id=DED3")
    assert set(dp.FETCHERS) == {"fama_french", "fred", "oecd"}


def test_trim_fama_french_drops_footer_keeps_header_and_data():
    text = (
        "This file was created by using the 202605 CRSP database.\n"
        "The Tbill return is the simple daily rate that ...\n"
        "compounds to 1-month TBill rate ...\n"
        "\n"
        ",Mkt-RF,SMB,HML,RF\n"
        "19260701,    0.09,   -0.25,   -0.27,    0.01\n"
        "20260529,    0.19,   -0.80,   -0.26,    0.02\n"
        "\n"
        "Copyright 2026 Eugene F. Fama and Kenneth R. French\n"
    )
    out = dp._trim_fama_french(text)
    lines = out.splitlines()
    assert "Copyright" not in out
    assert lines[4] == ",Mkt-RF,SMB,HML,RF"          # header still at index 4 -> load_rf skip_rows=4
    assert lines[-1] == "20260529,    0.19,   -0.80,   -0.26,    0.02"
    # round-trips through the same read load_rf uses
    rf = pl.read_csv(io.StringIO(out), skip_rows=4, schema_overrides={"RF": pl.Float64})
    assert rf.columns == ["", "Mkt-RF", "SMB", "HML", "RF"]
    assert rf.height == 2


def test_merge_fred_series_outer_joins_and_maps_dot_to_null():
    def csv(sid, rows):
        return f"observation_date,{sid}\n" + "\n".join(f"{d},{v}" for d, v in rows) + "\n"
    series_text = {
        "DED1": csv("DED1", [("1971-01-04", "5.0")]),
        "DED3": csv("DED3", [("1971-01-04", "5.5"), ("1971-01-05", ".")]),  # '.' -> null
        "DED6": csv("DED6", [("1971-01-05", "6.0")]),
        "EFFR": csv("EFFR", [("2000-01-03", "5.5")]),
        "DFF":  csv("DFF",  [("1971-01-04", "4.0")]),
        "SOFR": csv("SOFR", [("2018-04-03", "1.8")]),
    }
    df = dp._merge_fred_series(series_text)
    assert df.columns == ["date", "ded1", "ded3", "ded6", "effr", "dff", "sofr"]
    assert df["date"].is_sorted()
    row = df.filter(pl.col("date") == pl.date(1971, 1, 5)).to_dicts()[0]
    assert row["ded3"] is None            # '.' became null
    assert row["ded6"] == 6.0
    assert df.filter(pl.col("date") == pl.date(1971, 1, 4)).to_dicts()[0]["ded3"] == 5.5


def test_skip_if_present_does_not_download(tmp_path, monkeypatch):
    dest = tmp_path / "oecd.csv"
    dest.write_text("STRUCTURE,...\n")  # pretend it already exists
    called = []
    monkeypatch.setattr(dp, "_curl", lambda *a, **k: called.append(a))
    got = dp.fetch_oecd_stir(dest=dest, force=False)
    assert got == dest
    assert called == []                   # curl never invoked


def test_force_overwrites(tmp_path, monkeypatch):
    dest = tmp_path / "oecd.csv"
    dest.write_text("old\n")
    def fake_curl(url, d, accept=None):
        from pathlib import Path
        Path(d).write_text("new\n")
    monkeypatch.setattr(dp, "_curl", fake_curl)
    dp.fetch_oecd_stir(dest=dest, force=True)
    assert dest.read_text() == "new\n"


def test_main_only_routes_to_one_fetcher(monkeypatch):
    calls = []
    for name in dp.FETCHERS:
        monkeypatch.setitem(dp.FETCHERS, name, lambda force=False, _n=name: calls.append((_n, force)))
    assert dp.main(["--only", "oecd"]) == 0
    assert calls == [("oecd", False)]
    calls.clear()
    assert dp.main(["--force"]) == 0
    assert [c[0] for c in calls] == ["fama_french", "fred", "oecd"]
    assert all(force is True for _, force in calls)


def test_cli_registers_download_public():
    from globalmacro.cli import _STAGE_MODULES
    assert _STAGE_MODULES["download-public"] == "globalmacro.pipeline.download_public"


_NET = pytest.mark.skipif(
    not os.environ.get("GM_NETWORK_TESTS"),
    reason="hits FRED/OECD/Dartmouth; set GM_NETWORK_TESTS=1",
)


@_NET
def test_live_fama_french_reproduces_on_disk(tmp_path):
    from globalmacro.utils.paths import DATA_ROOT
    on_disk = DATA_ROOT / "misc" / "F-F_Research_Data_Factors_daily.csv"
    if not on_disk.exists():
        pytest.skip("no on-disk F-F factors to compare against")
    fetched = dp.fetch_fama_french(dest=tmp_path / "ff.csv", force=True)

    def rf(path):  # replicate load_rf's read (skip_rows=4; date=YYYYMMDD int, RF col)
        d = pl.read_csv(path, skip_rows=4, schema_overrides={"RF": pl.Float64})
        return d.rename({d.columns[0]: "date", "RF": "rf"}).select("date", "rf")

    j = rf(fetched).rename({"rf": "n"}).join(rf(on_disk).rename({"rf": "o"}), on="date", how="inner")
    assert j.height > 0
    assert j.filter((pl.col("n") - pl.col("o")).abs() > 1e-12).height == 0  # rf exact on shared dates


@_NET
def test_live_fred_ded3_reproduces_on_disk(tmp_path):
    from globalmacro.utils.paths import ECONOMICS_PATH
    on_disk = ECONOMICS_PATH / "ded3_wrds.csv"
    if not on_disk.exists():
        pytest.skip("no on-disk ded3_wrds.csv to compare against")
    fetched = dp.fetch_fred_rates(dest=tmp_path / "ded3.csv", force=True)
    new = pl.read_csv(fetched, schema_overrides={"date": pl.Date, "ded3": pl.Float64}).select("date", "ded3").drop_nulls()
    old = pl.read_csv(on_disk, schema_overrides={"date": pl.Date, "ded3": pl.Float64}).select("date", "ded3").drop_nulls()
    j = new.rename({"ded3": "n"}).join(old.rename({"ded3": "o"}), on="date", how="inner")
    assert j.filter((pl.col("n") - pl.col("o")).abs() > 1e-9).height == 0     # ded3 exact on shared dates
    assert old.join(new, on="date", how="anti").height == 0                   # no on-disk date dropped by the fetch


@_NET
def test_live_oecd_reproduces_on_disk(tmp_path):
    from globalmacro.utils.paths import ECONOMICS_PATH
    on_disk = ECONOMICS_PATH / "oecd.csv"
    if not on_disk.exists():
        pytest.skip("no on-disk oecd.csv to compare against")
    fetched = dp.fetch_oecd_stir(dest=tmp_path / "oecd.csv", force=True)
    k = ["Reference area", "TIME_PERIOD"]
    new = pl.read_csv(fetched, infer_schema_length=20000).select([*k, "OBS_VALUE"]).rename({"OBS_VALUE": "n"})
    old = pl.read_csv(on_disk, infer_schema_length=20000).select([*k, "OBS_VALUE"]).rename({"OBS_VALUE": "o"})
    j = old.join(new, on=k, how="left")
    assert j.filter(pl.col("n").is_null()).height == 0                        # every on-disk (area, month) still present
    # values agree except OECD's own historical restatements (small; empirically ~2% of pairs,
    # largest a 1969 Australia series-boundary point). Drift-robust: assert the bulk still matches.
    within = j.filter((pl.col("n") - pl.col("o")).abs() <= 0.05).height
    assert within / j.height > 0.98
