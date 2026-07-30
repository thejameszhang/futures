# tests/test_tickhistory_shards.py
from pathlib import Path

import polars as pl
import pytest

from globalmacro.pipeline import tickhistory_shards as ts


def test_registry_and_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    assert ts.SCHEMA_OVERRIDES["trades"] == {"Price": pl.Float64, "Volume": pl.Int64}
    assert ts.SCHEMA_OVERRIDES["quotes"] == {
        "Close Bid": pl.Float64, "Close Ask": pl.Float64, "GMT Offset": pl.Float32}
    assert ts.report_kind(tmp_path / "trades" / "tier1_commodity_trades.csv") == "trades"
    assert ts.report_kind(tmp_path / "quotes" / "tier1_commodity_quotes.csv") == "quotes"
    assert ts.shard_dir("trades", "tier1_commodity_trades") == tmp_path / "trades" / "tier1_commodity_trades"
    assert ts.shard_dir_name("tier1_commodity_trades.csv") == "tier1_commodity_trades"


def test_report_kind_rejects_unknown_and_shard_dir_name_no_suffix(tmp_path):
    with pytest.raises(ValueError):
        ts.report_kind(tmp_path / "quaffles" / "tier1_commodity_trades.csv")
    assert ts.shard_dir_name("tier1_commodity_trades") == "tier1_commodity_trades"


def _write_trades_csv(path: Path) -> None:
    path.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
        "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Trade,1580,1\n"
    )


def test_capture_schema(tmp_path):
    from globalmacro.pipeline import tickhistory_shards as ts
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    _write_trades_csv(m)
    sch = ts.capture_schema(m, "trades")
    assert sch["Price"] == pl.Float64
    assert sch["Volume"] == pl.Int64
    assert sch["Date-Time"] == pl.String          # never a native timestamp
    assert list(sch.keys()) == ["#RIC", "Domain", "Date-Time", "GMT Offset", "Type", "Price", "Volume"]


def test_split_monolith_partitions_by_month(tmp_path):
    from globalmacro.pipeline import tickhistory_shards as ts
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    # RIC-major: all of GCc1 (2 months), then SIc1 (2 months) — months interleave across the file.
    m.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
        "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Trade,1580,1\n"
        "SIc1,Market Price,2020-01-07T14:30:00.000000000Z,+0,Trade,18.1,5\n"
        "SIc1,Market Price,2020-02-11T14:31:00.000000000Z,+0,Trade,18.4,2\n"
    )
    out = tmp_path / "trades" / "tier1_commodity_trades"
    months = ts.split_monolith(m, out, "trades", batch_rows=1)  # batch_rows=1 forces multi-batch routing
    assert months == ["2020-01", "2020-02"]
    assert (out / "2020-01.parquet").exists() and (out / "2020-02.parquet").exists()
    jan = pl.read_parquet(out / "2020-01.parquet")
    assert jan.columns == ["#RIC", "Domain", "Date-Time", "GMT Offset", "Type", "Price", "Volume"]
    assert set(jan["#RIC"]) == {"GCc1", "SIc1"}          # both RICs land in the Jan shard
    assert jan.schema["Date-Time"] == pl.String
    assert (tmp_path / "trades" / "tier1_commodity_trades.tmp").exists() is False  # tmp cleaned up

    # Fidelity: no row dropped, no row duplicated across shards.
    total_rows = sum(pl.read_parquet(out / f"{m}.parquet").height for m in months)
    assert total_rows == 4

    # Feb shard exists and contains both RICs' Feb rows.
    feb = pl.read_parquet(out / "2020-02.parquet")
    assert set(feb["#RIC"]) == {"GCc1", "SIc1"}

    # A specific value survives the round-trip untouched.
    gc_jan = jan.filter(pl.col("#RIC") == "GCc1")
    assert gc_jan["Price"].to_list() == [1550.5]
    assert gc_jan["Volume"].to_list() == [3]


def test_split_monolith_rejects_bad_datetime(tmp_path):
    from globalmacro.pipeline import tickhistory_shards as ts
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    # One row has a blank Date-Time — must fail loud, not silently drop the row.
    m.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
        "GCc1,Market Price,,+0,Trade,1.0,1\n"
    )
    out = tmp_path / "trades" / "tier1_commodity_trades"
    with pytest.raises(ValueError):
        ts.split_monolith(m, out, "trades", batch_rows=1)


def test_verify_shards_pass_and_fail(tmp_path):
    import pytest

    from globalmacro.pipeline import tickhistory_shards as ts
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    m.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
        "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Trade,1580,1\n"
        "SIc1,Market Price,2020-01-07T14:30:00.000000000Z,+0,Trade,18.1,5\n"
    )
    out = tmp_path / "trades" / "tier1_commodity_trades"
    ts.split_monolith(m, out, "trades", batch_rows=2)
    ts.verify_shards(m, out, "trades", full=True)     # faithful -> no raise

    # (a) numeric corruption: change a Price -> whole-row hash digest (no full) catches it.
    bad = pl.read_parquet(out / "2020-01.parquet").with_columns(
        pl.when(pl.col("#RIC") == "SIc1").then(pl.lit(999.0)).otherwise(pl.col("Price")).alias("Price"))
    bad.write_parquet(out / "2020-01.parquet")
    with pytest.raises(AssertionError):
        ts.verify_shards(m, out, "trades", full=False)

    # (b) NON-numeric corruption: change GMT Offset -> the whole-row digest still catches it.
    ts.split_monolith(m, out, "trades", batch_rows=2)                     # rebuild clean
    bad2 = pl.read_parquet(out / "2020-01.parquet").with_columns(
        pl.when(pl.col("#RIC") == "SIc1").then(pl.lit("+9")).otherwise(pl.col("GMT Offset")).alias("GMT Offset"))
    bad2.write_parquet(out / "2020-01.parquet")
    with pytest.raises(AssertionError):
        ts.verify_shards(m, out, "trades", full=False)


def test_split_and_verify_quotes_roundtrip(tmp_path):
    from globalmacro.pipeline import tickhistory_shards as ts
    m = tmp_path / "quotes" / "tier1_commodity_quotes.csv"
    m.parent.mkdir(parents=True)
    # RIC-major: all of GCc1 (2 months), then SIc1 (2 months) — months interleave across the file.
    m.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Close Bid,Close Ask\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Intraday 1Min,1550.0,1551.0\n"
        "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Intraday 1Min,1580.0,1580.5\n"
        "SIc1,Market Price,2020-01-07T14:30:00.000000000Z,+0,Intraday 1Min,18.1,18.2\n"
        "SIc1,Market Price,2020-02-11T14:31:00.000000000Z,+0,Intraday 1Min,18.4,18.5\n"
    )
    out = tmp_path / "quotes" / "tier1_commodity_quotes"
    months = ts.split_monolith(m, out, "quotes", batch_rows=1)  # batch_rows=1 forces multi-batch routing
    assert months == ["2020-01", "2020-02"]
    ts.verify_shards(m, out, "quotes", full=True)          # faithful -> no raise

    # GMT Offset (Float32) is the one dtype with no downstream cast -- confirm it survives
    # both capture_schema and the actual Parquet round-trip.
    assert ts.capture_schema(m, "quotes")["GMT Offset"] == pl.Float32
    assert pl.read_parquet(out / "2020-01.parquet").schema["GMT Offset"] == pl.Float32


def test_convert_one_is_idempotent(monkeypatch, tmp_path):
    from globalmacro.pipeline import tickhistory_shards as ts
    # Confine shard_dir() (built from module-level TICKHISTORY_PATH) to tmp_path,
    # mirroring test_registry_and_paths -- otherwise convert_one would resolve its
    # output under the real, production data/tickhistory tree.
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    m.write_text(
        "#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
        "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
    )
    out = ts.convert_one(m, full_check=True)
    assert (out / "_GATE1_OK").exists()
    mtime = (out / "2020-01.parquet").stat().st_mtime_ns
    ts.convert_one(m, full_check=True)                     # second call: skipped, no rewrite
    assert (out / "2020-01.parquet").stat().st_mtime_ns == mtime
