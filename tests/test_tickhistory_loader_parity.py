import polars as pl

import globalmacro.pipeline.tickhistory as th
from globalmacro.pipeline import tickhistory_shards as ts

CSV = ("#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"
       "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1550.5,3\n"
       "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Trade,1580,1\n"
       "GCc2,Market Price,2020-01-06T14:30:00.000000000Z,+0,Trade,1551.0,2\n")

QUOTES_CSV = ("#RIC,Domain,Date-Time,GMT Offset,Type,Close Bid,Close Ask\n"
              "GCc1,Market Price,2020-01-06T14:30:00.000000000Z,+0,Intraday 1Min,1550.0,1551.0\n"
              "GCc1,Market Price,2020-02-10T14:31:00.000000000Z,+0,Intraday 1Min,1580.0,1580.5\n"
              "GCc2,Market Price,2020-01-06T14:30:00.000000000Z,+0,Intraday 1Min,1551.0,1552.0\n")

def test_trades_loader_parquet_equals_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(th, "ALL_RICS", ["GCc1", "GCc2"], raising=False)   # module global set only under __main__
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    m.write_text(CSV)
    ref = (pl.scan_csv(m, schema_overrides={"Price": pl.Float64, "Volume": pl.Int64})
           .filter(pl.col("#RIC").is_in(["GCc1", "GCc2"]))
           .select(["#RIC", "Date-Time", "Price", "Volume", "GMT Offset"]).collect())
    ts.convert_one(m, full_check=True)                       # build shards, then archive the monolith path
    m.rename(tmp_path / "trades" / "_moved.csv")             # ensure the loader is NOT reading the csv
    got = th.load_trades_data("tier1_commodity_trades.csv")
    key = ["#RIC", "Date-Time", "Price", "Volume"]
    assert got.select(key).sort(key).equals(ref.select(key).sort(key))


def test_quotes_loader_parquet_equals_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(th, "ALL_RICS", ["GCc1", "GCc2"], raising=False)   # module global set only under __main__
    m = tmp_path / "quotes" / "tier1_commodity_quotes.csv"
    m.parent.mkdir(parents=True)
    m.write_text(QUOTES_CSV)
    ref = (pl.scan_csv(m, schema_overrides={
                "Close Bid": pl.Float64, "Close Ask": pl.Float64, "GMT Offset": pl.Float32})
           .filter(pl.col("#RIC").is_in(["GCc1", "GCc2"]))
           .filter(pl.col("Type") == "Intraday 1Min")
           .select(["#RIC", "Date-Time", "Close Bid", "Close Ask"]).collect())
    ts.convert_one(m, full_check=True)                       # build shards, then archive the monolith path
    m.rename(tmp_path / "quotes" / "_moved.csv")              # ensure the loader is NOT reading the csv
    got = th.load_quotes_data("tier1_commodity_quotes.csv")
    key = ["#RIC", "Date-Time", "Close Bid", "Close Ask"]
    assert got.select(key).sort(key).equals(ref.select(key).sort(key))
