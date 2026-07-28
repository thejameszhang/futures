"""Finder: currency-return spikes in the shipped prod column (sync_daily.csv).
Used to author corrections and as the re-scan acceptance gate."""
from __future__ import annotations

import polars as pl

from globalmacro.utils.config import load_config
from globalmacro.utils.paths import DATASETS_ROOT, PROJECT_ROOT

THRESHOLD = 0.30


def currency_symbols() -> list[str]:
    cfg = load_config(PROJECT_ROOT / "tier1.yaml") + load_config(PROJECT_ROOT / "tier2.yaml")
    syms = {f.symbol for f in cfg if any(ac.name.lower() == "currency" for ac in f.asset_class)}
    return sorted(syms | {"DM"})


def scan_prod_currency_spikes(
    sync_daily: pl.DataFrame, currency_symbols: list[str], threshold: float = THRESHOLD
) -> pl.DataFrame:
    df = sync_daily.with_columns(pl.col("date").cast(pl.Date, strict=False))
    rows = []
    for sym in currency_symbols:
        if sym not in df.columns:
            continue
        s = df.select("date", pl.col(sym).cast(pl.Float64, strict=False).alias("ret")).drop_nulls("ret")
        for r in s.filter(pl.col("ret").abs() > threshold).iter_rows(named=True):
            rows.append({"symbol": sym, "date": r["date"], "ret": r["ret"]})
    schema = {"symbol": pl.Utf8, "date": pl.Date, "ret": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def main() -> int:
    syms = currency_symbols()
    hits = []
    for tier in (1, 2):
        p = DATASETS_ROOT / f"tier{tier}" / "sync" / "sync_daily.csv"
        if not p.exists():
            continue
        df = pl.read_csv(p, infer_schema_length=None)
        hits.append(scan_prod_currency_spikes(df, syms).with_columns(pl.lit(tier).alias("tier")))
    out = pl.concat(hits) if hits else pl.DataFrame()
    print(out)
    return 1 if out.height else 0


if __name__ == "__main__":
    raise SystemExit(main())
