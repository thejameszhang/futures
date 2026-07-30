from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq

from globalmacro.utils.paths import TICKHISTORY_PATH

# Mirror the consumer's scan_csv overrides EXACTLY (tickhistory.py:28 / :64).
SCHEMA_OVERRIDES: dict[str, dict[str, type[pl.DataType]]] = {
    "trades": {"Price": pl.Float64, "Volume": pl.Int64},
    "quotes": {"Close Bid": pl.Float64, "Close Ask": pl.Float64, "GMT Offset": pl.Float32},
}


def report_kind(monolith: Path) -> str:
    kind = monolith.parent.name
    if kind not in SCHEMA_OVERRIDES:
        raise ValueError(f"cannot infer report kind from {monolith} (parent={kind!r})")
    return kind


def shard_dir_name(csv_filename: str) -> str:
    return csv_filename[:-4] if csv_filename.endswith(".csv") else csv_filename


def shard_dir(kind: str, stem: str) -> Path:
    return TICKHISTORY_PATH / kind / stem


def capture_schema(monolith: Path, kind: str) -> pl.Schema:
    """The exact schema the consumer's scan_csv yields (overrides + default inference)."""
    return pl.scan_csv(monolith, schema_overrides=SCHEMA_OVERRIDES[kind]).collect_schema()


def split_monolith(monolith: Path, out_dir: Path, kind: str,
                   batch_rows: int = 2_000_000) -> list[str]:
    schema = capture_schema(monolith, kind)
    pinned = dict(schema)                       # pin EVERY column -> no per-batch inference drift
    arrow_schema = pl.DataFrame(schema=schema).to_arrow().schema

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    writers: dict[str, pq.ParquetWriter] = {}
    reader = pl.read_csv_batched(monolith, schema_overrides=pinned, batch_size=batch_rows)
    try:
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            for df in batches:
                df = df.with_columns(pl.col("Date-Time").str.slice(0, 7).alias("_ym"))
                invalid = pl.col("_ym").is_null() | ~pl.col("_ym").str.contains(r"^\d{4}-\d{2}$")
                bad = df.filter(invalid)
                if bad.height:
                    sample = bad["Date-Time"].head(3).to_list()
                    raise ValueError(
                        f"{bad.height} row(s) in {monolith.name} have a null/malformed Date-Time "
                        f"(cannot derive YYYY-MM); e.g. {sample!r}. Refusing to split (would drop rows)."
                    )
                for ym in df["_ym"].unique().sort().to_list():
                    sub = df.filter(pl.col("_ym") == ym).drop("_ym")
                    if ym not in writers:
                        writers[ym] = pq.ParquetWriter(tmp / f"{ym}.parquet", arrow_schema)
                    writers[ym].write_table(sub.to_arrow().cast(arrow_schema))
    finally:
        for w in writers.values():
            w.close()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp.rename(out_dir)
    return sorted(writers.keys())


def _digest(lf: pl.LazyFrame) -> pl.DataFrame:
    # Order-INDEPENDENT, exact content check: a whole-row hash summed per (#RIC, month).
    cols = lf.collect_schema().names()          # the 7 original columns (before we add _ym)
    # engine="streaming": the group state is tiny (~RICs×months), but a non-streaming
    # collect materializes the whole per-row struct-hash column (hundreds of millions of
    # rows -> ~150 GB on the 25 GB file). Streaming aggregates in bounded chunks.
    return (lf.with_columns(pl.col("Date-Time").str.slice(0, 7).alias("_ym"))
              .group_by(["#RIC", "_ym"])
              .agg([pl.len().alias("_n"), pl.struct(cols).hash().sum().alias("_rowhash")])
              .sort(["#RIC", "_ym"]).collect(engine="streaming"))


def verify_shards(monolith: Path, out_dir: Path, kind: str, full: bool = False) -> None:
    schema = capture_schema(monolith, kind)
    csv = pl.scan_csv(monolith, schema_overrides=dict(schema))
    par = pl.scan_parquet(out_dir / "*.parquet")

    assert par.collect_schema() == schema, (
        f"Gate1 schema mismatch:\n  csv={dict(schema)}\n  parquet={dict(par.collect_schema())}")

    n_csv = csv.select(pl.len()).collect().item()
    n_par = par.select(pl.len()).collect().item()
    assert n_csv == n_par, f"Gate1 row count mismatch: csv={n_csv} parquet={n_par}"

    assert _digest(csv).equals(_digest(par)), "Gate1 per-(RIC,month) whole-row hash digest mismatch"

    if full:
        cols = list(schema.keys())
        a = csv.sort(cols).collect()
        b = par.sort(cols).collect()
        assert a.equals(b), "Gate1 exact multiset mismatch"


_MARKER = "_GATE1_OK"


def convert_one(monolith: Path, full_check: bool = False, force: bool = False) -> Path:
    kind = report_kind(monolith)
    out = shard_dir(kind, shard_dir_name(monolith.name))
    if (out / _MARKER).exists() and not force:
        print(f"skip (already verified): {out}")
        return out
    print(f"splitting {monolith} -> {out}")
    split_monolith(monolith, out, kind)
    verify_shards(monolith, out, kind, full=full_check)
    (out / _MARKER).write_text("gate1 ok\n")
    print(f"Gate 1 OK: {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "verify"])
    ap.add_argument("target", help="a monolith .csv OR a 'trades'/'quotes' dir under TICKHISTORY_PATH")
    ap.add_argument("--full-check", action="store_true", help="also run the exact multiset check (small files)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    target = Path(args.target)
    monoliths = sorted(target.glob("*.csv")) if target.is_dir() else [target]
    for m in monoliths:
        if args.cmd == "split":
            convert_one(m, full_check=args.full_check, force=args.force)
        else:
            verify_shards(m, shard_dir(report_kind(m), shard_dir_name(m.name)),
                          report_kind(m), full=args.full_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
