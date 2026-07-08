#!/usr/bin/env python3
"""Smoke-check a fresh WRDS pull under the isolated FUTURES_DATA_ROOT: every
consumed table exists, is non-empty, has the same columns as the current repo
copy, and its date coverage reaches recent data. Exit non-zero on any failure."""
from __future__ import annotations
import sys
from pathlib import Path
import polars as pl
from globalmacro.pipeline.download import PULL_SPECS
from globalmacro.utils.paths import DATASTREAM_PATH, COMPUSTAT_PATH

REPO = Path(__file__).resolve().parents[1]
REPO_DATA = REPO / "data"

# where each database's tables land, for both the fresh (env) tree and the repo copy
def _dir_for(db: str, root_datastream: Path, root_comp: Path) -> Path:
    return root_comp if db == "comp" else root_datastream / db


def _date_col(cols: list[str]) -> str | None:
    for c in cols:
        lc = c.lower()
        if lc in ("date", "date_", "datadate", "eco_date", "obsdate") or lc.endswith("date"):
            return c
    return None


def main() -> int:
    errors = []
    for db, (_lib, tables) in PULL_SPECS.items():
        fresh_dir = _dir_for(db, DATASTREAM_PATH, COMPUSTAT_PATH)
        repo_dir = _dir_for(db, REPO_DATA / "datastream", REPO_DATA / "comp")
        for t in tables:
            fresh = fresh_dir / f"{t}.csv"
            if not fresh.exists():
                errors.append(f"MISSING: {fresh}")
                continue
            fdf = pl.scan_csv(str(fresh), infer_schema_length=0)
            fcols = fdf.collect_schema().names()
            n = fdf.select(pl.len()).collect().item()
            if n == 0:
                errors.append(f"EMPTY: {fresh}")
                continue
            repo = repo_dir / f"{t}.csv"
            if repo.exists():
                rcols = pl.scan_csv(str(repo), infer_schema_length=0).collect_schema().names()
                if set(fcols) != set(rcols):
                    errors.append(f"COLS differ for {t}: fresh={sorted(fcols)} repo={sorted(rcols)}")
            dc = _date_col(fcols)
            mx = None
            if dc is not None:
                mx = (pl.scan_csv(str(fresh), infer_schema_length=0)
                      .select(pl.col(dc).str.slice(0, 4).cast(pl.Int32, strict=False).max())
                      .collect().item())
                # Only a PLAUSIBLE-but-old coverage year is a real staleness signal.
                # An implausible max (e.g. a contract-metadata column like
                # dsfutcontrinfo reading year 1299) means the detected column isn't
                # a coverage date -> don't flag it.
                if mx is not None and 1990 <= mx < 2025:
                    errors.append(f"STALE {t}: max year {mx} < 2025 (fresh pull should reach recent data)")
            print(f"OK  {db}/{t}: rows={n} cols={len(fcols)} max_year={mx}")
    for e in errors:
        print("FAIL:", e)
    print(f"\n{len(errors)} smoke errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
