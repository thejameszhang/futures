#!/usr/bin/env python3
"""Gate 2: compare two {asset}_daily_returns.csv (baseline vs shards). A column passes
if byte-identical, or every diff is a provable tie-break (identical null pattern AND
corr>=0.999999 on overlapping non-null cells). Prints offenders; exits 1 if any.

CALIBRATION CAVEAT (measured 2026-07-30). The pipeline is not bitwise reproducible:
order-sensitive `.last()` / `unique(keep='first')` ops make two runs over IDENTICAL
input differ on ~26 commodity symbols by up to ~1%. This per-cell threshold flags that
noise, so a FAIL here is not on its own evidence of a regression -- rerun the baseline
against itself first to establish the noise floor, then compare. For whole-dataset
regression checks prefer scripts/compare_datasets.py, whose median-over-instruments
>= 0.99 over the overlap window is robust to this. Once the pipeline is made bitwise
reproducible, this threshold can be tightened to exact equality and the caveat drops.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

CORR_MIN = 0.999999
CONST_TOL = 1e-9


def columns_ok(base: pl.DataFrame, cand: pl.DataFrame) -> list[str]:
    bad: list[str] = []
    if "date" in base.columns and "date" in cand.columns:
        base_date, cand_date = base["date"], cand["date"]
        if base_date.len() != cand_date.len() or not base_date.equals(cand_date):
            return ["DATE-AXIS-MISMATCH"]
    syms = [c for c in base.columns if c not in ("date", "date_") and c in cand.columns]
    for c in syms:
        b, a = base[c], cand[c]
        if b.equals(a):
            continue
        b = b.cast(pl.Float64, strict=False)
        a = a.cast(pl.Float64, strict=False)
        b_missing, a_missing = b.is_null() | b.is_nan(), a.is_null() | a.is_nan()
        if b_missing.to_list() != a_missing.to_list():
            bad.append(c)                     # null/nan pattern changed -> not a tie-break
            continue
        mask = ~b_missing & ~a_missing
        bb, aa = b.filter(mask), a.filter(mask)
        if bb.len() == 0:
            continue
        if bb.n_unique() <= 1 or aa.n_unique() <= 1:      # corr undefined on a constant column
            if float((bb - aa).abs().max() or 0.0) > CONST_TOL:  # pyright: ignore[reportArgumentType]
                bad.append(c)
            continue
        corr = pl.DataFrame({"b": bb, "a": aa}).select(pl.corr("b", "a")).item()
        if corr is None or not (corr >= CORR_MIN):
            bad.append(c)
    missing = sorted(set(base.columns) ^ set(cand.columns))
    return bad + [f"COLSET:{m}" for m in missing]


def main() -> int:
    base = pl.read_csv(Path(sys.argv[1]))
    cand = pl.read_csv(Path(sys.argv[2]))
    bad = columns_ok(base, cand)
    if bad:
        print(f"FAIL {Path(sys.argv[2]).name}: non-tie-break differences in:", ", ".join(bad))
        return 1
    print(f"OK {Path(sys.argv[2]).name}: {len(base.columns) - 1} symbol columns identical-or-tie-break")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
