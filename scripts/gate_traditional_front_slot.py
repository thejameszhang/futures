"""Verification gate for `front_slot` on the traditional-cycle branch of `process_future`.

The main Step 5 gate for `attach_front_slot` (see `tests/test_sync_provenance.py` and the
sync-provenance task history) excludes every `TRADITIONAL_*` debug table by design -- the
traditional branch is built by a different code path (`attach_traditional_front_slot`) and
applying the non-traditional rule to it produces thousands of false mismatches. That leaves
the traditional slot rule with no check against real data at all: a mutation to it can pass
the unit suite (which only checks synthetic `front_month` values) and still reach production
undetected until a full rerun. This script closes that gap directly against the debug tables,
with no pipeline stage run.

    .venv/bin/python scripts/gate_traditional_front_slot.py [tables_dir]

`tables_dir` defaults to `data/tickhistory/debug/tables`. For every `TRADITIONAL_*.csv`
found there, it recomputes `front_slot` from the table's own `front_month` column via
`attach_traditional_front_slot`, then asserts that the price already shipped in
`front_month_settlement` equals `settlement_c{front_slot}` -- built directly from the table's
`settlement_c1..c4` columns, independently of the recomputed `front_month_settlement`, so a
slot-only regression (the price chain untouched, only the slot rule broken) cannot cancel
itself out of the comparison.

Two things here are load-bearing, mirroring the non-traditional gate:

  - **`infer_schema_length=None`.** At the default (100 rows), a table whose first 100
    `settlement_*` rows are null gets those columns typed `String`, and the comparison then
    raises `ComputeError: cannot compare string with numeric type (f64)`.
  - **`ne_missing`, not `!=`.** `(got != shipped).fill_null(False)` is null-blind: a
    null-against-value pair compares to null and is swallowed by `fill_null(False)`, reporting
    zero mismatches when the true count is nonzero. That is exactly the shape a broken
    fallback (`.otherwise(...)`) produces.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import polars as pl

from globalmacro.pipeline.tickhistory import attach_traditional_front_slot

_NEEDED = {"front_month", "settlement_c1", "settlement_c2", "settlement_c3", "settlement_c4", "front_month_settlement"}

_PICK_BY_SLOT = (
    pl.when(pl.col("front_slot") == 1).then(pl.col("settlement_c1"))
    .when(pl.col("front_slot") == 2).then(pl.col("settlement_c2"))
    .when(pl.col("front_slot") == 3).then(pl.col("settlement_c3"))
    .when(pl.col("front_slot") == 4).then(pl.col("settlement_c4"))
    .otherwise(None)
    .alias("picked")
)


def run(tables_dir: Path) -> tuple[int, int, set, int]:
    """Returns (total_rows, total_mismatches, front_slot_domain, front_slot_nulls)."""
    total_rows = 0
    total_mismatches = 0
    domain: set = set()
    slot_nulls = 0
    for f in sorted(glob.glob(str(tables_dir / "TRADITIONAL_*.csv"))):
        d = pl.read_csv(f, try_parse_dates=True, infer_schema_length=None)
        if not _NEEDED.issubset(set(d.columns)):
            print(f"  SKIP {f}: missing columns")
            continue
        shipped = d["front_month_settlement"]
        recomputed = attach_traditional_front_slot(d.drop("front_month_settlement"))
        slot = recomputed["front_slot"]
        picked = recomputed.select(_PICK_BY_SLOT)["picked"]
        n = int(shipped.ne_missing(picked).sum())
        total_rows += d.height
        total_mismatches += n
        domain |= set(slot.unique().to_list())
        slot_nulls += slot.null_count()
        if n:
            print(f"  MISMATCH {f}: {n}")
    return total_rows, total_mismatches, domain, slot_nulls


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tables_dir = Path(argv[0]) if argv else Path("data/tickhistory/debug/tables")
    rows, mismatches, domain, nulls = run(tables_dir)
    print(f"traditional rows: {rows}")
    print(f"total mismatches (ne_missing): {mismatches}")
    print(f"front_slot domain: {sorted(domain)}  nulls: {nulls}")
    if rows == 0:
        print("no TRADITIONAL_*.csv tables found -- nothing was checked")
        return 1
    ok = mismatches == 0 and domain == {1, 2, 3, 4} and nulls == 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
