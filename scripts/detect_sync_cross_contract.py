"""Independent, value-only detector for sync-panel returns computed between two
different contracts.

Verification only -- this never runs as part of the pipeline. It exists to check
the provenance guard (`apply_provenance_guard` in `tickhistory.py`) from the
outside: it does not import `attach_front_slot`, `provenance_break_mask`,
`apply_provenance_guard`, `rescue_ret1_adjusted`, or `is_rescuable_mask`, and it
never reconstructs the slot the guard recorded. What it reads instead is the
settlement grid a panel actually shipped -- `settlement_c1..c4` and the finalised
`ret1_adjusted` -- from the debug tables (the shipped dataset CSVs under
`datasets/` carry only `date_` and the symbol columns; the settlement grid
survives only in `data/tickhistory/debug/tables`).

For every shipped, non-null return it enumerates which ordered
`(numerator slot, denominator slot)` pair could have produced that exact value as
`settlement_a(t) / settlement_b(t-1) - 1`, and calls a cell cross-contract only
when EVERY pair that reproduces it has `a != b` -- i.e. no same-slot story exists
for the number that shipped. A cell some same-slot pair also reproduces is left
alone even if a cross-slot pair matches too: a legitimate roll can compute its
return across two slots on purpose (the same contract just moved slots), and from
the numbers alone that looks identical to a defect, so a same-slot match always
wins. A cell no pair reproduces at all is undecidable, not cross-contract: the
detector only flags a cell when it can name the pair responsible.

This is a genuinely different method from the guard, not a restatement of it: the
guard carries a label recorded once, at price-selection time, and never revisited;
this detector never sees that label and re-derives consistency purely from the
prices that shipped. The two are expected to disagree on rows the value alone
cannot decide -- see the module docstring in `tickhistory.apply_provenance_guard`
for the scoped legitimate-roll cases this produces. A cell this detector still
calls cross-contract in scope after the guard runs is the finding that matters;
a gap between the two counts on rows outside that scope is not.

Tolerance -- measured against the shipped debug tables, not assumed:
For 642,453 cells whose `front_month_settlement` is independently confirmed (by
direct equality against `settlement_c{i}`, never by this module) to have used the
SAME slot at both ends, the matching same-slot candidate reproduces the shipped
value to a residual of exactly 0.0 for every single one -- the CSV round-trip
loses nothing, and the arithmetic this module repeats is bit-identical to what the
pipeline already did. Among cells where that same check confirms the slot
genuinely moved, the smallest observed gap between the shipped value and any
candidate pair is ~1.4e-9, and three quarters of that population misses every
same-slot candidate by far more (typical gaps run from ~1e-8 into double digits).
No residual was observed between 2.3e-16 (the largest float-rounding noise on a
coincidental tie, e.g. a day `settlement_c1 == settlement_c2` exactly) and 1.4e-9 --
that gap is about six orders of magnitude wide and completely empty. DEFAULT_TOL
sits in the middle of it, with ~1,000x headroom against float noise and >10x
headroom against the closest genuine mismatch.

    .venv/bin/python scripts/detect_sync_cross_contract.py [tables_dir] [--tol TOL]
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import polars as pl

SLOTS = (1, 2, 3, 4)
DEFAULT_TOL = 1e-10


def _present_slots(df: pl.DataFrame) -> list[int]:
    return [i for i in SLOTS if f"settlement_c{i}" in df.columns]


def _match_frame(df: pl.DataFrame, slots: list[int], tol: float) -> pl.DataFrame:
    """Attach one boolean column per ordered (num, den) slot pair: whether
    settlement_c{num}(t) / settlement_c{den}(t-1) - 1 reproduces the shipped
    ret1_adjusted(t) within tol. Null on either side of the ratio cannot match
    anything, hence fill_null(False) rather than letting a null comparison
    disappear into "no pair matches" silently the wrong way.
    """
    exprs = []
    for num in slots:
        for den in slots:
            candidate = pl.col(f"settlement_c{num}") / pl.col(f"settlement_c{den}").shift(1) - 1
            hit = ((candidate - pl.col("ret1_adjusted")).abs() <= tol).fill_null(False)
            exprs.append(hit.alias(f"_match_{num}_{den}"))
    return df.with_columns(exprs)


def cross_contract_cells(df: pl.DataFrame, tol: float = DEFAULT_TOL) -> pl.DataFrame:
    """Rows whose shipped ret1_adjusted is explained ONLY by slot pairs with
    num != den. Requires at least two settlement_c{i} columns and a
    ret1_adjusted column; returns an empty frame (same schema) otherwise.
    """
    slots = _present_slots(df)
    if len(slots) < 2 or "ret1_adjusted" not in df.columns:
        return df.clear()
    matched = _match_frame(df, slots, tol)
    same_cols = [f"_match_{i}_{i}" for i in slots]
    cross_cols = [f"_match_{n}_{d}" for n in slots for d in slots if n != d]
    same_slot_ok = pl.any_horizontal(same_cols)
    cross_slot_ok = pl.any_horizontal(cross_cols)
    flagged = pl.col("ret1_adjusted").is_not_null() & cross_slot_ok & ~same_slot_ok
    return matched.filter(flagged).select(df.columns)


def summarize(df: pl.DataFrame, tol: float = DEFAULT_TOL) -> dict[str, int]:
    """Per-table counts for reporting: how many shipped (non-null) cells are
    clean (some same-slot pair reproduces the value), cross_contract (only
    cross-slot pairs do), undecidable (no pair does), and ambiguous (more than
    one DISTINCT slot pair reproduces the value). `ambiguous` is a tolerance
    diagnostic, not a fourth classification -- clean/cross_contract/undecidable
    always sum to `shipped` on their own.
    """
    slots = _present_slots(df)
    empty = {"shipped": 0, "clean": 0, "cross_contract": 0, "undecidable": 0, "ambiguous": 0}
    if len(slots) < 2 or "ret1_adjusted" not in df.columns:
        return empty
    matched = _match_frame(df, slots, tol)
    all_cols = [f"_match_{n}_{d}" for n in slots for d in slots]
    same_cols = [f"_match_{i}_{i}" for i in slots]
    cross_cols = [c for c in all_cols if c not in same_cols]
    shipped = matched.filter(pl.col("ret1_adjusted").is_not_null())
    if shipped.height == 0:
        return empty
    same_slot_ok = pl.any_horizontal(same_cols)
    cross_slot_ok = pl.any_horizontal(cross_cols)
    n_matching_pairs = pl.sum_horizontal([pl.col(c).cast(pl.Int32) for c in all_cols])
    labeled = shipped.with_columns([
        same_slot_ok.alias("_clean"),
        (cross_slot_ok & ~same_slot_ok).alias("_cross"),
        (~cross_slot_ok & ~same_slot_ok).alias("_undecidable"),
        (n_matching_pairs > 1).alias("_ambiguous"),
    ])
    return {
        "shipped": shipped.height,
        "clean": int(labeled["_clean"].sum()),
        "cross_contract": int(labeled["_cross"].sum()),
        "undecidable": int(labeled["_undecidable"].sum()),
        "ambiguous": int(labeled["_ambiguous"].sum()),
    }


def run(tables_dir: Path, tol: float = DEFAULT_TOL) -> dict[str, int]:
    """Scan every *.csv in tables_dir and aggregate summarize()'s counts, printing
    a per-file line wherever a cross-contract cell is found. Does not write
    anything -- registering a prediction from these numbers is a separate step
    with its own sign-off, not this script's job.
    """
    totals = {"files": 0, "shipped": 0, "clean": 0, "cross_contract": 0, "undecidable": 0, "ambiguous": 0}
    for f in sorted(glob.glob(str(Path(tables_dir) / "*.csv"))):
        d = pl.read_csv(f, try_parse_dates=True, infer_schema_length=None)
        if "ret1_adjusted" not in d.columns or "settlement_c1" not in d.columns:
            continue
        # A settlement_c{i} column that is null on every row (some debug tables
        # never populate c3/c4) is inferred as String rather than Float64, even
        # with the whole file scanned above -- strict=False turns that into an
        # all-null Float64 column instead of raising on the division below.
        numeric_cols = [c for c in d.columns if c.startswith("settlement_c") or c == "ret1_adjusted"]
        d = d.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in numeric_cols])
        counts = summarize(d, tol)
        if counts["shipped"] == 0:
            continue
        totals["files"] += 1
        for key in ("shipped", "clean", "cross_contract", "undecidable", "ambiguous"):
            totals[key] += counts[key]
        if counts["cross_contract"]:
            print(
                f"  {Path(f).name}: cross_contract={counts['cross_contract']} "
                f"undecidable={counts['undecidable']} ambiguous={counts['ambiguous']}"
            )
    return totals


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    tol = DEFAULT_TOL
    positional = []
    i = 0
    while i < len(argv):
        if argv[i] == "--tol":
            tol = float(argv[i + 1])
            i += 2
        else:
            positional.append(argv[i])
            i += 1
    tables_dir = Path(positional[0]) if positional else Path("data/tickhistory/debug/tables")
    totals = run(tables_dir, tol)
    print(f"files scanned: {totals['files']}")
    print(f"shipped non-null returns: {totals['shipped']}")
    print(
        f"clean: {totals['clean']}  cross_contract: {totals['cross_contract']}  "
        f"undecidable: {totals['undecidable']}  ambiguous: {totals['ambiguous']}"
    )
    if totals["files"] == 0:
        print("no debug tables with a settlement grid found -- nothing was checked")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
