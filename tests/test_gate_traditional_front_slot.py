"""A guard that cannot be shown to fire is not a guard: these tests exercise
scripts/gate_traditional_front_slot.py's `run()` directly against small synthetic
debug tables, so the comparison logic is checked without depending on the real
data/tickhistory/debug/tables tree being present or unchanged.
"""
from __future__ import annotations

import polars as pl

from scripts import gate_traditional_front_slot as gate


def _write_table(root, name: str, rows: dict):
    p = root / f"TRADITIONAL_{name}.csv"
    pl.DataFrame(rows).write_csv(p)
    return p


def _base_rows(front_month, settlement_c1, settlement_c2, settlement_c3, settlement_c4, front_month_settlement):
    return {
        "front_month": front_month,
        "settlement_c1": settlement_c1,
        "settlement_c2": settlement_c2,
        "settlement_c3": settlement_c3,
        "settlement_c4": settlement_c4,
        "front_month_settlement": front_month_settlement,
    }


def test_agreeing_table_reports_zero_mismatches_and_the_full_slot_domain(tmp_path):
    _write_table(tmp_path, "A", _base_rows(
        front_month=[1, 2, 3, 4, 100],
        settlement_c1=[10.0, 20.0, 30.0, 40.0, 10.0],
        settlement_c2=[11.0, 21.0, 31.0, 41.0, 11.0],
        settlement_c3=[12.0, 22.0, 32.0, 42.0, 12.0],
        settlement_c4=[13.0, 23.0, 33.0, 43.0, 13.0],
        # front_month == 100 is out of range, so front_slot falls back to 1 -> settlement_c1
        front_month_settlement=[10.0, 21.0, 32.0, 43.0, 10.0],
    ))
    rows, mismatches, domain, nulls = gate.run(tmp_path)
    assert rows == 5
    assert mismatches == 0
    assert domain == {1, 2, 3, 4}
    assert nulls == 0


def test_a_shipped_price_disagreeing_with_settlement_c_front_slot_is_caught(tmp_path):
    # front_month == 2 means the correct slot is 2 (settlement_c2 == 21.0), but the
    # table's shipped front_month_settlement carries settlement_c1's value instead --
    # exactly the shape a wrong slot rule (or a wrong price chain) produces.
    _write_table(tmp_path, "B", _base_rows(
        front_month=[2],
        settlement_c1=[20.0],
        settlement_c2=[21.0],
        settlement_c3=[22.0],
        settlement_c4=[23.0],
        front_month_settlement=[20.0],
    ))
    rows, mismatches, domain, nulls = gate.run(tmp_path)
    assert rows == 1
    assert mismatches == 1


def test_null_vs_value_mismatches_are_counted_not_swallowed(tmp_path):
    # A naive (got != shipped).fill_null(False) treats a null-against-value pair as
    # "not unequal" and reports zero here; ne_missing must not. A second, agreeing row
    # is included so front_month_settlement round-trips through CSV as Float64 rather
    # than an all-null column typing as String (infer_schema_length=None scans the
    # whole column, but an entirely-null column still has nothing numeric to infer).
    _write_table(tmp_path, "C", _base_rows(
        front_month=[1, 1],
        settlement_c1=[5.0, 5.0],
        settlement_c2=[6.0, 6.0],
        settlement_c3=[7.0, 7.0],
        settlement_c4=[8.0, 8.0],
        front_month_settlement=[None, 5.0],
    ))
    rows, mismatches, domain, nulls = gate.run(tmp_path)
    assert rows == 2
    assert mismatches == 1, "ne_missing must count a null-vs-value disagreement"

    naive_would_report = int(
        (pl.Series([5.0]) != pl.Series([None], dtype=pl.Float64)).fill_null(False).sum()
    )
    assert naive_would_report == 0, "the naive comparator this test guards against should stay 0 here"


def test_no_traditional_tables_found_is_a_failure_not_a_silent_pass(tmp_path):
    rows, mismatches, domain, nulls = gate.run(tmp_path)
    assert rows == 0
    assert gate.main([str(tmp_path)]) == 1


def test_main_exits_nonzero_on_a_mismatching_table(tmp_path, capsys):
    _write_table(tmp_path, "D", _base_rows(
        front_month=[3],
        settlement_c1=[1.0],
        settlement_c2=[2.0],
        settlement_c3=[3.0],
        settlement_c4=[4.0],
        front_month_settlement=[999.0],
    ))
    assert gate.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
