import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import verify_tickhistory_shards as v  # noqa: E402


def test_tiebreak_accept_and_reject():
    base = pl.DataFrame({"date": [1, 2, 3], "GC": [0.01, -0.02, 0.03], "SI": [0.0, 0.1, None]})
    ok = base.with_columns(pl.col("GC") + 1e-12)          # tie-break-scale, same null pattern -> accept
    assert v.columns_ok(base, ok) == []
    bad = base.with_columns(pl.when(pl.col("date") == 1).then(None).otherwise(pl.col("GC")).alias("GC"))
    assert "GC" in v.columns_ok(base, bad)                # a moved NaN -> reject


def test_new_nan_is_rejected():
    base = pl.DataFrame({"date": [1, 2, 3], "GC": [0.01, 0.02, 0.03]})
    cand = base.with_columns(pl.Series("GC", [0.01, float("nan"), 0.03]))
    assert "GC" in v.columns_ok(base, cand)                # a real number turned NaN -> reject


def test_string_symbol_columns_do_not_crash():
    # Illiquid contracts in real {asset}_daily_returns.csv infer as String, not
    # Float64 -- columns_ok must cast (strict=False) before is_nan() instead of crashing.
    base = pl.DataFrame({"date": [1, 2, 3], "SR3": pl.Series(["0.01", "", "0.03"])})
    assert v.columns_ok(base, base) == []                  # identical String columns -> accept

    moved_null = base.with_columns(pl.Series("SR3", ["0.01", "0.05", "0.03"]))
    assert "SR3" in v.columns_ok(base, moved_null)          # empty cell -> real value -> reject

    dates = list(range(1, 9))
    base_str = [f"{0.01 * n:.3f}" for n in range(1, 9)]
    cand_str = list(base_str)
    cand_str[0] = f"{0.01 + 1e-9:.12f}"                     # tie-break-scale nudge on one cell
    long_base = pl.DataFrame({"date": dates, "SR3": pl.Series(base_str)})
    long_cand = pl.DataFrame({"date": dates, "SR3": pl.Series(cand_str)})
    assert v.columns_ok(long_base, long_cand) == []         # still ties out at CORR_MIN -> accept


def test_date_axis_mismatch_is_rejected():
    base = pl.DataFrame({"date": [1, 2, 3], "GC": [0.01, -0.02, 0.03]})
    cand = pl.DataFrame({"date": [1, 2, 4], "GC": [0.01, -0.02, 0.03]})
    assert "DATE-AXIS-MISMATCH" in v.columns_ok(base, cand)  # shifted axis, positional match -> reject
