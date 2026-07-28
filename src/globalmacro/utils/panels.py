"""Column-presence primitives shared by build and the FX blend.

Moved out of build.py so utils/sync_fx.py can compute per-currency cutoffs without
importing build (which would be a build -> sync_fx -> build import cycle). build.py
re-imports these names, so `from globalmacro.build import first_valid_date` (used by
validation/synthetic*.py) keeps working.
"""
from __future__ import annotations

from typing import Any

import polars as pl


def is_present_expr(col: str) -> pl.Expr:
    """True where the column is neither null nor NaN. (No inf screen -- see is_finite_expr.)"""
    numeric_is_nan = (
        pl.col(col)
        .cast(pl.Float64, strict=False)
        .is_nan()
        .fill_null(False)
    )
    return pl.col(col).is_not_null() & ~numeric_is_nan


def is_finite_expr(col: str) -> pl.Expr:
    """True where the column is finite (not null, NaN, or +/-inf). The predicate the FX
    blend uses for 'the future was observed' -- inf is a bad print, never an observation."""
    return pl.col(col).cast(pl.Float64, strict=False).is_finite().fill_null(False)


def first_valid_date(df: pl.DataFrame, col: str, *, date_col: str = "date") -> Any:
    """Earliest date where `col` is present (is_present_expr). Unchanged from build.py."""
    return df.select(pl.col(date_col).filter(is_present_expr(col)).min()).to_series().item()


def first_finite_date(df: pl.DataFrame, col: str, *, date_col: str = "date") -> Any:
    """Earliest date where `col` is finite. The FX-blend cutoff: real-future start with
    inf bad prints screened out. None if the column is never finite."""
    return df.select(pl.col(date_col).filter(is_finite_expr(col)).min()).to_series().item()
