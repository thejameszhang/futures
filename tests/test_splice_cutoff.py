"""coalesce_before_cutoff: the splice fix that stops a discontinued contract's
history from resurrecting a null a downstream guard (the tickhistory provenance
guard, among others) deliberately left inside the active contract's own live range.

A plain `pl.coalesce([primary, donor])` cannot distinguish "primary is null because
this predates the splice" from "primary is null because something deliberately left
this cell empty" -- both are just a null to coalesce. Restricting `donor` to dates
strictly before `primary`'s own first observation removes the second case without
touching the first: a splice's only legitimate job is backfilling pre-inception
history, and a cutoff at `primary`'s own start can never fall after a cell inside
its live range.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

import globalmacro.build as build_module
from globalmacro.build import coalesce_before_cutoff


def _frame(dates, primary, donor):
    return pl.DataFrame({"date": dates, "active": primary, "historic": donor})


def test_donor_backfills_strictly_before_primarys_first_observation():
    df = _frame(
        [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        [None, None, 1.0],
        [0.1, 0.2, 0.3],
    )
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    assert out["active"].to_list() == [0.1, 0.2, 1.0]


def test_donor_never_fills_a_null_inside_primarys_own_live_range():
    # 01-01 starts primary's live range; 01-03 is a null INSIDE it (the shape a
    # guard's deliberate null takes) -- the donor has a real value there, but must
    # not be used to fill it. A bare pl.coalesce would fill it with 0.3.
    df = _frame(
        [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        [1.0, 2.0, None],
        [0.1, 0.2, 0.3],
    )
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    assert out["active"].to_list() == [1.0, 2.0, None]


def test_the_cutoff_date_itself_keeps_primarys_own_value():
    # The cutoff is primary's OWN first valid date -- primary already has a value
    # there, so this pins date < cutoff (strict), not <=, as the boundary: getting
    # the comparison direction wrong would silently swap in the donor's value here
    # even though primary was never null on this date.
    df = _frame([date(2020, 1, 1)], [1.0], [99.0])
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    assert out["active"].to_list() == [1.0]


def test_primarys_own_non_null_values_always_win_regardless_of_position():
    df = _frame(
        [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)],
        [None, 2.0, None, 4.0],
        [0.1, 0.2, 0.3, 0.4],
    )
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    # 01-01 (pre-inception): donor fills. 01-02, 01-04: primary's own value, kept.
    # 01-03 (inside the live range, primary itself null): stays null.
    assert out["active"].to_list() == [0.1, 2.0, None, 4.0]


def test_an_entirely_null_primary_takes_the_full_donor_series():
    # No cutoff exists to compute -- nothing to protect yet, so this degrades to a
    # bare coalesce (donor fills every row), the same as before the fix.
    df = _frame(
        [date(2020, 1, 1), date(2020, 1, 2)],
        [None, None],
        [0.1, 0.2],
    )
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    assert out["active"].to_list() == [0.1, 0.2]


def test_a_donor_gap_before_the_cutoff_stays_null_not_zero():
    # The donor itself can be null pre-cutoff (its own data gap); this must not
    # manufacture a value out of nothing.
    df = _frame(
        [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        [None, None, 1.0],
        [None, 0.2, 0.3],
    )
    out = df.with_columns(coalesce_before_cutoff(df, "active", "historic").alias("active"))
    assert out["active"].to_list() == [None, 0.2, 1.0]


def test_a_splice_chain_carries_the_previous_steps_pre_inception_history_forward():
    # Mirrors the real RL <- TF <- RTY chain: TF is first backfilled by RL, and the
    # NEXT splice step (RTY <- TF) must see that already-spliced TF, not the
    # pre-splice one -- coalesce_before_cutoff recomputes first_valid_date on
    # whatever frame it is handed, not a snapshot taken before the chain started.
    df = pl.DataFrame({
        "date": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)],
        "rty": [None, None, 5.0],
        "tf": [None, None, None],
        "rl": [0.1, 0.2, 0.3],
    })
    df = df.with_columns(coalesce_before_cutoff(df, "tf", "rl").alias("tf")).drop("rl")
    df = df.with_columns(coalesce_before_cutoff(df, "rty", "tf").alias("rty")).drop("tf")
    # tf after step 1: [0.1, 0.2, 0.3] (rl backfills tf's own empty history in full).
    # rty's own first observation is 01-03, so tf backfills 01-01 and 01-02 only.
    assert df["rty"].to_list() == [0.1, 0.2, 5.0]


# ---------------------------------------------------------------------------
# The real SPLICING_MAP / USE_TRAD call sites in build_synced_dataset both have to
# route through coalesce_before_cutoff, not a bare pl.coalesce -- a source check
# alongside the pure-function tests above so a call site quietly reverting to the
# unguarded form fails a test even though coalesce_before_cutoff itself is untouched.
# ---------------------------------------------------------------------------


def test_the_splicing_map_and_trad_loops_use_the_cutoff_guarded_coalesce():
    text = Path(build_module.__file__).read_text()
    start = text.index("def build_synced_dataset(")
    end = text.index("\ndef ", start + 1)
    body = text[start:end]
    assert body.count("coalesce_before_cutoff(") == 2, (
        "expected exactly the SPLICING_MAP loop and the USE_TRAD/trad_ct loop to "
        "call coalesce_before_cutoff inside build_synced_dataset"
    )
    assert "pl.coalesce([pl.col(active), pl.col(historic)])" not in body
    assert "pl.coalesce([pl.col(symbol_ct), pl.col(symbol)])" not in body


# ---------------------------------------------------------------------------
# End-to-end through the real build_synced_dataset, on a minimal fixture tree --
# proves the wiring, not just the helper in isolation.
# ---------------------------------------------------------------------------


def _write_class_csv(root, tier, cls, rows):
    path = root / f"tier{tier}" / "sync" / f"{cls}_daily_returns.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(path)


_DATES = [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]


def _minimal_fixture_tree(datasets_root, data_root, *, traditional_extra=None, stir_extra=None,
                           nonus_equity_extra=None, us_equity_extra=None, currency_extra=None):
    """Every file/column build_synced_dataset(tier=1) unconditionally touches, filled
    null (a no-op through every keep_after_date/set_null_on_date call) except where a
    test overrides a column to set up its own scenario.
    """
    use_trad_cols = {s: [None] * len(_DATES) for s in ["I", "SI", "HG", "L", "AP", "GE", "TIFEY"]}
    _write_class_csv(datasets_root, 1, "traditional", {
        "date": _DATES, **(use_trad_cols | (traditional_extra or {}))
    })
    for cls in ["commodity", "bond", "volatility"]:
        _write_class_csv(datasets_root, 1, cls, {"date": _DATES})
    _write_class_csv(datasets_root, 1, "nonus_equity", {
        "date": _DATES,
        "FSMI": [None] * len(_DATES), "OMX": [None] * len(_DATES), "FTI": [None] * len(_DATES),
        **(nonus_equity_extra or {}),
    })
    _write_class_csv(datasets_root, 1, "us_equity", {
        "date": _DATES,
        "YM": [None] * len(_DATES), "EMD": [None] * len(_DATES), "DJ": [None] * len(_DATES),
        **(us_equity_extra or {}),
    })
    _write_class_csv(datasets_root, 1, "stir", {"date": _DATES, **(use_trad_cols | (stir_extra or {}))})
    _write_class_csv(datasets_root, 1, "currency", {"date": _DATES, **(currency_extra or {})})

    jkp_path = data_root / "jkp" / "updated_daily_ind_gics_synced.csv"
    jkp_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"date": _DATES, "gics": ["10"] * len(_DATES), "ret_vw_cap": [0.0] * len(_DATES)}).write_csv(jkp_path)


def test_build_synced_dataset_never_refills_a_live_range_null_from_the_historic_leg(tmp_path, monkeypatch):
    datasets_root, data_root = tmp_path / "datasets", tmp_path / "data"
    # 6E (active) starts 01-02; 01-04 is null INSIDE its own live range -- the shape
    # a guard's deliberate null takes. DM (historic) is non-null on every date,
    # including 01-04, and disagrees with what 6E holds elsewhere -- a real historic
    # value the fix must not let leak into 6E's own live range.
    _minimal_fixture_tree(datasets_root, data_root, currency_extra={
        "6E": [None, 0.01, 0.02, None],
        "DM": [0.10, 0.11, 0.12, 0.13],
    })
    monkeypatch.setattr(build_module, "DATASETS_ROOT", datasets_root)
    monkeypatch.setattr(build_module, "DATA_ROOT", data_root)

    out = build_module.build_synced_dataset(tier=1)
    row = out.filter(pl.col("date").is_in(_DATES)).sort("date")
    assert row["6E"].to_list() == [0.10, 0.01, 0.02, None]
    assert "DM" not in out.columns


def test_trad_ct_loop_never_refills_a_live_range_null_either(tmp_path, monkeypatch):
    # Same shape, the OTHER coalesce site: GE_ct (traditional) is primary, GE
    # (stir) is the donor -- 01-04 is a live-range null in GE_ct that the stir leg
    # must not be allowed to fill.
    datasets_root, data_root = tmp_path / "datasets", tmp_path / "data"
    _minimal_fixture_tree(
        datasets_root, data_root,
        traditional_extra={"GE": [None, 0.01, 0.02, None]},
        stir_extra={"GE": [0.10, 0.11, 0.12, 0.13]},
    )
    monkeypatch.setattr(build_module, "DATASETS_ROOT", datasets_root)
    monkeypatch.setattr(build_module, "DATA_ROOT", data_root)

    out = build_module.build_synced_dataset(tier=1)
    row = out.filter(pl.col("date").is_in(_DATES)).sort("date")
    assert row["GE"].to_list() == [0.10, 0.01, 0.02, None]
