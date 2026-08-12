from datetime import date

import polars as pl
import pytest

import globalmacro.build as build
from globalmacro.build import (
    GICS_SECTOR_TICKERS,
    load_sync_observation_panel,
    save_usd_datasets,
)
from globalmacro.pipeline.futures import apply_bad_slot2_print_guard
from globalmacro.pipeline.tickhistory import (
    apply_stale_relabel_repair,
    observation_flag,
    repair_stale_relabel,
)
from globalmacro.pipeline.to_usd import usd_panel
from globalmacro.utils.models import AssetClass


def test_observation_is_front_month_settlement_presence_not_return_nullity():
    # a row priced but with a null return -- the exact cell the nullity rule misses -- is observed
    df = pl.DataFrame({"front_month_settlement": [100.0, 110.0, None],
                       "ret1_adjusted": [None, None, None]})
    assert observation_flag(df).to_list() == [True, True, False]


def _write_observation_panel(tmp_path, rows):
    obs = tmp_path / "observations"
    obs.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(obs / "tier1_commodity_et.parquet")


def test_sync_observation_panel_pivots_symbol_by_date(tmp_path, monkeypatch):
    _write_observation_panel(tmp_path, {
        "symbol": ["BRN", "BRN"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, False],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)
    panel = load_sync_observation_panel(1, "et", ["BRN"]).sort("date")
    assert "BRN" in panel.columns and "date" in panel.columns
    assert panel["BRN"].to_list() == [True, False]


def test_zero_coverage_raises_not_silently_degrades(tmp_path, monkeypatch):
    (tmp_path / "observations").mkdir()
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)
    with pytest.raises(ValueError, match="no observation coverage"):
        load_sync_observation_panel(1, "et", ["NOTREAL"])


def test_partial_coverage_hole_raises_not_only_zero_coverage(tmp_path, monkeypatch):
    # BB has a panel, XT does not -- some coverage, but not full. The old guard only
    # fired at zero coverage, so a hole like this shipped silently.
    _write_observation_panel(tmp_path, {
        "symbol": ["BB", "BB"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)
    with pytest.raises(ValueError, match="XT"):
        load_sync_observation_panel(1, "et", ["BB", "XT"])


def test_uncovered_jkp_sector_ticker_does_not_raise(tmp_path, monkeypatch):
    # JKP sector tickers carry no tick data by construction -- the one coverage hole
    # that is expected, not a bug -- so the strengthened guard must not trip on it.
    sector_ticker = next(iter(GICS_SECTOR_TICKERS.values()))
    _write_observation_panel(tmp_path, {
        "symbol": ["BB", "BB"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)
    panel = load_sync_observation_panel(1, "et", ["BB", sector_ticker])
    assert "BB" in panel.columns
    assert sector_ticker not in panel.columns


def test_tier1_sync_observation_panel_ignores_tier2_parquets(tmp_path, monkeypatch):
    # A wider glob filtered by `wanted` must stay behavior-preserving for tier 1: its
    # requested symbols only ever come from tier1_*.parquet, so pulling in a tier-2
    # parquet -- even one that happens to name the same symbol -- must not change the
    # tier-1 result. This tier-2 file carries the OPPOSITE observed values for BRN, so
    # if tier 1 ever globbed it in, the .any() union would flip [True, False] to
    # [True, True].
    _write_observation_panel(tmp_path, {
        "symbol": ["BRN", "BRN"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, False],
    })
    obs = tmp_path / "observations"
    pl.DataFrame({
        "symbol": ["BRN", "BRN"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [False, True],
    }).write_parquet(obs / "tier2_equity_et.parquet")
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    panel = load_sync_observation_panel(1, "et", ["BRN"]).sort("date")
    assert panel["BRN"].to_list() == [True, False]


def test_tier2_sync_observation_panel_includes_tier1_inherited_symbols(tmp_path, monkeypatch):
    # tier 2's sync panel is assembled from every tier-1 class CSV plus two tier-2-only
    # extras (build_synced_dataset), so a tier-1-inherited symbol like BB must be
    # covered by a tier-2 request even though its own observation panel was written
    # under the tier-1 glob (tier1_currency_et.parquet), not a tier2_* one.
    obs = tmp_path / "observations"
    obs.mkdir()
    pl.DataFrame({
        "symbol": ["BB", "BB"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, False],
    }).write_parquet(obs / "tier1_currency_et.parquet")
    pl.DataFrame({
        "symbol": ["BXF", "BXF"],
        "date": [date(2020, 1, 1), date(2020, 1, 2)],
        "observed": [True, True],
    }).write_parquet(obs / "tier2_equity_et.parquet")
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    panel = load_sync_observation_panel(2, "et", ["BB", "BXF"]).sort("date")
    assert "BB" in panel.columns
    assert "BXF" in panel.columns
    assert panel["BB"].to_list() == [True, False]


# ---------------------------------------------------------------------------
# splice_cutoffs: folding a SPLICING_MAP donor's own observation rows into its
# active's mask, gated to strictly before that active's splice cutoff.
# ---------------------------------------------------------------------------


def test_donor_observation_folds_into_active_mask_strictly_before_the_splice_cutoff(tmp_path, monkeypatch):
    # FGBL<-BDL: BDL kept trading in parallel with FGBL for a while past FGBL's own
    # splice cutoff (a real overlap window, not a hypothetical), so a date AT the
    # cutoff must be excluded -- only strictly-before dates are ones the value
    # assembly (coalesce_before_cutoff) ever actually draws BDL's return from.
    cutoff = date(2020, 1, 3)
    _write_observation_panel(tmp_path, {
        "symbol": ["BDL", "BDL", "FGBL"],
        "date": [date(2020, 1, 1), cutoff, date(2020, 1, 5)],
        "observed": [True, True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    panel = load_sync_observation_panel(1, "et", ["FGBL"], splice_cutoffs={"FGBL": cutoff}).sort("date")
    rows = dict(zip(panel["date"].to_list(), panel["FGBL"].to_list(), strict=True))
    assert rows[date(2020, 1, 1)] is True, "BDL's pre-cutoff observation must fold into FGBL's mask"
    assert rows.get(cutoff) is not True, "BDL's AT-cutoff observation must not leak in (strictly before)"
    assert rows[date(2020, 1, 5)] is True, "FGBL's own observation must still come through unaffected"


def test_no_splice_cutoffs_argument_leaves_todays_behaviour_untouched(tmp_path, monkeypatch):
    # Same fixture as above, but the caller never passes splice_cutoffs -- the
    # donor must never be scanned at all, matching every call site before this fix.
    cutoff = date(2020, 1, 3)
    _write_observation_panel(tmp_path, {
        "symbol": ["BDL", "BDL", "FGBL"],
        "date": [date(2020, 1, 1), cutoff, date(2020, 1, 5)],
        "observed": [True, True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    panel = load_sync_observation_panel(1, "et", ["FGBL"]).sort("date")
    rows = dict(zip(panel["date"].to_list(), panel["FGBL"].to_list(), strict=True))
    assert date(2020, 1, 1) not in rows, "with no splice_cutoffs, BDL must never be folded in"
    assert rows[date(2020, 1, 5)] is True


def test_a_splice_chain_folds_every_hop_gated_by_that_hops_own_cutoff(tmp_path, monkeypatch):
    # RTY<-TF<-RL. Requesting only RTY must still pick up TF's own observations
    # (gated by cutoff[RTY]) AND, transitively, RL's (gated by cutoff[TF]) --
    # closing the chain, not just the first hop.
    cutoff_rty = date(2020, 1, 10)
    cutoff_tf = date(2020, 1, 3)
    _write_observation_panel(tmp_path, {
        "symbol": ["TF", "RL"],
        "date": [date(2020, 1, 5), date(2020, 1, 1)],
        "observed": [True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    panel = load_sync_observation_panel(
        1, "et", ["RTY"], splice_cutoffs={"RTY": cutoff_rty, "TF": cutoff_tf}
    ).sort("date")
    rows = dict(zip(panel["date"].to_list(), panel["RTY"].to_list(), strict=True))
    assert rows[date(2020, 1, 5)] is True, "TF's observation (< cutoff[RTY]) must fold into RTY"
    assert rows[date(2020, 1, 1)] is True, "RL's observation (< cutoff[TF]) must fold into RTY transitively"


def test_sync_daily_usd_wiring_narrows_fx_leg_around_a_masked_observation(tmp_path, monkeypatch):
    # BRN's return is null on the middle day -- exactly the cell the provenance guard can
    # null while the front month still priced -- but the tick pipeline still marks it
    # observed. Without the mask, the following day's FX leg would widen across the gap.
    days = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4), date(2020, 1, 5)]
    _write_observation_panel(tmp_path, {
        "symbol": ["BRN"] * 4,
        "date": days,
        "observed": [True, True, True, True],
    })
    monkeypatch.setattr(build, "TICKHISTORY_PATH", tmp_path)

    local = pl.DataFrame({"date": days, "BRN": [0.01, None, 0.02, 0.03]})
    fx = pl.DataFrame({
        "date": [date(2020, 1, 1), *days],
        "GBP": [1.00, 1.02, 1.01, 1.05, 1.00],
    })
    symbol_to_ccy = {"BRN": "GBP"}

    panel = load_sync_observation_panel(1, "et", ["BRN"])
    masked = usd_panel(local, symbol_to_ccy, fx, traded=panel)
    unmasked = usd_panel(local, symbol_to_ccy, fx, traded=None)
    masked_leg = masked.filter(pl.col("date") == date(2020, 1, 4))["BRN"].item()
    unmasked_leg = unmasked.filter(pl.col("date") == date(2020, 1, 4))["BRN"].item()
    assert masked_leg != pytest.approx(unmasked_leg)

    out_root = tmp_path / "out"
    save_usd_datasets(
        tier1_synced=local, tier1_asynced=local,
        tier2_synced=None, tier2_asynced=local,
        symbol_to_ccy=symbol_to_ccy, fx_async=fx, fx_sync=fx,
        out_root=out_root,
    )
    written = pl.read_csv(out_root / "tier1" / "sync" / "sync_daily_usd.csv", try_parse_dates=True)
    written_leg = written.filter(pl.col("date") == date(2020, 1, 4))["BRN"].item()
    assert written_leg == pytest.approx(masked_leg)
    assert written_leg != pytest.approx(unmasked_leg)


def _bad_slot2_frame(vol2, rt2, vol1, rt1, exp1, ret1):
    return pl.DataFrame({"clscode": [1] * len(vol2), "volume_2": vol2, "ret_temp_2": rt2,
                         "volume_1": vol1, "ret_temp_1": rt1, "exp_1": exp1, "ret_1": ret1})


def test_substitutes_the_front_month_on_both_legs_of_a_zero_volume_bad_print():
    # t: +0.336 at zero volume; t+1: -0.321 revert, the same bad print now sitting in the
    # denominator (volume_2 = 5 that day)
    df = _bad_slot2_frame([0, 5], [0.336, -0.321], [11238, 200], [-0.064, -0.034], [1, 1], [0.336, -0.321])
    out = apply_bad_slot2_print_guard(df)
    assert out["ret_1"].to_list() == [-0.064, -0.034]


def test_does_not_touch_a_large_move_that_actually_traded():
    df = _bad_slot2_frame([120], [0.336], [11238], [-0.064], [1], [0.336])
    assert apply_bad_slot2_print_guard(df)["ret_1"].to_list() == [0.336]


def test_nulls_when_no_traded_front_month_exists():
    df = _bad_slot2_frame([0], [0.336], [0], [-0.064], [1], [0.336])
    assert apply_bad_slot2_print_guard(df)["ret_1"].to_list() == [None]


def test_exp_1_term_in_fire_keeps_a_poisoned_non_roll_cell_untouched():
    # t: a genuine bad print -- fires, substituted. t+1: exp_1 == 0, so it is not on the
    # roll at all, even though it is poisoned via bad.shift(1); only the exp_1 == 1 term
    # inside fire keeps this cell from being touched too.
    df = _bad_slot2_frame([0, 999], [0.35, 0.10], [100, 50], [0.01, -0.10], [1, 0], [0.35, 0.05])
    assert apply_bad_slot2_print_guard(df)["ret_1"].to_list() == [0.01, 0.05]


def test_bad_print_does_not_leak_across_clscode_into_the_next_symbols_first_row():
    # clscode 1's only row is a bad print; clscode 2's first row would look poisoned by
    # bad.shift(1) if that shift ever crossed the partition boundary. It must not: the
    # frame is windowed over("clscode"), so clscode 2's first row sees no prior row at
    # all and ships untouched.
    df = pl.DataFrame({
        "clscode": [1, 2],
        "volume_2": [0, 999],
        "ret_temp_2": [0.35, 0.05],
        "volume_1": [100, 50],
        "ret_temp_1": [0.01, -0.10],
        "exp_1": [1, 1],
        "ret_1": [0.35, 0.05],
    })
    assert apply_bad_slot2_print_guard(df)["ret_1"].to_list() == [0.01, 0.05]


def test_leaves_a_pre_roll_settlement_spike_and_its_revert_alone():
    # t: zero-volume slot-2 print moves +38.5% while exp_1 == 0 -- a settlement spike on the
    # day before the roll, not the roll-day defect this guard targets. t+1: exp_1 == 1, and
    # its own ret_temp_2 is a <30% revert, so it doesn't independently look bad -- the old
    # discriminator only poisons it via bad.shift(1) from t. Both cells should ship untouched:
    # nulling t+1 while leaving t's spike stranded would inject a one-sided error.
    df = _bad_slot2_frame([0, 0], [0.3848, -0.2855], [100, 0], [0.30, -0.29], [0, 1], [0.3848, -0.2855])
    assert apply_bad_slot2_print_guard(df)["ret_1"].to_list() == [0.3848, -0.2855]


def test_repairs_a_stale_c1_relabel_to_ret_c2():
    # settlement_c1 on the relabel day is a one-day carry-forward of c1's last real
    # print (c1 did not trade that day, and the price goes dark again the next row);
    # settlement_c2 traded throughout, so ret_c2 is the value a live relabel would
    # have produced instead of the stale-print return the panel shipped
    df = pl.DataFrame({
        "shift": [0, -1, 0],
        "settlement_c1": [17.71, 17.76, None],
        "settlement_c2": [17.16, 16.855, 17.26],
        "c1_traded": [True, False, False],
        "c2_traded": [True, True, True],
        "ret_c2": [None, 16.855 / 17.16 - 1, None],
        "ret1_adjusted": [None, 17.76 / 17.16 - 1, None],
    })
    out = repair_stale_relabel(df)
    assert abs(out["ret1_adjusted"][1] - (16.855 / 17.16 - 1)) < 1e-9


def test_settlement_c1_persisting_next_row_does_not_fire_the_stale_relabel_repair():
    # settlement_c1 is still non-null the row after the relabel -- a persisting print,
    # not a one-day carry-forward that ran out -- so even though c1 recorded no trade
    # and slot 2 traded with a present ret_c2, this must not read as the stale-print
    # shape and must ship the original relabel return untouched.
    df = pl.DataFrame({
        "shift": [0, -1, 0],
        "settlement_c1": [17.71, 17.76, 17.80],
        "settlement_c2": [17.16, 16.855, 17.26],
        "c1_traded": [True, False, False],
        "c2_traded": [True, True, True],
        "ret_c2": [None, 16.855 / 17.16 - 1, None],
        "ret1_adjusted": [None, 17.76 / 17.16 - 1, None],
    })
    out = repair_stale_relabel(df)
    assert out["ret1_adjusted"][1] == pytest.approx(17.76 / 17.16 - 1)


def test_last_row_relabel_still_fires_on_a_genuine_stale_print():
    # The recoverable frame ends on the fire row, so settlement_c1.shift(-1) is null
    # purely because there is no next row to read -- not because a real print goes dark
    # tomorrow. Locks in the current, intended behavior: this still fires and still
    # substitutes ret_c2, because c1_traded is False independently of that shift -- the
    # RIC recorded zero trades that day regardless of what tomorrow's data would show,
    # so the print is known-stale on its own, not merely inferred stale from the edge.
    df = pl.DataFrame({
        "shift": [0, -1],
        "settlement_c1": [17.71, 17.76],
        "settlement_c2": [17.16, 16.855],
        "c1_traded": [True, False],
        "c2_traded": [True, True],
        "ret_c2": [None, 16.855 / 17.16 - 1],
        "ret1_adjusted": [None, 17.76 / 17.16 - 1],
    })
    out = repair_stale_relabel(df)
    assert out["ret1_adjusted"][1] == pytest.approx(16.855 / 17.16 - 1)


def test_leaves_a_live_relabel_untouched():
    df = pl.DataFrame({"shift": [-1], "settlement_c1": [17.76], "settlement_c2": [16.855],
                       "c1_traded": [True], "c2_traded": [True], "ret_c2": [-0.0178],
                       "ret1_adjusted": [0.034965]})
    assert abs(repair_stale_relabel(df)["ret1_adjusted"][0] - 0.034965) < 1e-9


def test_does_not_recompute_when_c2_is_itself_untraded():
    # c1 also reads untraded here, but with no trade on either leg there is no live
    # contract to recompute the relabel against, so the shipped value stays
    df = pl.DataFrame({"shift": [-1], "settlement_c1": [17.76], "settlement_c2": [16.855],
                       "c1_traded": [False], "c2_traded": [False], "ret_c2": [-0.0178],
                       "ret1_adjusted": [0.034965]})
    assert abs(repair_stale_relabel(df)["ret1_adjusted"][0] - 0.034965) < 1e-9


def test_stale_relabel_repair_is_gated_out_on_the_traditional_branch():
    # front_month == 3: the traditional branch's shift == -1 roll here is
    # rollback_c4_to_c3 (settlement_c3 over settlement_c4), a different contract pair
    # than ret_c2 (settlement_c2 over settlement_c2) -- the stale-c1 signature
    # repair_stale_relabel looks for is present on this row too, so an unguarded
    # repair would overwrite the correct rollback with an unrelated contract's return.
    df = pl.DataFrame({
        "date_": [date(1996, 12, 20), date(1996, 12, 23), date(1996, 12, 24)],
        "shift": [0, -1, 0],
        "front_month": [3, 3, 1],
        "settlement_c1": [3895.6, 3895.6, None],
        "ret_c2": [None, -0.0035842293906810374, None],
        "ret1_adjusted": [None, -0.015505846466700612, None],  # the shipped rollback_c4_to_c3
    })
    counts = pl.DataFrame(
        {"ric": ["SMIc2"], "date_": [date(1996, 12, 23)]},
        schema={"ric": pl.Utf8, "date_": pl.Date},
    )

    traditional = apply_stale_relabel_repair(df, "SMI", counts, AssetClass.TRADITIONAL)
    assert traditional["ret1_adjusted"][1] == pytest.approx(-0.015505846466700612)

    # Same frame and counts, a non-traditional asset class: confirms the fire
    # condition really is satisfied here, so the traditional assertion above is an
    # actual gate rather than a coincidence of these inputs.
    non_traditional = apply_stale_relabel_repair(df, "SMI", counts, AssetClass.COMMODITY)
    assert non_traditional["ret1_adjusted"][1] == pytest.approx(-0.0035842293906810374)
