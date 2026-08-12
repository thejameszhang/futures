from datetime import date

import polars as pl
import pytest

from globalmacro.pipeline.tickhistory import (
    apply_provenance_guard,
    attach_front_slot,
    attach_traditional_front_slot,
    build_config,
    clscodes_for,
    ladder_expiries_for,
    rescue_ret1_adjusted,
)
from globalmacro.utils import trade_presence
from globalmacro.utils.models import AssetClass, Future
from globalmacro.utils.trade_presence import (
    configured_rics,
    count_trades_in_frame,
    is_rescuable_mask,
)
from scripts import build_trade_presence


def test_front_slot_records_which_slot_supplied_the_price():
    # Row 4 carries a null expiring_this_month (a join miss upstream) -- the guard's
    # `!=` comparison on front_slot is only safe because front_slot is never null on
    # either producer, so that property is pinned here alongside the traditional
    # producer's own null_count() == 0 assertion.
    df = pl.DataFrame({
        "expiring_this_month": [0, 1, 0, None],
        "settlement_c1": [100.0, 200.0, None, 400.0],
        "settlement_c2": [101.0, 201.0, 301.0, 401.0],
    })
    out = attach_front_slot(df)
    assert out["front_month_settlement"].to_list() == [100.0, 201.0, 301.0, 400.0]
    assert out["front_slot"].to_list() == [1, 2, 2, 1]
    assert out["front_slot"].null_count() == 0


def test_front_slot_is_not_recoverable_from_value_when_slots_are_equal():
    df = pl.DataFrame({
        "expiring_this_month": [0, 1],
        "settlement_c1": [100.0, 100.0],
        "settlement_c2": [100.0, 100.0],
    })
    assert attach_front_slot(df)["front_slot"].to_list() == [1, 2]


def test_no_c1_fallback_in_the_expiry_month():
    # In the expiry month (expiring_this_month == 1) the front price is always c2 --
    # a missing c2 leaves the price null rather than falling back to c1, and the slot
    # still records 2 (the slot the selection targeted, not proof a price exists).
    df = pl.DataFrame({"expiring_this_month": [1], "settlement_c1": [100.0], "settlement_c2": [None]})
    out = attach_front_slot(df)
    assert out["front_month_settlement"].to_list() == [None]
    assert out["front_slot"].to_list() == [2]


# ---------------------------------------------------------------------------
# attach_traditional_front_slot: exhaustive over the reachable front_month domain.
# front_month is expected to land in 1..4, but nothing upstream guarantees it, so the
# fallback branch (null, 0, negatives, and out-of-range values) is exercised alongside
# the four real slots -- both a mislabeled boundary (e.g. rejecting front_month == 4)
# and an always-1 slot rule would still pass a test that only checked 1..4.
# ---------------------------------------------------------------------------

_FRONT_MONTH_DOMAIN = [None, -2, -1, 0, 1, 2, 3, 4, 5, 6, 100]


def test_traditional_front_slot_matches_the_selected_price_over_the_whole_domain():
    n = len(_FRONT_MONTH_DOMAIN)
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [1.0] * n,
        "settlement_c2": [2.0] * n,
        "settlement_c3": [3.0] * n,
        "settlement_c4": [4.0] * n,
    })
    out = attach_traditional_front_slot(df)
    for front_month, price, slot in zip(
        out["front_month"].to_list(), out["front_month_settlement"].to_list(), out["front_slot"].to_list(),
        strict=True,
    ):
        expected_slot = front_month if front_month in (1, 2, 3, 4) else 1
        assert slot == expected_slot, (front_month, price, slot)
        assert price == float(expected_slot), (front_month, price, slot)


def test_traditional_front_slot_matches_the_selected_price_when_some_settlements_are_null():
    n = len(_FRONT_MONTH_DOMAIN)
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [None] * n,
        "settlement_c2": [2.0] * n,
        "settlement_c3": [None] * n,
        "settlement_c4": [4.0] * n,
    })
    out = attach_traditional_front_slot(df)
    slot_to_col = {1: "settlement_c1", 2: "settlement_c2", 3: "settlement_c3", 4: "settlement_c4"}
    for row in out.iter_rows(named=True):
        expected_slot = row["front_month"] if row["front_month"] in (1, 2, 3, 4) else 1
        assert row["front_slot"] == expected_slot, row
        assert row["front_month_settlement"] == row[slot_to_col[expected_slot]], row


def test_traditional_front_slot_domain_is_exactly_one_through_four():
    df = pl.DataFrame({
        "front_month": _FRONT_MONTH_DOMAIN,
        "settlement_c1": [1.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c2": [2.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c3": [3.0] * len(_FRONT_MONTH_DOMAIN),
        "settlement_c4": [4.0] * len(_FRONT_MONTH_DOMAIN),
    })
    out = attach_traditional_front_slot(df)
    assert set(out["front_slot"].to_list()) == {1, 2, 3, 4}
    assert out["front_slot"].null_count() == 0


# ---------------------------------------------------------------------------
# apply_provenance_guard: null a finalised ret1_adjusted cell whenever its two ends
# came from different contract slots, scoped so legitimate rolls survive. Unscoped,
# this mask nulls the sync panels' roll returns wholesale -- rollback_c2_to_c1
# deliberately divides settlement_c1(t) by settlement_c2(t-1), so disagreeing slots
# are its normal condition, not a defect. Each clause in the mask gets its own test
# below that fails when that clause alone is removed, not just a test the whole
# guard happens to pass -- a clause nothing distinguishes from "always true" is
# indistinguishable from a clause that was never checked in.
# ---------------------------------------------------------------------------


def _guard_frame(shift, slots, ret1, ret1_adjusted=None):
    return pl.DataFrame({
        "shift": shift,
        "front_slot": slots,
        "ret1": ret1,
        "ret1_adjusted": ret1 if ret1_adjusted is None else ret1_adjusted,
    })


def test_nulls_a_return_whose_ends_came_from_different_slots():
    out = apply_provenance_guard(_guard_frame([0, 0], [1, 2], [0.01, 0.02]))
    assert out["ret1_adjusted"].to_list() == [0.01, None]


def test_leaves_a_backward_roll_untouched():
    out = apply_provenance_guard(_guard_frame([0, -1], [1, 2], [0.01, 0.02]))
    assert out["ret1_adjusted"].to_list() == [0.01, 0.02]


def test_leaves_a_forward_roll_untouched():
    out = apply_provenance_guard(_guard_frame([0, 1], [1, 2], [0.01, 0.02]))
    assert out["ret1_adjusted"].to_list() == [0.01, 0.02]


def test_ignores_rows_where_ret1_was_not_the_shipped_value():
    out = apply_provenance_guard(_guard_frame([0, 0], [1, 2], [0.01, None]))
    assert out["ret1_adjusted"].to_list() == [0.01, None]


def test_leaves_a_same_slot_return_untouched():
    # front_slot agrees at both ends and this isn't a roll, so there is nothing
    # cross-contract here. Drop the front_slot comparison from the mask and only
    # "not a roll, ret1 non-null" survive, which would null this row too.
    out = apply_provenance_guard(_guard_frame([0, 0], [1, 1], [0.01, 0.02]))
    assert out["ret1_adjusted"].to_list() == [0.01, 0.02]


def test_a_backfilled_return_survives_when_ret1_itself_was_never_shipped():
    # A synthetic shape, not one real data produces -- the rows that look like this
    # (front_slot moved, not a roll, ret1 null) are already excluded by the shift
    # scope or the slot comparison before this clause ever runs. It exists to pin the
    # ret1-not-null clause's precondition directly: front_slot only describes the
    # price sitting in ret1_adjusted when ret1_adjusted is ret1 itself. Drop the
    # clause and this row would be nulled even though its two ends never disagreed.
    frame = _guard_frame([0, 0], [1, 2], [0.01, None], ret1_adjusted=[0.01, 0.05])
    out = apply_provenance_guard(frame)
    assert out["ret1_adjusted"].to_list() == [0.01, 0.05]


def test_guard_is_idempotent():
    # apply_provenance_guard is called once per symbol, but nothing about its own
    # code path relies on that -- pin the fixed point. Two consecutively flagged
    # rows (front_slot moves at both) are needed to discriminate this from a guard
    # that substitutes a neighbouring value instead of nulling outright: with only
    # one flagged row, that kind of mutant would already be a fixed point too, since
    # its untouched neighbour never changes between passes.
    frame = _guard_frame([0, 0, 0], [1, 2, 3], [0.01, 0.02, 0.03])
    once = apply_provenance_guard(frame)
    twice = apply_provenance_guard(once)
    assert twice["ret1_adjusted"].to_list() == once["ret1_adjusted"].to_list()


def test_guard_preserves_column_order_and_dtypes():
    # Column order matters downstream: the debug CSV's header, and the code that
    # reads it back, both depend on ret1_adjusted being replaced in place rather
    # than dropped and re-appended.
    frame = _guard_frame([0, 0], [1, 2], [0.01, 0.02])
    out = apply_provenance_guard(frame)
    assert out.columns == frame.columns
    assert out.schema == frame.schema


# ---------------------------------------------------------------------------
# count_trades_in_frame: per-RIC per-day trade counts from raw tick rows. Every
# fixture uses shard-shaped `Date-Time` strings (never a naive/already-parsed
# datetime) -- an unparsed-string bug in the real implementation would still pass a
# test built on a pre-parsed fixture, and the SLURM job would die on the first shard.
# ---------------------------------------------------------------------------


def _ticks(rows):
    """rows: (ric, date_time_string, price, volume)."""
    return pl.DataFrame(
        {
            "#RIC": [r[0] for r in rows],
            "Date-Time": [r[1] for r in rows],
            "Price": [r[2] for r in rows],
            "Volume": [r[3] for r in rows],
            "GMT Offset": [0] * len(rows),
        },
        schema_overrides={"Date-Time": pl.String},
    )


def test_counts_group_by_ric_and_day_in_the_target_timezone():
    out = count_trades_in_frame(_ticks([
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, 5),
        ("CLc1", "2020-01-02T14:45:00.000000000Z", 11.0, 3),
        ("CLc2", "2020-01-02T14:31:00.000000000Z", 12.0, 1),
    ]), "et").sort("ric")
    assert out["ric"].to_list() == ["CLc1", "CLc2"]
    assert out["n_trades"].to_list() == [2, 1]


def test_the_day_boundary_follows_the_sync_target():
    # 02:30 UTC is the 2nd in London and the 1st (21:30) in ET. Verified.
    t = _ticks([("CLc1", "2020-01-02T02:30:00.000000000Z", 10.0, 1)])
    assert count_trades_in_frame(t, "london")["date_"].to_list() == [date(2020, 1, 2)]
    assert count_trades_in_frame(t, "et")["date_"].to_list() == [date(2020, 1, 1)]


def test_a_non_z_offset_is_parsed_not_assumed():
    # +05:00 puts this tick at 2020-01-03T01:00Z = 2020-01-02 20:00 ET. Discarding
    # the offset and stamping UTC gives 2020-01-03. Verified both ways.
    out = count_trades_in_frame(
        _ticks([("CLc1", "2020-01-03T06:00:00.000000000+05:00", 10.0, 1)]), "et"
    )
    assert out["date_"].to_list() == [date(2020, 1, 2)]


def test_zero_price_prints_are_not_trades():
    # load_trades_data nulls Price == 0.0 and drops the row; this must agree.
    out = count_trades_in_frame(_ticks([
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 0.0, 1),
        ("CLc1", "2020-01-02T14:45:00.000000000Z", 11.0, 1),
    ]), "et")
    assert out["n_trades"].to_list() == [1]


def test_duplicate_ticks_are_counted_once():
    # load_trades_data de-duplicates before counting; a >=50% trade-count collapse
    # threshold elsewhere is calibrated against de-duplicated counts.
    out = count_trades_in_frame(_ticks([
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, 5),
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, 5),
    ]), "et")
    assert out["n_trades"].to_list() == [1]


def test_a_null_volume_falls_back_to_one_and_collapses_with_an_explicit_one():
    # Null Volume is common in the real shards. The fallback maps it to 1.0 before
    # dedup runs, so a null-Volume print and an explicit Volume=1 print of the same
    # trade collapse into one row instead of double-counting a single trade.
    out = count_trades_in_frame(_ticks([
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, None),
        ("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, 1),
    ]), "et")
    assert out["n_trades"].to_list() == [1]


def test_a_string_gmt_offset_is_cast_before_dedup():
    # Some shard directories carry GMT Offset as a String column ("+2", "-4", ...)
    # rather than Int64. The cast to Float32 has to run before dedup, so two prints
    # of the same offset written in different string forms still collapse the way an
    # Int64 shard's rows would; without the cast they'd compare unequal as strings.
    out = count_trades_in_frame(pl.DataFrame(
        {
            "#RIC": ["CLc1", "CLc1"],
            "Date-Time": ["2020-01-02T14:31:00.000000000Z"] * 2,
            "Price": [10.0, 10.0],
            "Volume": [1, 1],
            "GMT Offset": ["+2", "2"],
        },
        schema_overrides={"Date-Time": pl.String, "GMT Offset": pl.String},
    ), "et")
    assert out["n_trades"].to_list() == [1]


def test_a_dark_contract_has_no_row_not_a_zero():
    out = count_trades_in_frame(
        _ticks([("CLc1", "2020-01-02T14:31:00.000000000Z", 10.0, 1)]), "et"
    )
    assert "CLc2" not in out["ric"].to_list()


def test_output_order_is_deterministic():
    # group_by does not guarantee row order, so two runs over identical input can
    # otherwise emit the same rows in a different physical order -- a different
    # artifact on disk despite agreeing on every value. A handful of rows fed out of
    # order catches a missing sort only some of the time, because a small group_by
    # sometimes happens to emit rows in something close to sorted order anyway; 16
    # rows cycling through 4 RICs in strict reverse (ric, date) order gives group_by's
    # own emission order much more room to disagree with a missing sort, so the
    # mismatch shows up reliably instead of intermittently.
    rows = [
        (f"CLc{4 - i % 4}", f"2020-01-{28 - i:02d}T14:31:00.000000000Z", 10.0, 1)
        for i in range(16)
    ]
    out = count_trades_in_frame(_ticks(rows), "et")
    result_rows = out.select("ric", "date_").rows()
    assert result_rows == sorted(result_rows)


# ---------------------------------------------------------------------------
# configured_rics: the c1..c4 RICs one (tier, asset class) triple's config actually
# names, mirroring the CONFIG/ALL_RICS construction the ingestion stage builds for
# itself. A minimal synthetic tier config exercises the fields that matter (asset
# class match, an absent `ric`, `traditional`'s extra `ct` restriction, a two-RIC
# future) without depending on the real tier1.yaml/tier2.yaml staying unchanged.
# ---------------------------------------------------------------------------

_FUTURE_TEMPLATE_FIELDS = [
    "contrcode: 1",
    "exchange: 1",
    "exchange_name: Test Exchange",
    "clscode: 1",
    "calcseriesname: TEST SERIES",
    "name: Test Future",
]


def _write_tier_config(tmp_path, entries: dict[str, str]):
    """`entries` maps a symbol to extra YAML lines (e.g. `asset_class`, `ric`, `ct`)
    appended after the required fields every `Future` needs -- indented as a mapping
    under that symbol's key, the way `tier1.yaml`/`tier2.yaml` themselves are shaped.
    """
    blocks = []
    for symbol, extra in entries.items():
        fields = _FUTURE_TEMPLATE_FIELDS + extra.strip("\n").split("\n")
        blocks.append(f"{symbol}:\n" + "\n".join(f"  {line}" for line in fields))
    path = tmp_path / "tier1.yaml"
    path.write_text("\n".join(blocks) + "\n")
    return path


def test_configured_rics_expands_c1_through_c4_for_a_matching_future(tmp_path):
    path = _write_tier_config(tmp_path, {
        "CL": "asset_class: commodity\nric: CLX\n",
    })
    assert configured_rics(1, "commodity", config_path=path) == {"CLXc1", "CLXc2", "CLXc3", "CLXc4"}


def test_configured_rics_skips_a_future_with_no_ric(tmp_path):
    # Some configured commodity/equity instruments carry no `ric` at all; the ingestion
    # stage's own CONFIG builder never adds them (`future.ric is not None` guards it),
    # so this must not raise and must not synthesize RICs out of the symbol.
    path = _write_tier_config(tmp_path, {"CL": "asset_class: commodity\n"})
    assert configured_rics(1, "commodity", config_path=path) == set()


def test_configured_rics_only_matches_the_requested_asset_class(tmp_path):
    path = _write_tier_config(tmp_path, {"CL": "asset_class: bond\nric: CLX\n"})
    assert configured_rics(1, "commodity", config_path=path) == set()


def test_configured_rics_for_traditional_requires_a_trading_cycle(tmp_path):
    # traditional gets one restriction on top of the asset-class match: the pipeline's
    # own CONFIG builder drops any future whose `ct` (trading-cycle months) is unset,
    # even if it is otherwise a traditional-class future with a RIC.
    path = _write_tier_config(tmp_path, {
        "CL": "asset_class: traditional\nric: CLX\n",
        "GC": "asset_class: traditional\nric: GCX\nct: [3, 6, 9, 12]\n",
    })
    assert configured_rics(1, "traditional", config_path=path) == {"GCXc1", "GCXc2", "GCXc3", "GCXc4"}


def test_configured_rics_expands_every_ric_when_a_future_has_two(tmp_path):
    path = _write_tier_config(tmp_path, {
        "CL": "asset_class: commodity\nric: [CLOLD, CLNEW]\n",
    })
    assert configured_rics(1, "commodity", config_path=path) == {
        "CLOLDc1", "CLOLDc2", "CLOLDc3", "CLOLDc4",
        "CLNEWc1", "CLNEWc2", "CLNEWc3", "CLNEWc4",
    }


# ---------------------------------------------------------------------------
# artifact_path / shard_class_for / trade_counts: small pure functions that back
# the traps `build_trade_presence.py` exists to avoid (a tier-less filename, an
# equity directory that doesn't collapse, an artifact-missing failure mode that
# looks like "every RIC was dark"). Each gets a direct assertion so a regression
# in the function itself is caught here, not only through the CLI tests below.
# ---------------------------------------------------------------------------


def test_artifact_path_names_the_tier_into_the_filename():
    # A tier-less name would let a tier-2 run open a tier-1 artifact, match no RICs,
    # and silently rescue nothing.
    assert trade_presence.artifact_path(2, "currency", "et").name == "tier2_currency_et.parquet"


def test_shard_class_for_collapses_us_and_nonus_equity_onto_one_directory():
    assert trade_presence.shard_class_for("us_equity") == "equity"
    assert trade_presence.shard_class_for("nonus_equity") == "equity"


def test_trade_counts_raises_naming_the_path_and_the_build_command(tmp_path, monkeypatch):
    # An empty frame here would be indistinguishable from "every RIC was dark every
    # day", which is the wrong answer for "the artifact was never built".
    monkeypatch.setattr(trade_presence, "TICKHISTORY_PATH", tmp_path)
    with pytest.raises(FileNotFoundError, match="build_trade_presence.py"):
        trade_presence.trade_counts(9, "nope", "et")


# ---------------------------------------------------------------------------
# build_trade_presence.main: the CLI end to end. configured_rics and
# count_trades_in_frame are both well covered above in isolation, but nothing
# short of driving main() itself proves the CLI actually applies the RIC filter,
# reads the collapsed equity directory, or refuses to write under datasets/ --
# each of those is a defect that leaves every other test in this file green.
# ---------------------------------------------------------------------------

def _write_shard(shard_dir, filename, rows):
    """rows: (ric, date_time_string, price, volume, gmt_offset)."""
    shard_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "#RIC": [r[0] for r in rows],
            "Date-Time": [r[1] for r in rows],
            "Price": [r[2] for r in rows],
            "Volume": [r[3] for r in rows],
            "GMT Offset": [r[4] for r in rows],
        },
        schema_overrides={"Date-Time": pl.String},
    ).write_parquet(shard_dir / filename)


def test_cli_filters_to_configured_rics_and_writes_the_tier_named_artifact(tmp_path, monkeypatch):
    # A two-row shard, one configured RIC and one that no future in the synthetic
    # config names. If the CLI stops applying the RIC filter, the unconfigured RIC
    # ships in the artifact; if artifact_path stops naming the tier, this test's own
    # read of the expected filename fails instead.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_tier_config(config_dir, {"CL": "asset_class: commodity\nric: CLX\n"})
    data_root = tmp_path / "tickhistory"
    monkeypatch.setattr(trade_presence, "PROJECT_ROOT", config_dir)
    monkeypatch.setattr(trade_presence, "TICKHISTORY_PATH", data_root)
    monkeypatch.setattr(build_trade_presence, "TICKHISTORY_PATH", data_root)

    _write_shard(data_root / "trades" / "tier1_commodity_trades", "2020-01.parquet", [
        ("CLXc1", "2020-01-02T14:31:00.000000000Z", 10.0, 1, 0),
        ("CLXc1", "2020-01-02T14:45:00.000000000Z", 11.0, 1, 0),
        ("ZZc1", "2020-01-02T14:31:00.000000000Z", 12.0, 1, 0),
    ])

    rc = build_trade_presence.main(["--tier", "1", "--asset_class", "commodity", "--sync_target", "et"])
    assert rc == 0

    out = pl.read_parquet(data_root / "trade_presence" / "tier1_commodity_et.parquet")
    assert out["ric"].to_list() == ["CLXc1"]
    assert out["n_trades"].to_list() == [2]


def test_cli_collapses_us_and_nonus_equity_onto_the_shared_equity_shard(tmp_path, monkeypatch):
    # us_equity and nonus_equity are two separate asset classes that read one shard
    # directory (tier{tier}_equity_trades). If that collapse stopped happening, a
    # nonus_equity run would look for tier1_nonus_equity_trades, which does not
    # exist, and abort instead of reading real ticks.
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_tier_config(config_dir, {"AEX": "asset_class: nonus_equity\nric: AEX\n"})
    data_root = tmp_path / "tickhistory"
    monkeypatch.setattr(trade_presence, "PROJECT_ROOT", config_dir)
    monkeypatch.setattr(trade_presence, "TICKHISTORY_PATH", data_root)
    monkeypatch.setattr(build_trade_presence, "TICKHISTORY_PATH", data_root)

    _write_shard(data_root / "trades" / "tier1_equity_trades", "2020-01.parquet", [
        ("AEXc1", "2020-01-02T14:31:00.000000000Z", 10.0, 1, 0),
    ])

    rc = build_trade_presence.main(["--tier", "1", "--asset_class", "nonus_equity", "--sync_target", "et"])
    assert rc == 0

    out = pl.read_parquet(data_root / "trade_presence" / "tier1_nonus_equity_et.parquet")
    assert out["ric"].to_list() == ["AEXc1"]


def test_cli_refuses_to_write_the_artifact_under_datasets(tmp_path, monkeypatch):
    # Mirrors the pipeline stages' own startup checks: this artifact is a debug/
    # intermediate product, never a shipped dataset, so a misconfigured path env var
    # must not be able to make main() overwrite something under datasets/. Both
    # TICKHISTORY_PATH references are pinned to an empty tmp tree (no shard directory
    # under it) so that if the guard itself were the thing removed, main() would hit
    # the harmless "no trades shard directory" exit next rather than falling through
    # to a real, multi-hundred-GB production shard scan -- and the match string below
    # is the guard's own wording, not a path fragment either exit could share.
    datasets_root = tmp_path / "datasets"
    monkeypatch.setattr(build_trade_presence, "DATASETS_ROOT", datasets_root)
    monkeypatch.setattr(build_trade_presence, "TICKHISTORY_PATH", datasets_root / "tick")
    monkeypatch.setattr(trade_presence, "TICKHISTORY_PATH", datasets_root / "tick")

    with pytest.raises(SystemExit, match="refusing to write"):
        build_trade_presence.main(["--tier", "1", "--asset_class", "commodity", "--sync_target", "et"])


# ---------------------------------------------------------------------------
# is_rescuable_mask: a flagged cell is a genuine roll, not a real cross-contract
# return, when slot 1 was dark on every day from a ladder expiry through the panel's
# previous row and traded again at the flagged row. Darkness is read entirely off the
# counts artifact (an absent row, never a zero); the panel supplies only the window's
# right edge, which is the previous panel ROW, not the calendar day before -- weekends
# and holidays sit between them, and an expiry landing in that gap must not rescue.
# ---------------------------------------------------------------------------


def test_rescues_a_genuine_roll():
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 20)],
        "n_trades": [10, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [True]


def test_rescues_when_the_last_trade_lands_exactly_on_the_expiry_day():
    # A contract's own expiry is routinely also its last real day of trading (BRN,
    # 1996-01-16: LCOc1 prints 330 trades on the ladder's own expiry date, then goes
    # dark). The window's left edge has to be closed at that date, not open, or this
    # -- the ordinary case, not an edge case -- would never rescue.
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 16), date(2020, 1, 20)],
        "n_trades": [10, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [True]


def test_declines_when_no_expiry_falls_inside_the_dark_run():
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 20)],
        "n_trades": [10, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 3, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_declines_when_the_expiry_falls_after_the_panel_s_previous_row():
    # Slot 1 last traded 01-15; the panel's previous row is 01-17; the flagged cell is
    # 01-20. The expiry lands 01-19 -- inside the loose calendar window (01-15, 01-20)
    # but outside the closed panel window [01-16, 01-17]. The right edge has to be the
    # panel's own previous row: a window that instead runs to date_ minus one calendar
    # day reaches across whatever weekend or holiday separates the two rows and rescues
    # a return an expiry sitting in that gap should not touch. This is the one case
    # that tells the two readings apart -- every other test here passes under either.
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 20)],
        "n_trades": [10, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 19)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_the_left_edge_search_ignores_trading_outside_the_panel_calendar():
    # LGOc1 (ICE Gas Oil) prints 6 trades on Sunday 2004-01-18, between the panel's
    # previous row (Friday the 16th) and the flagged row (Monday the 19th) -- a day
    # the panel never has a row for. Searching the full calendar for "the last trade
    # before the flagged row" would land on that Sunday, which sits after
    # prev_row_date and inverts the window so no expiry can ever fall inside it, on a
    # RIC the panel's own return never asked about the weekend at all. The search has
    # to stop at prev_row_date, not date_, so the Sunday print cannot reach it.
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1", "Xc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 18), date(2020, 1, 20)],
        "n_trades": [10, 6, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [True]


def test_declines_when_slot_one_traded_at_t_minus_1():
    # Darkness is an absent row, never a zero: here c1 has a row at t-1 (it traded),
    # so the dark run ending at the panel's previous row is empty and no expiry can
    # fall inside it, regardless of where the expiry actually sits.
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1", "Xc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 17), date(2020, 1, 20)],
        "n_trades": [10, 8, 12],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_declines_when_the_ric_never_appears_in_counts_at_all():
    # No prior trade to anchor a dark run on means there is no window to test an
    # expiry against -- absence of the artifact's evidence must not be read as "always
    # dark", which is the wrong direction (over-rescue) for a rule with a measured
    # false-positive target of zero.
    counts = pl.DataFrame({
        "ric": ["Yc1"], "date_": [date(2020, 1, 15)], "n_trades": [5],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_declines_when_slot_one_does_not_trade_at_the_flagged_row():
    # Trading resuming at t is the other half of the rule alongside the dark run --
    # a row that is still dark at t is not yet a roll back into slot 1.
    counts = pl.DataFrame({
        "ric": ["Xc1"], "date_": [date(2020, 1, 15)], "n_trades": [10],
    })
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_declines_when_the_flagged_row_is_the_rics_first_ever_trade():
    # The only trade this ric has ever printed is the flagged row itself -- there is
    # no earlier trade to anchor a dark run on, so there is no window to test an
    # expiry against, the same as when the ric is absent from counts altogether. This
    # also has to not raise: with no earlier trade, "the trade before the flagged row"
    # is undefined, not some sentinel date an unguarded comparison could silently
    # accept.
    counts = pl.DataFrame({"ric": ["Xc1"], "date_": [date(2020, 1, 20)], "n_trades": [12]})
    flagged = pl.DataFrame({
        "symbol": ["X"], "date_": [date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17)], "expiries": [[date(2020, 1, 16)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [False]


def test_rescues_multiple_flagged_rows_independently():
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1", "Yc1", "Yc1"],
        "date_": [date(2020, 1, 15), date(2020, 1, 20), date(2020, 1, 15), date(2020, 1, 20)],
        "n_trades": [10, 12, 7, 9],
    })
    flagged = pl.DataFrame({
        "symbol": ["X", "Y"],
        "date_": [date(2020, 1, 20), date(2020, 1, 20)],
        "prev_row_date": [date(2020, 1, 17), date(2020, 1, 17)],
        "expiries": [[date(2020, 1, 16)], [date(2020, 3, 1)]],
    })
    assert is_rescuable_mask(flagged, counts).to_list() == [True, False]


# ---------------------------------------------------------------------------
# rescue_ret1_adjusted: wires is_rescuable_mask into a symbol's own debug frame.
# Recomputes which rows the guard flagged from ret1/shift/front_slot itself, then
# restores ret1 into ret1_adjusted only for the rows the rescue rule actually clears
# -- a real cross-contract break next to a rescued one must not leak the restore.
# ---------------------------------------------------------------------------


def test_rescue_restores_only_the_rescuable_flagged_row():
    # Three rows: 01-15 is an ordinary day (never flagged); 01-17 is flagged (slot
    # moved 1->2, not a roll) but slot 1 never trades again at 01-17 in counts, so it
    # stays null; 01-20 is flagged (slot moved 2->1) and slot 1's dark run (last trade
    # 01-10, an expiry at 01-16) resolves inside [first dark day, 01-17], so it gets
    # its ret1 restored.
    symbol_data = pl.DataFrame({
        "date_": [date(2020, 1, 15), date(2020, 1, 17), date(2020, 1, 20)],
        "front_slot": [1, 2, 1],
        "shift": [0, 0, 0],
        "ret1": [0.01, 0.02, 0.03],
        "ret1_adjusted": [0.01, None, None],
    })
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 10), date(2020, 1, 20)],
        "n_trades": [5, 10],
    })
    out = rescue_ret1_adjusted(symbol_data, "X", [date(2020, 1, 16)], counts)
    assert out["ret1_adjusted"].to_list() == [0.01, None, 0.03]
    assert out.columns == symbol_data.columns
    assert out.schema == symbol_data.schema


def test_rescue_declines_a_backward_slot_move_even_when_darkness_alone_would_rescue_it():
    # front_slot moves 1->2 at the flagged row -- a continuation chain never reverts
    # to an earlier contract, so this can only be the price selector's gap-fill firing
    # on a missing front-month print, a genuine cross-contract return regardless of
    # what the darkness check says. Slot 1 here satisfies every darkness condition
    # (a dark run from an expiry through the panel's previous row, trading again at
    # the flagged row) on its own, so restoring ret1 would prove the direction check
    # is not doing any work; leaving it null proves it is.
    symbol_data = pl.DataFrame({
        "date_": [date(2020, 1, 15), date(2020, 1, 17), date(2020, 1, 20)],
        "front_slot": [1, 1, 2],
        "shift": [0, 0, 0],
        "ret1": [0.01, 0.02, 0.03],
        "ret1_adjusted": [0.01, 0.02, None],
    })
    counts = pl.DataFrame({
        "ric": ["Xc1", "Xc1"],
        "date_": [date(2020, 1, 10), date(2020, 1, 20)],
        "n_trades": [5, 10],
    })
    out = rescue_ret1_adjusted(symbol_data, "X", [date(2020, 1, 16)], counts)
    assert out["ret1_adjusted"].to_list() == [0.01, 0.02, None]


def test_rescue_is_a_no_op_when_nothing_was_flagged():
    symbol_data = pl.DataFrame({
        "date_": [date(2020, 1, 15), date(2020, 1, 16)],
        "front_slot": [1, 1],
        "shift": [0, 0],
        "ret1": [0.01, 0.02],
        "ret1_adjusted": [0.01, 0.02],
    })
    counts = pl.DataFrame({"ric": [], "date_": [], "n_trades": []}, schema={
        "ric": pl.String, "date_": pl.Date, "n_trades": pl.UInt32,
    })
    out = rescue_ret1_adjusted(symbol_data, "X", [], counts)
    assert out["ret1_adjusted"].to_list() == [0.01, 0.02]


# ---------------------------------------------------------------------------
# ladder_expiries_for: the union of a clscode's own rolling lasttrddate_1..5 columns,
# never dsfutcontrinfo.csv -- each row is only a snapshot of the next few expiries as
# of some date, so the full history has to come from every row, not the latest one.
# ---------------------------------------------------------------------------


def test_ladder_expiries_for_unions_every_rung_across_every_row():
    ladder = pl.DataFrame({
        "clscode": [1, 1, 2],
        "lasttrddate_1": [date(2020, 1, 15), date(2020, 2, 15), date(2020, 6, 1)],
        "lasttrddate_2": [date(2020, 2, 15), date(2020, 3, 15), None],
        "lasttrddate_3": [None, None, None],
        "lasttrddate_4": [None, None, None],
        "lasttrddate_5": [None, None, None],
    })
    assert ladder_expiries_for([1], ladder) == [date(2020, 1, 15), date(2020, 2, 15), date(2020, 3, 15)]


def test_ladder_expiries_for_combines_multiple_clscodes():
    ladder = pl.DataFrame({
        "clscode": [1, 2],
        "lasttrddate_1": [date(2020, 1, 15), date(2020, 6, 1)],
        "lasttrddate_2": [None, None],
        "lasttrddate_3": [None, None],
        "lasttrddate_4": [None, None],
        "lasttrddate_5": [None, None],
    })
    assert ladder_expiries_for([1, 2], ladder) == [date(2020, 1, 15), date(2020, 6, 1)]


# ---------------------------------------------------------------------------
# clscodes_for / build_config: small pure lookups the rescue's expiry ladder and the
# offline measurement script both depend on -- a regression here silently changes
# which contracts the ladder is built from, or which futures a panel measures at all.
# ---------------------------------------------------------------------------


def _minimal_future(symbol, asset_class, ric=None, ct=None, clscode=1):
    return Future(
        symbol=symbol, contrcode=1, exchange=1, exchange_name="Test Exchange",
        clscode=clscode, calcseriesname="TEST", name="Test Future",
        asset_class=[AssetClass(asset_class)], ric=ric, ct=ct,
    )


def test_clscodes_for_uses_the_symbol_specific_mapping():
    assert clscodes_for(_minimal_future("BRN", "commodity")) == [1175, 1970]
    assert clscodes_for(_minimal_future("G", "commodity")) == [1181, 1176]


def test_clscodes_for_falls_back_to_the_futures_own_clscode():
    assert clscodes_for(_minimal_future("CL", "commodity", clscode=42)) == [42]


def test_build_config_splits_a_two_ric_future_into_historical_and_active_legs(tmp_path):
    path = _write_tier_config(tmp_path, {"CL": "asset_class: commodity\nric: [CLOLD, CLNEW]\n"})
    config = build_config(1, AssetClass.COMMODITY, config_path=str(path))
    assert [(f.symbol, f.ric) for f in config] == [("CLOLD", ["CLOLD"]), ("CL", ["CLNEW"])]


def test_build_config_for_traditional_requires_a_trading_cycle(tmp_path):
    path = _write_tier_config(tmp_path, {
        "CL": "asset_class: traditional\nric: CLX\n",
        "GC": "asset_class: traditional\nric: GCX\nct: [3, 6, 9, 12]\n",
    })
    config = build_config(1, AssetClass.TRADITIONAL, config_path=str(path))
    assert [f.symbol for f in config] == ["GC"]
