from datetime import date

import polars as pl
import pytest

from globalmacro.pipeline.tickhistory import (
    attach_front_slot,
    attach_traditional_front_slot,
)
from globalmacro.utils import trade_presence
from globalmacro.utils.trade_presence import configured_rics, count_trades_in_frame
from scripts import build_trade_presence


def test_front_slot_records_which_slot_supplied_the_price():
    df = pl.DataFrame({
        "expiring_this_month": [0, 1, 0],
        "settlement_c1": [100.0, 200.0, None],
        "settlement_c2": [101.0, 201.0, 301.0],
    })
    out = attach_front_slot(df)
    assert out["front_month_settlement"].to_list() == [100.0, 201.0, 301.0]
    assert out["front_slot"].to_list() == [1, 2, 2]


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
