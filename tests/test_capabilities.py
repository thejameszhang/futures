import re
from pathlib import Path

import pytest

from globalmacro.utils import capabilities as cap

REPO = Path(__file__).resolve().parents[1]


def _make_shards(root, stems, marker=True):
    for stem in stems:
        for kind in ("trades", "quotes"):
            d = root / kind / f"{stem}_{kind}"
            d.mkdir(parents=True)
            (d / "2020-01.parquet").write_bytes(b"x")
            if marker:
                (d / "_GATE1_OK").write_text("gate1 ok\n")


def test_all_present_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    _make_shards(tmp_path, cap.SHARD_STEMS)
    c = cap.shards_ready()
    assert c.ready is True
    assert c.message is None


def test_none_present_is_not_ready_and_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    (tmp_path / "trades").mkdir(parents=True)
    (tmp_path / "quotes").mkdir(parents=True)
    c = cap.shards_ready()
    assert c.ready is False
    assert c.message is None          # the expected researcher state, not a warning


def test_partial_names_the_missing_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    _make_shards(tmp_path, cap.SHARD_STEMS[:-1])
    c = cap.shards_ready()
    assert c.ready is False
    # Pin the `missing` branch's own phrasing, not just the directory names it shares
    # with the `unverified` branch (a nonexistent directory also lacks _GATE1_OK, so
    # asserting only the names passes even if the `missing` branch is deleted).
    assert c.message is not None
    assert "missing tick shard directories" in c.message
    assert "tier2_equity_trades" in c.message
    assert "tier2_equity_quotes" in c.message
    assert cap.GATE1_MARKER not in c.message


def test_unverified_names_a_command_that_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    _make_shards(tmp_path, cap.SHARD_STEMS, marker=False)
    c = cap.shards_ready()
    assert c.ready is False
    assert c.message is not None
    assert "_GATE1_OK" in c.message
    assert "split_tickhistory.sh" in c.message


def test_unsplit_monoliths_name_the_split_script(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    for kind in ("trades", "quotes"):
        (tmp_path / kind).mkdir(parents=True)
        (tmp_path / kind / f"tier1_commodity_{kind}.csv").write_text("#RIC\n")
    c = cap.shards_ready()
    assert c.ready is False
    assert c.message is not None
    assert "split_tickhistory.sh" in c.message


def test_partial_with_leftover_monoliths_names_the_split_hint(tmp_path, monkeypatch):
    """The natural intermediate state: split_tickhistory.sh takes one side at a time,
    so trades can be fully split (with markers) while quotes is still bare monolith
    CSVs. The researcher should get the split-script hint, not just bare dir names."""
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    for stem in cap.SHARD_STEMS:
        d = tmp_path / "trades" / f"{stem}_trades"
        d.mkdir(parents=True)
        (d / cap.GATE1_MARKER).write_text("gate1 ok\n")
    (tmp_path / "quotes").mkdir(parents=True)
    for stem in cap.SHARD_STEMS:
        (tmp_path / "quotes" / f"{stem}_quotes.csv").write_text("#RIC\n")

    c = cap.shards_ready()
    assert c.ready is False
    assert c.message is not None
    assert "missing tick shard directories" in c.message
    assert "split_tickhistory.sh" in c.message


def test_stage_outputs_do_not_collapse_like_shard_stems():
    names = {f"{tier}/{cls}" for tier, cls in cap.SYNC_STAGE_OUTPUTS}
    assert "tier1/us_equity" in names and "tier1/nonus_equity" in names
    assert len(cap.SYNC_STAGE_OUTPUTS) == 10
    assert len(cap.SHARD_STEMS) == 9


def test_resolve_mode():
    ready = cap.Capability(True, None)
    absent = cap.Capability(False, None)
    assert cap.resolve_mode(None, ready, "build") == "full"
    assert cap.resolve_mode(None, absent, "build") == "async-only"
    assert cap.resolve_mode("async-only", ready, "build") == "async-only"
    assert cap.resolve_mode("full", ready, "build") == "full"
    with pytest.raises(SystemExit):
        cap.resolve_mode("full", absent, "build")


def test_resolve_mode_auto_prints_diagnostic_on_partial_state(capsys):
    """The auto path must not throw away cap.message -- a partial state (e.g. 6
    of 10 tickhistory stage outputs present) must surface it, on stderr only."""
    partial = cap.Capability(False, "missing tickhistory stage outputs: foo.csv")
    mode = cap.resolve_mode(None, partial, "build")
    assert mode == "async-only"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "globalmacro build: sync inputs not ready -> async-only mode." in captured.err
    assert "missing tickhistory stage outputs: foo.csv" in captured.err


def test_resolve_mode_auto_silent_on_clean_absent_state(capsys):
    """A cleanly-absent capability (message is None -- the researcher's normal state)
    must stay silent: that is not a warning."""
    absent = cap.Capability(False, None)
    mode = cap.resolve_mode(None, absent, "build")
    assert mode == "async-only"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_resolve_mode_auto_silent_on_ready_path(capsys):
    ready = cap.Capability(True, None)
    mode = cap.resolve_mode(None, ready, "build")
    assert mode == "full"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_resolve_mode_explicit_flags_stay_silent_regardless_of_message(capsys):
    """Unchanged behaviour for both explicit flags: the diagnostic print is on the
    auto path (flag is None) only."""
    ready = cap.Capability(True, None)
    partial = cap.Capability(False, "missing tickhistory stage outputs: foo.csv")
    assert cap.resolve_mode("full", ready, "build") == "full"
    assert cap.resolve_mode("async-only", partial, "build") == "async-only"
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


def test_resolve_mode_rejects_unrecognized_flags():
    """A flag that isn't exactly "full" or "async-only" must not silently fall through
    to auto-detection -- that's the exact silent degradation --full exists to prevent."""
    ready = cap.Capability(True, None)
    absent = cap.Capability(False, None)
    with pytest.raises(ValueError, match="Full"):
        cap.resolve_mode("Full", absent, "build")
    with pytest.raises(ValueError, match="bogus"):
        cap.resolve_mode("bogus", ready, "build")


def _make_sync_stage_outputs(root, pairs):
    for tier, cls in pairs:
        d = root / tier / "sync"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{cls}_daily_returns.csv").write_text("date\n")


def test_sync_stage_outputs_all_present_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_stage_outputs(tmp_path, cap.SYNC_STAGE_OUTPUTS)
    c = cap.sync_stage_outputs_ready()
    assert c.ready is True
    assert c.message is None


def test_sync_stage_outputs_none_present_is_not_ready_and_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    c = cap.sync_stage_outputs_ready()
    assert c.ready is False
    assert c.message is None


def test_sync_stage_outputs_partial_names_the_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_stage_outputs(tmp_path, cap.SYNC_STAGE_OUTPUTS[:-1])
    c = cap.sync_stage_outputs_ready()
    assert c.ready is False
    assert c.message is not None
    assert "tier2/sync/equity_daily_returns.csv" in c.message


def _make_sync_panels(root, present):
    all_paths = {
        "tier1_daily": root / "tier1" / "sync" / "sync_daily.csv",
        "tier2_daily": root / "tier2" / "sync" / "sync_daily.csv",
        "tier2_currency": root / "tier2" / "sync" / "currency_daily_returns.csv",
    }
    for key in present:
        p = all_paths[key]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("date\n")


def test_sync_panels_all_present_is_ready(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    # sync_panels_ready() consults sync_stage_outputs_ready() first, so a
    # healthy check needs the 10 stage outputs on disk too, not just the 3 aggregates.
    _make_sync_stage_outputs(tmp_path, cap.SYNC_STAGE_OUTPUTS)
    _make_sync_panels(tmp_path, ["tier1_daily", "tier2_daily", "tier2_currency"])
    c = cap.sync_panels_ready()
    assert c.ready is True
    assert c.message is None


def test_sync_panels_none_present_is_not_ready_and_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    c = cap.sync_panels_ready()
    assert c.ready is False
    assert c.message is None


def test_sync_panels_partial_names_the_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_stage_outputs(tmp_path, cap.SYNC_STAGE_OUTPUTS)
    # tier2_currency's path IS one of SYNC_STAGE_OUTPUTS (both predicates check the
    # same tier2/sync/currency_daily_returns.csv -- see sync_panels_fresh's comment),
    # so it's already present from the stage-outputs fixture above; the only
    # genuinely independent missing panel is sync_daily.csv itself, a build.py-written
    # aggregate that is not a raw stage output.
    _make_sync_panels(tmp_path, ["tier1_daily", "tier2_currency"])
    c = cap.sync_panels_ready()
    assert c.ready is False
    assert c.message is not None
    assert "tier2/sync/sync_daily.csv" in c.message


# --- sync_panels_ready() must agree with sync_stage_outputs_ready() ----------------


def test_sync_panels_ready_defers_to_stage_outputs_when_they_disagree(tmp_path, monkeypatch):
    """The exact defect this guards against: the 3 aggregate panels present, but the raw
    tickhistory stage outputs gone (a researcher reclaiming disk after a full run).
    Before this fix, sync_panels_ready() reported ready=True (validate resolves full) while
    build's own sync_stage_outputs_ready()-based check reported not-ready (build resolves
    async-only) -- a silent disagreement. Now both must agree: not ready."""
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_panels(tmp_path, ["tier1_daily", "tier2_daily", "tier2_currency"])
    stage = cap.sync_stage_outputs_ready()
    panels = cap.sync_panels_ready()
    assert stage.ready is False
    assert panels.ready is False
    assert panels.message == stage.message


def test_sync_panels_ready_propagates_partial_stage_outputs_message(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    _make_sync_stage_outputs(tmp_path, cap.SYNC_STAGE_OUTPUTS[:-1])
    _make_sync_panels(tmp_path, ["tier1_daily", "tier2_daily", "tier2_currency"])
    c = cap.sync_panels_ready()
    assert c.ready is False
    assert c.message is not None
    assert "tier2/sync/equity_daily_returns.csv" in c.message


def test_sync_panels_ready_silent_on_genuinely_clean_tree(tmp_path, monkeypatch):
    """Nothing on disk anywhere: stage outputs are cleanly absent (message=None), and
    that must still propagate as message=None -- not a warning for the researcher's
    normal starting state."""
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path)
    c = cap.sync_panels_ready()
    assert c.ready is False
    assert c.message is None


def test_healthy_full_machine_all_four_predicates_ready(tmp_path, monkeypatch):
    """sync_panels_ready()'s consultation of sync_stage_outputs_ready() must be a
    no-op on a healthy full machine: all 10 stage outputs, all 3
    sync panels, matching async panels (for freshness), and a full tick-shard tree
    all present -> every predicate reports ready."""
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path / "tick")
    monkeypatch.setattr(cap, "DATASETS_ROOT", tmp_path / "datasets")
    _make_shards(tmp_path / "tick", cap.SHARD_STEMS)
    _make_sync_stage_outputs(tmp_path / "datasets", cap.SYNC_STAGE_OUTPUTS)
    # async_daily.csv BEFORE the sync panels: build.py always writes the async
    # aggregate first and the sync aggregate immediately after (sync_panels_fresh's
    # own rule), so sync must not predate async here or freshness would (correctly)
    # flag this fixture as stale.
    for tier in ("tier1", "tier2"):
        d = tmp_path / "datasets" / tier / "async"
        d.mkdir(parents=True, exist_ok=True)
        (d / "async_daily.csv").write_text("date\n")
    _make_sync_panels(tmp_path / "datasets", ["tier1_daily", "tier2_daily", "tier2_currency"])

    assert cap.shards_ready().ready is True
    assert cap.sync_stage_outputs_ready().ready is True
    assert cap.sync_panels_ready().ready is True
    assert cap.sync_panels_fresh().ready is True


def _collapse(cls: str) -> str:
    """Mirrors tickhistory.py's `__main__` block -- us_equity and nonus_equity share
    one shard set."""
    return "equity" if "equity" in cls else cls


def test_shard_stems_match_run_all_sh():
    """SHARD_STEMS must equal the stems run_all.sh's tickhistory jobs resolve to.
    The two london currency jobs reuse tier{1,2}_currency, so 12 jobs -> 9 stems."""
    text = (REPO / "slurm" / "run_all.sh").read_text()
    loops = re.findall(r"for c in ([\w ]+); do", text)
    assert len(loops) == 2, f"expected the tier-1 and tier-2 class loops, found {loops}"
    tier1_classes, tier2_classes = (set(g.split()) for g in loops)
    expected = ({f"tier1_{_collapse(c)}" for c in tier1_classes}
                | {f"tier2_{_collapse(c)}" for c in tier2_classes})
    assert set(cap.SHARD_STEMS) == expected, (
        f"SHARD_STEMS drifted from run_all.sh: {set(cap.SHARD_STEMS) ^ expected}")


def test_sync_stage_outputs_match_build_synced_dataset():
    """SYNC_STAGE_OUTPUTS must equal the (tier, class) pairs build_synced_dataset
    actually reads via DATASETS_ROOT -- guards the 10 filename literals against
    drift, the way test_shard_stems_match_run_all_sh guards SHARD_STEMS."""
    text = (REPO / "src" / "globalmacro" / "build.py").read_text()
    start = text.index("def build_synced_dataset(")
    end = text.index("\ndef ", start + 1)
    body = text[start:end]

    literal_pairs = set(re.findall(
        r'DATASETS_ROOT\s*/\s*"(tier\d)"\s*/\s*"sync"\s*/\s*"(\w+)_daily_returns\.csv"',
        body))

    loop_match = re.search(
        r'for (\w+) in \[([^\]]+)\]:'
        r'(?:.*\n)*?.*DATASETS_ROOT\s*/\s*"(tier\d)"\s*/\s*"sync"\s*/\s*f"\{(\w+)\}_daily_returns\.csv"',
        body)
    assert loop_match, "expected the tier-2 f-string read_csv loop in build_synced_dataset"
    loop_var, items, loop_tier, used_var = loop_match.groups()
    assert loop_var == used_var
    loop_pairs = {(loop_tier, item.strip().strip('"').strip("'")) for item in items.split(",")}

    found = literal_pairs | loop_pairs
    assert found, "parser found zero read_csv(DATASETS_ROOT...) calls; fix the regex " \
                   "before trusting this guard"
    assert found == set(cap.SYNC_STAGE_OUTPUTS), (
        "SYNC_STAGE_OUTPUTS drifted from build_synced_dataset: "
        f"{found ^ set(cap.SYNC_STAGE_OUTPUTS)}")
