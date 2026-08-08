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
    assert "tier2_equity_trades" in c.message
    assert "tier2_equity_quotes" in c.message


def test_unverified_names_a_command_that_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    _make_shards(tmp_path, cap.SHARD_STEMS, marker=False)
    c = cap.shards_ready()
    assert c.ready is False
    assert "_GATE1_OK" in c.message
    assert "split_tickhistory.sh" in c.message


def test_unsplit_monoliths_name_the_split_script(tmp_path, monkeypatch):
    monkeypatch.setattr(cap, "TICKHISTORY_PATH", tmp_path)
    for kind in ("trades", "quotes"):
        (tmp_path / kind).mkdir(parents=True)
        (tmp_path / kind / f"tier1_commodity_{kind}.csv").write_text("#RIC\n")
    c = cap.shards_ready()
    assert c.ready is False
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


def _collapse(cls: str) -> str:
    """Mirrors tickhistory.py:719 -- us_equity and nonus_equity share one shard set."""
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
