"""tickhistory.py must be reproducible: the same logical input, in a different physical row
order or read back under a different POLARS_MAX_THREADS, must produce a byte-identical
output. See .superpowers/sdd/tickhistory-nondeterminism-rootcause.md for the investigation
this fixes.

Two independent mechanisms are involved, so two independent kinds of fixture are needed:

1. `.unique(keep='first')` (load_trades_data/load_quotes_data) returns its surviving rows in
   a different physical order on every call -- NOT a value bug (the dedup spans every
   column), but every downstream sort/tie-break inherits a fresh permutation each time.
   `maintain_order=True` pins it. This reproduces reliably even at ~30 rows scanned from a
   real parquet shard (verified while designing this fixture: without maintain_order=True,
   the same 30-row shard gave 10-15 distinct row orders across 15 calls; a directly
   in-memory-constructed DataFrame of the same rows did NOT reorder -- the disorder is a
   property of the scan/streaming-collect path the loaders actually use, which is exactly
   why these tests go through a real written-and-reread parquet shard rather than a bare
   pl.DataFrame).

2. The VWAP's `(Price*Volume).sum()` is a float reduction, so its result depends on addition
   order, which for a plain `.sum()` follows physical row/chunk order -- but only once polars'
   parallel group_by actually partitions the aggregation across threads. That turns out to
   need real production scale: tried synthetic parquet fixtures up to 2M rows, in a single
   settlement-window group, written with a tiny row_group_size to force many physical chunks,
   and none of them reproduced the drift (a directly in-memory-constructed frame doesn't
   either). So the two VWAP tests below are DATA-COUPLED to the real tick data instead of
   synthetic (skip, not fail, if it's absent) -- see their docstrings.
"""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import globalmacro.pipeline.tickhistory as th
from globalmacro.pipeline import tickhistory_shards as ts
from globalmacro.utils.paths import TICKHISTORY_PATH as REAL_TICKHISTORY_PATH

# ---------------------------------------------------------------------------
# Fix 1: loader row-order determinism (`.unique(keep='first', maintain_order=True)`)
# ---------------------------------------------------------------------------

# 5 distinct timestamps x 2 distinct trades (different Price) per timestamp, each row
# duplicated 3x (so .unique() has real deduplication to do, AND the surviving rows still
# tie on datetime -- both properties are needed: the dup rows exercise the loader's own
# .unique() call, and the datetime ties are what let a leftover permutation from .unique()
# leak past the loader's own final .sort("local_datetime") instead of being silently
# absorbed by it).
def _dup_trades_csv() -> str:
    lines = ["#RIC,Domain,Date-Time,GMT Offset,Type,Price,Volume\n"]
    for i in range(5):
        ts_str = f"2020-01-06T14:{30 + i:02d}:00.000000000Z"
        for j, price in enumerate((1550.0 + i, 1560.0 + i)):
            for _dup in range(3):
                lines.append(f"GCc1,Market Price,{ts_str},+0,Trade,{price},{j + 1}\n")
    return "".join(lines)


def _dup_quotes_csv() -> str:
    lines = ["#RIC,Domain,Date-Time,GMT Offset,Type,Close Bid,Close Ask\n"]
    for i in range(5):
        ts_str = f"2020-01-06T14:{30 + i:02d}:00.000000000Z"
        for bid, ask in ((1550.0 + i, 1550.5 + i), (1560.0 + i, 1560.5 + i)):
            for _dup in range(3):
                lines.append(f"GCc1,Market Price,{ts_str},+0,Intraday 1Min,{bid},{ask}\n")
    return "".join(lines)


def _hash_row_order(df: pl.DataFrame, cols: list[str]) -> str:
    return hashlib.sha256(df.select(cols).hash_rows(seed=0).to_numpy().tobytes()).hexdigest()


def test_load_trades_data_row_order_is_stable_across_repeated_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(th, "ALL_RICS", ["GCc1"], raising=False)
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    m.write_text(_dup_trades_csv())
    ts.convert_one(m, full_check=True)
    m.rename(tmp_path / "trades" / "_moved.csv")  # loader must read the shard, not the csv

    hashes = {_hash_row_order(th.load_trades_data("tier1_commodity_trades.csv"), ["#RIC", "Price", "Volume", "datetime"]) for _ in range(15)}
    assert len(hashes) == 1, f"row order varies across repeated calls: saw {len(hashes)} distinct orders in 15 calls"


def test_load_quotes_data_row_order_is_stable_across_repeated_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(th, "ALL_RICS", ["GCc1"], raising=False)
    m = tmp_path / "quotes" / "tier1_commodity_quotes.csv"
    m.parent.mkdir(parents=True)
    m.write_text(_dup_quotes_csv())
    ts.convert_one(m, full_check=True)
    m.rename(tmp_path / "quotes" / "_moved.csv")

    hashes = {_hash_row_order(th.load_quotes_data("tier1_commodity_quotes.csv"), ["#RIC", "Close Bid", "Close Ask", "datetime"]) for _ in range(15)}
    assert len(hashes) == 1, f"row order varies across repeated calls: saw {len(hashes)} distinct orders in 15 calls"


# ---------------------------------------------------------------------------
# Fix 2: VWAP sum determinism (`.sort_by(datetime, Price, Volume).sum()`)
# ---------------------------------------------------------------------------

# A tiny synthetic parquet (even multi-chunk, even 2M rows -- tried up to that while writing
# this test) never reproduces the VWAP's addition-order sensitivity: it turns out to require
# a DataFrame large enough that polars' parallel group_by actually partitions the work across
# threads, which a small fixture never triggers regardless of physical chunk count. So these
# two tests are DATA-COUPLED to the real tick data (skip, not fail, if it's absent) rather
# than synthetic -- the same tradeoff test_fx_futures_vs_spot.py makes for its real-data
# integration test. Both are read-only: the real shard is only ever symlinked into a scratch
# dir, never copied or written to, and TICKHISTORY_PATH is monkeypatched so nothing touches
# the real data/tickhistory/debug/ tree.
_REAL_COMMODITY_SHARD = REAL_TICKHISTORY_PATH / "trades" / "tier1_commodity_trades" / "2014-06.parquet"
_RICS_5x4 = [f"{r}c{i}" for r in ("CL", "NG", "W", "S", "KC") for i in (1, 2, 3, 4)]


def _symlink_shard(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "shard_dir"
    (shard_dir / "trades" / "tier1_commodity_trades").mkdir(parents=True)
    (shard_dir / "trades" / "tier1_commodity_trades" / "2014-06.parquet").symlink_to(_REAL_COMMODITY_SHARD)
    return shard_dir


def _settlement_window(shard_dir: Path, monkeypatch: pytest.MonkeyPatch) -> pl.DataFrame:
    monkeypatch.setattr(th, "TICKHISTORY_PATH", shard_dir)
    monkeypatch.setattr(th, "ALL_RICS", _RICS_5x4, raising=False)
    got = th.load_trades_data("tier1_commodity_trades.csv")
    return got.with_columns(pl.col("datetime_et").dt.date().alias("date_")).filter(
        (pl.col("datetime_et").dt.time() >= pl.time(9, 30)) & (pl.col("datetime_et").dt.time() <= pl.time(9, 31))
    )


@pytest.mark.skipif(not _REAL_COMMODITY_SHARD.exists(), reason="real tickhistory shard absent")
def test_vwap_is_identical_under_a_row_shuffle(tmp_path, monkeypatch):
    """DATA-COUPLED (see note above). The real 2014-06 commodity shard, CL/NG/W/S/KC c1-c4,
    388 settlement-window groups."""
    w = _settlement_window(_symlink_shard(tmp_path), monkeypatch)

    # Canary: prove the fixture is actually order-sensitive for a PLAIN sum (not calling any
    # shipped code) so a future polars upgrade that stops exhibiting the underlying
    # non-associativity fails loud here rather than leaving the fix assertion below silently
    # vacuous.
    def plain_vwap(f):
        agg = [(pl.col("Price") * pl.col("Volume")).sum().alias("pv"), pl.col("Volume").sum().alias("v")]
        return f.group_by(["#RIC", "date_", "order"]).agg(agg).with_columns((pl.col("pv") / pl.col("v")).alias("vwap"))

    shuffled = w.sample(fraction=1.0, shuffle=True, seed=5)
    plain_a = plain_vwap(w).sort(["#RIC", "date_", "order"])
    plain_b = plain_vwap(shuffled).sort(["#RIC", "date_", "order"])
    plain_diff = (plain_a["vwap"] - plain_b["vwap"]).abs()
    assert (plain_diff > 0).sum() > 0, "fixture stopped being order-sensitive for a plain VWAP sum -- canary invalid"

    fixed_a = th.compute_vwap(w, "datetime_et").sort(["#RIC", "date_", "order"])
    fixed_b = th.compute_vwap(shuffled, "datetime_et").sort(["#RIC", "date_", "order"])
    assert fixed_a["vwap"].equals(fixed_b["vwap"])


_THREAD_SCRIPT = """
import sys
from pathlib import Path
import polars as pl
import globalmacro.pipeline.tickhistory as th

th.TICKHISTORY_PATH = Path({shard_dir!r})
th.ALL_RICS = {rics!r}
got = th.load_trades_data("tier1_commodity_trades.csv")
w = got.with_columns(pl.col("datetime_et").dt.date().alias("date_")).filter(
    (pl.col("datetime_et").dt.time() >= pl.time(9, 30)) & (pl.col("datetime_et").dt.time() <= pl.time(9, 31))
)
out = th.compute_vwap(w, "datetime_et").sort(["#RIC", "date_", "order"])
import hashlib
print(hashlib.sha256(out["vwap"].to_numpy().tobytes()).hexdigest())
"""


@pytest.mark.skipif(not _REAL_COMMODITY_SHARD.exists(), reason="real tickhistory shard absent")
def test_vwap_is_identical_across_polars_max_threads(tmp_path):
    """DATA-COUPLED (see note above). Guards the exact property .superpowers/sdd/
    tickhistory-nondeterminism-rootcause.md sec 6 reports for the VWAP: same physical input,
    different POLARS_MAX_THREADS, must give the same result.

    HONEST CAVEAT, found while mutation-testing this test (reverting compute_vwap's keyed
    sum back to plain .sum() and re-running): on this shard, the mutant does NOT fail this
    test -- the *exact* shipped group_by(["#RIC","date_","order"]) shape turns out to be
    thread-count-stable even without the fix, at 1-32 threads, both on the 5-symbol sample
    used here and re-checked separately against the full shard (151 RICs, 16.7M rows, 1668
    groups). A group_by keyed on just ["#RIC","date_"] (dropping "order", which is redundant
    with "#RIC") DOES show thread sensitivity on the same data and thread range -- so the
    effect is real, but it depends on grouping-key shape in a way this investigation did not
    fully resolve, and it is the row-shuffle test above, not this one, that is the confirmed,
    mutation-killing regression guard for Fix 2. This test is kept because the invariant is
    still one the shipped code must hold and item 4 of the task asked for it explicitly, not
    because it is proven to catch a reversion on this data.
    """
    shard_dir = _symlink_shard(tmp_path)
    script = _THREAD_SCRIPT.format(shard_dir=str(shard_dir), rics=_RICS_5x4)

    hashes = set()
    for threads in (1, 2, 4, 8):
        env = {**os.environ, "POLARS_MAX_THREADS": str(threads)}
        out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        hashes.add(out.stdout.strip())
    assert len(hashes) == 1, f"VWAP hash varies across thread counts: {hashes}"


# ---------------------------------------------------------------------------
# Sites 2/3 (:326 today_lasttrdprices, :331 all_lasttrdprices): DECISION, not a fix.
# Pinning the loader's row order (Fix 1) is sufficient to make these deterministic; no
# secondary tie-break key was added. Evidence:
#   - naive repeated-call hashing of the FULL output frame appeared to "vary" (20/20 distinct)
#     even on a fixed physical input -- but that was a row-order artifact of group_by's own
#     non-guaranteed emission order (these frames are consumed downstream via a KEYED join on
#     #RIC/date_/order, never positionally), not a value change. Sorting on the total group
#     key before comparing (as this test does) is the correct way to ask "did the tie-break
#     assign a different VALUE to this key" -- and the answer, measured on the real 2014-06
#     commodity shard, is no: identical across 20 repeated calls AND across
#     POLARS_MAX_THREADS in {1, 2, 4, 8, 16, 32} in fresh processes.
#   - This test reproduces that finding at fixture scale, against the real (fixed)
#     load_trades_data, using expressions that mirror :326/:331 verbatim.
# ---------------------------------------------------------------------------

def _value_by_key_hash(df: pl.DataFrame, value_col: str) -> str:
    ordered = df.sort(["#RIC", "date_", "order"])
    return hashlib.sha256(ordered.select(["#RIC", "date_", "order", value_col]).hash_rows(seed=0).to_numpy().tobytes()).hexdigest()


def test_lasttrdprice_tiebreak_value_is_stable_once_loader_order_is_pinned(tmp_path, monkeypatch):
    monkeypatch.setattr(th, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(ts, "TICKHISTORY_PATH", tmp_path)
    monkeypatch.setattr(th, "ALL_RICS", ["GCc1"], raising=False)
    m = tmp_path / "trades" / "tier1_commodity_trades.csv"
    m.parent.mkdir(parents=True)
    m.write_text(_dup_trades_csv())  # same ambiguous-tie fixture as the loader test above
    ts.convert_one(m, full_check=True)
    m.rename(tmp_path / "trades" / "_moved.csv")

    datetime_column = "local_datetime"
    settlement_start = pl.time(23, 59)  # wide open: every row qualifies for "today"

    today_hashes = set()
    all_hashes = set()
    for _ in range(15):
        trades_data = th.load_trades_data("tier1_commodity_trades.csv").with_columns(
            pl.col(datetime_column).dt.date().alias("date_")
        )
        # mirrors tickhistory.py :326 (today_lasttrdprices)
        today = trades_data.filter(pl.col(datetime_column).dt.time() <= settlement_start).group_by(
            ["#RIC", "date_", "order"]
        ).agg([pl.col("Price").sort_by(pl.col(datetime_column)).last().alias("lasttrdprice_today")])
        today_hashes.add(_value_by_key_hash(today, "lasttrdprice_today"))

        # mirrors tickhistory.py :331 (all_lasttrdprices)
        allp = trades_data.group_by(["#RIC", "date_", "order"]).agg(
            [pl.col("Price").sort_by(pl.col(datetime_column)).last().alias("lasttrdprice")]
        )
        all_hashes.add(_value_by_key_hash(allp, "lasttrdprice"))

    assert len(today_hashes) == 1, f"today_lasttrdprices (:326) value-per-key varies: {today_hashes}"
    assert len(all_hashes) == 1, f"all_lasttrdprices (:331) value-per-key varies: {all_hashes}"
