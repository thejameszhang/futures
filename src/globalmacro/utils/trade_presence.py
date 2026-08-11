"""Per-RIC, per-day trade counts derived from the raw tick record.

The provenance guard in `globalmacro.pipeline.tickhistory` nulls a sync return whenever
its two ends came from different contract slots. Some of those breaks are cosmetic --
the slot label moved because the previously-tracked contract went quiet, not because a
different contract's price shipped. Telling the two apart needs to know whether a
contract actually traded on a given day, and the sync panel's own prices cannot answer
that (a stale carry-forward print looks identical to a fresh one). This module builds
the artifact that can: how many trades a RIC printed on each calendar day, counted the
same way the ingestion stage counts them.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass
from globalmacro.utils.paths import PROJECT_ROOT, TICKHISTORY_PATH

# The two sync targets the pipeline supports, and the timezone each settlement day is
# keyed on -- mirrors DATETIME_COLUMN's et/london branches in
# globalmacro.pipeline.tickhistory.process_future. Local tick time and UTC are both the
# wrong day boundary (measured divergence on real shards); the day a trade counts
# toward must be derived exactly the way the consuming run derives it.
SYNC_TIMEZONES: dict[str, str] = {
    "et": "America/New_York",
    "london": "Europe/London",
}


def shard_class_for(asset_class: str) -> str:
    """Map an asset class onto the trades shard directory it actually lives in.

    `us_equity` and `nonus_equity` are two separate asset classes but read one shard
    directory (`tier{tier}_equity_trades`) -- the ingestion stage collapses them the
    same way when it builds the trades filename. A class -> directory map that skips
    this collapse opens a directory that does not exist for one of the two.
    """
    return "equity" if "equity" in asset_class else asset_class


def configured_rics(tier: int, asset_class: str, config_path: Path | None = None) -> set[str]:
    """The `c1`..`c4` RICs the ingestion stage actually queries for one (tier, asset
    class) triple.

    Mirrors the `CONFIG`/`ALL_RICS` construction in `tickhistory.py`'s `__main__`: a
    future with no `ric` never enters `CONFIG` there (some commodity and equity
    instruments carry no RIC), so it is skipped here rather than treated as an empty
    RIC list. `traditional` carries one more restriction the pipeline applies on top
    of the asset-class match -- only futures with a trading-cycle `ct` set enter its
    `CONFIG` at all -- so that restriction is reproduced here too, or the artifact
    would carry RICs the traditional run never queries.
    """
    target = AssetClass(asset_class)
    path = config_path if config_path is not None else PROJECT_ROOT / f"tier{tier}.yaml"
    rics: set[str] = set()
    for future in load_config(path):
        if future.ric is None or target not in future.asset_class:
            continue
        if target == AssetClass.TRADITIONAL and future.ct is None:
            continue
        for r in future.ric:
            rics.update(f"{r}c{i}" for i in range(1, 5))
    return rics


def artifact_path(tier: int, asset_class: str, sync_target: str) -> Path:
    """Where the trade-count artifact for one (tier, asset class, sync target) lives.

    The tier is part of the filename, not just the shard path, because tier-1 and
    tier-2 symbol sets are disjoint -- a tier-less name would let a tier-2 run open a
    tier-1 artifact, match no RICs at all, and silently rescue nothing.
    """
    return TICKHISTORY_PATH / "trade_presence" / f"tier{tier}_{asset_class}_{sync_target}.parquet"


def count_trades_in_frame(
    ticks: pl.DataFrame | pl.LazyFrame, sync_target: str
) -> pl.DataFrame:
    """Count trades per RIC per day, counting exactly what `load_trades_data` counts.

    `ticks` carries the raw shard columns `#RIC`, `Date-Time` (a string, never a
    parsed datetime), `Price`, `Volume`, `GMT Offset`. The transforms below run in the
    same order `load_trades_data` runs them in -- in particular, de-duplication runs
    BEFORE the null-price filter, so reordering them would count a different set of
    rows than the ingestion stage actually consumes.

    `Date-Time`'s offset is parsed from the string itself, never assumed to be UTC: a
    shard row can legitimately carry a non-`Z` offset, and stamping everything UTC
    would silently push the wrong ticks across a day boundary.

    The result is sorted by `(ric, date_)` before it is returned. `group_by` does not
    guarantee row order, so two runs over byte-identical input can otherwise emit the
    same rows in a different physical order -- a different artifact on disk despite
    agreeing on every value. `load_trades_data` hits the same non-determinism from its
    own `.unique()` and pins order for the same reason.
    """
    if sync_target not in SYNC_TIMEZONES:
        raise ValueError(
            f"unknown sync_target {sync_target!r}; expected one of {sorted(SYNC_TIMEZONES)}"
        )
    tz = SYNC_TIMEZONES[sync_target]
    lf = ticks.lazy() if isinstance(ticks, pl.DataFrame) else ticks
    return (
        lf.with_columns([
            pl.when(pl.col("Price") == 0.0).then(None).otherwise(pl.col("Price")).alias("Price"),
            pl.when(
                (pl.col("Volume") <= 0.0) | pl.col("Volume").is_null() | pl.col("Volume").is_nan()
            ).then(1.0).otherwise(pl.col("Volume")).alias("Volume"),
            pl.col("GMT Offset").cast(pl.Float32).alias("GMT Offset"),
        ])
        .unique(keep="first", maintain_order=True)
        .with_columns(
            pl.col("Date-Time")
            .str.replace("Z$", "+00:00")
            .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f%z")
            .dt.convert_time_zone(tz)
            .dt.date()
            .alias("date_")
        )
        .filter(pl.col("Price").is_not_null())
        .group_by(["#RIC", "date_"])
        .agg(pl.len().alias("n_trades"))
        .rename({"#RIC": "ric"})
        .sort(["ric", "date_"])
        .collect(engine="streaming")
    )


def trade_counts(tier: int, asset_class: str, sync_target: str) -> pl.DataFrame:
    """Read the artifact for one (tier, asset class, sync target) triple.

    Raises `FileNotFoundError` naming the missing path and the build command, rather
    than returning an empty frame -- an empty frame here would be indistinguishable
    from "every RIC was dark every day", which is the wrong answer for "the artifact
    was never built".
    """
    path = artifact_path(tier, asset_class, sync_target)
    if not path.is_file():
        raise FileNotFoundError(
            f"no trade-presence artifact at {path}; build it with "
            f".venv/bin/python scripts/build_trade_presence.py "
            f"--tier {tier} --asset_class {asset_class} --sync_target {sync_target}"
        )
    return pl.read_parquet(path)
