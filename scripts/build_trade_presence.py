"""Build the per-RIC per-day trade-count artifact for one (tier, asset class, sync
target) triple.

    .venv/bin/python scripts/build_trade_presence.py --tier 1 --asset_class commodity --sync_target et

Run once per `globalmacro tickhistory` job (see `slurm/trade_presence.sh`): scans that
triple's trades shard directory, applies the same trade-counting rule
`globalmacro.utils.trade_presence.count_trades_in_frame` implements, and writes the
result to `globalmacro.utils.trade_presence.artifact_path(...)`.
"""

from __future__ import annotations

import argparse
import sys

import polars as pl

from globalmacro.utils.paths import DATASETS_ROOT, TICKHISTORY_PATH
from globalmacro.utils.trade_presence import (
    artifact_path,
    configured_rics,
    count_trades_in_frame,
    shard_class_for,
)

TRADE_COLUMNS = ["#RIC", "Date-Time", "Price", "Volume", "GMT Offset"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", type=int, required=True, choices=[1, 2])
    parser.add_argument("--asset_class", type=str, required=True)
    parser.add_argument("--sync_target", type=str, required=True, choices=["et", "london"])
    args = parser.parse_args(argv)

    out_path = artifact_path(args.tier, args.asset_class, args.sync_target)
    # Mirrors the pipeline stages' own startup checks: this artifact is a debug/
    # intermediate product, never a shipped dataset, so writing under datasets/ would
    # mean a misconfigured path env var silently clobbered a shipped file.
    if out_path.resolve().is_relative_to(DATASETS_ROOT.resolve()):
        raise SystemExit(f"refusing to write under datasets/: {out_path}")

    shard_dir = (
        TICKHISTORY_PATH / "trades" / f"tier{args.tier}_{shard_class_for(args.asset_class)}_trades"
    )
    if not shard_dir.is_dir():
        raise SystemExit(f"no trades shard directory at {shard_dir}")

    # us_equity and nonus_equity read the same shard directory (shard_class_for
    # collapses both to "equity"), so this RIC filter -- not the directory -- is what
    # makes their artifacts genuinely distinct rather than duplicates of one unfiltered
    # scan: it is scoped to the RICs THIS asset class's config actually names.
    rics = sorted(configured_rics(args.tier, args.asset_class))
    if not rics:
        raise SystemExit(
            f"no configured RICs for tier{args.tier} asset_class={args.asset_class!r} "
            f"-- check tier{args.tier}.yaml"
        )

    ticks = (
        pl.scan_parquet(shard_dir / "*.parquet")
        .select(TRADE_COLUMNS)
        .filter(pl.col("#RIC").is_in(rics))
    )
    counts = count_trades_in_frame(ticks, args.sync_target)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts.write_parquet(out_path)
    present = set(counts["ric"].unique().to_list())
    print(
        f"wrote {out_path} ({counts.height} rows, "
        f"{len(present)} distinct RICs; {len(rics)} configured, "
        f"{len(rics) - len(present)} never traded)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
