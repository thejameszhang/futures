"""What can this machine build?

Three separate questions, because three stages consume three different artifacts:
run_all.sh asks about tick shards (should tickhistory be submitted), build asks about
the tickhistory stage's OUTPUTS, and validate asks about the shipped sync panels.
Shard presence is not build-readiness.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from globalmacro.utils.paths import DATASETS_ROOT, TICKHISTORY_PATH

# The 9 shard stems the DAG consumes, cross-checked against slurm/run_all.sh by
# tests/test_capabilities.py. The 12 tickhistory jobs collapse to 9 because the two
# london currency jobs reuse tier{1,2}_currency, and because tickhistory.py:719 maps
# both us_equity and nonus_equity to "equity".
SHARD_STEMS: tuple[str, ...] = (
    "tier1_bond", "tier1_commodity", "tier1_currency", "tier1_equity",
    "tier1_stir", "tier1_traditional", "tier1_volatility",
    "tier2_currency", "tier2_equity",
)

# The 10 tickhistory-stage outputs build_synced_dataset reads (build.py:528-559). NOT
# the full list of files that function reads: it also reads a MANUAL prerequisite,
# DATA_ROOT/jkp/updated_daily_ind_gics_synced.csv (see USAGE.md), which
# is deliberately excluded here -- see sync_stage_outputs_ready()'s docstring.
# Deliberately NOT the same shape as SHARD_STEMS: us_equity and nonus_equity are
# separate outputs produced by two jobs sharing the one tier1_equity shard set.
SYNC_STAGE_OUTPUTS: tuple[tuple[str, str], ...] = (
    ("tier1", "traditional"), ("tier1", "commodity"), ("tier1", "currency"),
    ("tier1", "bond"), ("tier1", "nonus_equity"), ("tier1", "us_equity"),
    ("tier1", "volatility"), ("tier1", "stir"),
    ("tier2", "currency"), ("tier2", "equity"),
)

GATE1_MARKER = "_GATE1_OK"
_SIDES = ("trades", "quotes")
_SPLIT_HINT = "sbatch slurm/split_tickhistory.sh <trades-or-quotes-dir>"


@dataclass(frozen=True)
class Capability:
    ready: bool
    message: str | None      # None when ready, or when cleanly absent


def shard_dirs() -> list[Path]:
    return [TICKHISTORY_PATH / side / f"{stem}_{side}"
            for stem in SHARD_STEMS for side in _SIDES]


def _monolith_csvs() -> list[str]:
    return [p.name for side in _SIDES
            for p in sorted((TICKHISTORY_PATH / side).glob("*.csv"))]


def shards_ready() -> Capability:
    dirs = shard_dirs()
    present = [d for d in dirs if d.is_dir()]
    if not present:
        monoliths = _monolith_csvs()
        if monoliths:
            return Capability(False, (
                f"found {len(monoliths)} monolith CSV(s) under {TICKHISTORY_PATH} but no "
                f"shard directories; split them first: {_SPLIT_HINT}"))
        return Capability(False, None)          # clean researcher state, not a warning

    missing = [d.name for d in dirs if not d.is_dir()]
    if missing:
        message = "missing tick shard directories: " + ", ".join(sorted(missing))
        # The natural intermediate state: split_tickhistory.sh takes one side at a
        # time, so trades can be fully split while quotes is still bare monolith
        # CSVs (or vice versa). Without this, the researcher gets bare directory
        # names and no hint that a split is what closes the gap.
        monoliths = _monolith_csvs()
        if monoliths:
            message += (f"; {len(monoliths)} monolith CSV(s) still unsplit: "
                         + _SPLIT_HINT)
        return Capability(False, message)

    unverified = [d.name for d in dirs if not (d / GATE1_MARKER).exists()]
    if unverified:
        # Deliberately NOT the `tickhistory_shards verify` subcommand: it globs *.csv
        # monoliths out of the target dir and no-ops when there are none, and it never
        # writes the marker. Re-splitting (or re-fetching the bundle) is the real fix.
        return Capability(False, (
            f"tick shard directories present but unverified (no {GATE1_MARKER}): "
            + ", ".join(sorted(unverified))
            + f"; re-split from the source CSVs ({_SPLIT_HINT}) or re-download the bundle"))

    return Capability(True, None)


def _missing_files(paths: list[Path]) -> list[str]:
    return [p.as_posix() for p in paths if not p.exists()]


def sync_stage_outputs_ready() -> Capability:
    """True does not guarantee build_synced_dataset can run. That function also reads
    one file outside this predicate: DATA_ROOT/jkp/updated_daily_ind_gics_synced.csv
    a MANUAL prerequisite (see USAGE.md) that must be hand-placed and
    is not produced by any tickhistory stage. It is deliberately excluded here -- a
    missing manual prerequisite should crash loudly in full mode, not silently
    downgrade the machine to async-only."""
    paths = [DATASETS_ROOT / tier / "sync" / f"{cls}_daily_returns.csv"
             for tier, cls in SYNC_STAGE_OUTPUTS]
    missing = _missing_files(paths)
    if not missing:
        return Capability(True, None)
    if len(missing) == len(paths):
        return Capability(False, None)
    return Capability(False, "missing tickhistory stage outputs: " + ", ".join(missing))


def sync_panels_ready() -> Capability:
    paths = [DATASETS_ROOT / "tier1" / "sync" / "sync_daily.csv",
             DATASETS_ROOT / "tier2" / "sync" / "sync_daily.csv",
             DATASETS_ROOT / "tier2" / "sync" / "currency_daily_returns.csv"]
    missing = _missing_files(paths)
    if not missing:
        return Capability(True, None)
    if len(missing) == len(paths):
        return Capability(False, None)
    return Capability(False, "missing sync panels: " + ", ".join(missing))


# The pair compared per tier by sync_panels_fresh(). Deliberately just the two final
# aggregates, NOT sync_panels_ready()'s third path (tier2/.../currency_daily_returns.csv):
# that file is a tickhistory-stage OUTPUT (tickhistory.py:672, also tracked in
# SYNC_STAGE_OUTPUTS above), one that build.py only ever READS (build.py:547, :573) and
# never writes. As a stage *input*, it has no bearing on whether the sync and async
# *panels* came from the same build() call -- comparing it against a different stage's
# aggregate output is a category error, not a timing inconvenience. On a cluster run
# the tickhistory stage can finish hours or days before build runs, so its mtime lag
# against async_daily.csv is unbounded. Confirmed against the real datasets/ directory
# on this machine: the two sync_daily.csv/async_daily.csv pairs land within ~30ms of
# each other -- close enough that the comparison below uses st_mtime_ns, not the float
# st_mtime, to stay exact.
_STALENESS_TIERS: tuple[str, ...] = ("tier1", "tier2")


def sync_panels_fresh() -> Capability:
    """Existence is not the same as "from this build". async-only builds rewrite
    tier{1,2}/async/async_daily.csv on every run (build.py:659-660, 677-678) but skip
    tier{1,2}/sync/sync_daily.csv entirely (build.py:661-662, 679-680 are gated on
    mode == "full"). So a machine that once ran full mode and later runs async-only
    keeps its old sync panels on disk untouched: sync_panels_ready() still reports
    ready, but a subsequent full-mode validate would grade the FRESH async panels
    against those STALE sync panels -- a confident, wrong verdict for the one check
    that actually compares the two (consistency_check).

    Deterministic proxy for "same build": within a single tier, sync_daily.csv must be
    at least as new as async_daily.csv, since build.py always writes the async
    aggregate first and the sync aggregate immediately after (same function call, same
    mode). Call this only after sync_panels_ready() reports ready -- a missing file on
    either side means there is nothing meaningful to compare, so it reports fresh.
    """
    stale: list[str] = []
    for tier in _STALENESS_TIERS:
        sync_path = DATASETS_ROOT / tier / "sync" / "sync_daily.csv"
        async_path = DATASETS_ROOT / tier / "async" / "async_daily.csv"
        if not sync_path.exists() or not async_path.exists():
            continue          # nothing to compare; sync_panels_ready() covers absence
        if sync_path.stat().st_mtime_ns < async_path.stat().st_mtime_ns:
            stale.append(sync_path.as_posix())
    if not stale:
        return Capability(True, None)
    return Capability(False, (
        "sync panels predate their async counterparts, so they look like leftovers "
        "from an earlier full build rather than this one: " + ", ".join(stale)
        + ". Only `consistency` is compromised by this -- it is the one check that "
          "grades sync against async; every other full-mode check grades sync against "
          "third-party reference data, and the async-side checks don't read the sync "
          "panels at all. Rerun `globalmacro build --full` to rebuild both halves "
          "together, or pass --async-only to validate to run the rest."))


def resolve_mode(flag: str | None, cap: Capability, stage: str) -> str:
    """Explicit flag wins. `--full` is a demand, not a preference: it fails rather
    than silently falling back, which is the whole point of passing it."""
    if flag == "full":
        if not cap.ready:
            raise SystemExit(
                f"globalmacro {stage}: --full requires the sync inputs, which are not ready. "
                + (cap.message or "None were found."))
        return "full"
    if flag == "async-only":
        return "async-only"
    if flag is not None:
        raise ValueError(
            f"globalmacro {stage}: unrecognized mode {flag!r}; "
            'expected "full", "async-only", or unset for auto-detection')
    if cap.ready:
        return "full"
    if cap.message:
        # stderr, never stdout: run_all.sh's capability probe parses this function's
        # stdout (via a caller a layer up) to decide the mode for the whole DAG, so
        # noise here would corrupt that parse. A clean-absent capability (message is
        # None -- the researcher's normal state) stays silent: that is not a warning,
        # and noise there would train people to ignore it. Without this, a partial
        # state (some tickhistory stage outputs missing, e.g. a died job) resolves
        # async-only, prints nothing, and silently overwrites the async panels while
        # leaving the sync panels stale.
        print(f"globalmacro {stage}: sync inputs not ready -> async-only mode. "
              f"{cap.message}", file=sys.stderr)
    return "async-only"
