#!/bin/bash
# run_all.sh — submit the full globalmacro pipeline as a SLURM dependency DAG.
# This script runs ON THE LOGIN NODE (not via sbatch); it submits the per-stage
# jobs with sbatch and exits. Usage: slurm/run_all.sh [--with-download] [--async-only|--full] [--dry-run]
#   --with-download : also run the WRDS download stage (OFF by default; it
#                     overwrites raw data and needs credentials).
#   --async-only    : skip the tickhistory stage and build/validate async-only.
#   --full          : demand the sync inputs; fails fast if tick shards aren't ready.
#   --dry-run       : print sbatch commands + deps, submit nothing.
# With neither --async-only nor --full, the mode is auto-detected from tick shard
# presence on disk (see globalmacro.utils.capabilities.shards_ready).
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
S=slurm

# A's TRIVIAL 1: every temp file this script creates (the capability-probe stderr
# capture below, and the dry-run job-id counter further down) is cleaned up on ANY
# exit, including a `set -e` abort in between two mktemp calls -- not just the
# happy path. Installed here, before the FIRST mktemp, and appended to (never
# re-registered) by each: a second `trap ... EXIT` would silently REPLACE this one
# rather than add to it, un-covering whatever it already tracked.
_TMPFILES=()
trap 'rm -f "${_TMPFILES[@]}"' EXIT

DRY=0; WITH_DL=0; MODE=""; MODE_EXPLICIT=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --with-download) WITH_DL=1 ;;
    --async-only)
      [ "$MODE_EXPLICIT" = 1 ] && [ "$MODE" != async-only ] \
        && { echo "run_all.sh: --async-only conflicts with an earlier --full" >&2; exit 2; }
      MODE=async-only; MODE_EXPLICIT=1 ;;
    --full)
      [ "$MODE_EXPLICIT" = 1 ] && [ "$MODE" != full ] \
        && { echo "run_all.sh: --full conflicts with an earlier --async-only" >&2; exit 2; }
      MODE=full; MODE_EXPLICIT=1 ;;
    -h|--help) echo "usage: run_all.sh [--with-download] [--async-only|--full] [--dry-run]"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# Mode: an explicit flag wins; otherwise ask the capabilities module. Guarded, because
# run_all.sh must stay usable for --dry-run on a machine with no built venv.
if [ -z "$MODE" ]; then
  if [ -x .venv/bin/python ]; then
    # F7: capture the probe's stderr instead of discarding it (was 2>/dev/null) --
    # a broken/partial `uv sync`, a .venv built for another interpreter, or a
    # misresolved root previously yielded MODE=full with only "capability check
    # failed; assuming --full" and the actual import error thrown away. The
    # fail-toward-full direction stays: this only adds visibility into WHY.
    _CAPERR=$(mktemp); _TMPFILES+=("$_CAPERR")
    _CAP=$(.venv/bin/python -c "
from globalmacro.utils.capabilities import shards_ready
c = shards_ready()
print('full' if c.ready else 'async-only')
print(c.message or '')
" 2>"$_CAPERR") || _CAP=""
    if [ -z "$_CAP" ]; then
      MODE=full
      echo "[run_all] capability check failed; assuming --full" >&2
      if [ -s "$_CAPERR" ]; then
        echo "[run_all] capability check error (first 5 lines):" >&2
        head -5 "$_CAPERR" >&2
      fi
    else
      MODE=$(printf '%s\n' "$_CAP" | head -1)
      # Allowlist: `head -1` takes whatever line lands first, so a stray banner
      # ahead of the two `print`s (e.g. a library that logs on import) must not
      # silently become the mode -- that both degrades the DAG (a garbage $MODE
      # fails `[ "$MODE" = full ]`, so async-only wins by default) and, unquoted,
      # would word-split into extra sbatch args wherever $MODE is interpolated.
      case "$MODE" in
        full|async-only) ;;
        *) MODE=full
           echo "[run_all] unparseable capability output; assuming --full" >&2 ;;
      esac
      _MSG=$(printf '%s\n' "$_CAP" | sed -n 2p)
      [ -n "$_MSG" ] && echo "[run_all] $_MSG" >&2
    fi
    rm -f "$_CAPERR"
  else
    MODE=full
    echo "[run_all] no .venv/bin/python; assuming --full" >&2
  fi
fi
echo "mode=$MODE" >&2

# R2-1 (Opus review, both reviewers independently): mark whether $MODE was TYPED by
# the researcher or AUTO-detected from disk. Every `submit ... "--$MODE"` call below
# forwards the RESOLVED mode as an explicit flag either way (by design, Task 9) --
# so a bare argv (e.g. validate.sh's "$@") cannot tell an auto-detected async-only
# downgrade nobody asked for apart from a deliberate --async-only. Exporting here,
# before the first submit()/sbatch call, is enough for every job this script
# submits to see it: sbatch's default is --export=ALL ("All of the user's
# environment will be loaded" -- `man sbatch` on this cluster), i.e. it loads the
# SUBMITTING shell's environment (this script's), not the login shell's -- confirmed
# by reading the man page, not assumed. Consuming side: validation/run.py's
# explicit_async_only.
if [ "$MODE_EXPLICIT" = 0 ]; then
  export GM_MODE_AUTODETECTED=1
fi

# Refuse to silently downgrade a machine that has already built the sync half. Only
# reachable when the mode was AUTO-detected to async-only (MODE_EXPLICIT=0): an
# explicit --async-only means the researcher asked for this deliberately, and must
# proceed unchanged. Losing the shards but keeping stale sync artifacts on disk (a
# scratch purge, an unmounted TICKHISTORY_PATH, a split_tickhistory.sh job killed
# before it writes a _GATE1_OK marker) would otherwise submit the async-only DAG,
# skip all 12 tickhistory jobs, and leave the previous run's sync_daily.csv on disk
# now stale, with only one stderr line as the signal. Guarded the same defensive way
# as the capability probe above: `|| _X=""` on a crash, and an empty/unparseable
# result means "cannot tell" -- do NOT abort on it, a broken venv must not brick the
# DAG. Two predicates because build reads the tickhistory stage's raw outputs while
# validate reads the aggregated sync panels; either one being ready on disk is
# enough to mean "this machine already has sync artifacts".
if [ "$MODE" = async-only ] && [ "$MODE_EXPLICIT" = 0 ] && [ -x .venv/bin/python ]; then
  _SYNC_CHECK=$(.venv/bin/python -c "
from globalmacro.utils.capabilities import sync_panels_ready, sync_stage_outputs_ready
found = []
if sync_stage_outputs_ready().ready:
    found.append('tickhistory stage outputs under datasets/tier{1,2}/sync/')
if sync_panels_ready().ready:
    found.append('sync panels (datasets/tier{1,2}/sync/sync_daily.csv)')
print('yes' if found else 'no')
print('; '.join(found))
" 2>/dev/null) || _SYNC_CHECK=""
  if [ -n "$_SYNC_CHECK" ]; then
    _SYNC_FOUND=$(printf '%s\n' "$_SYNC_CHECK" | head -1)
    case "$_SYNC_FOUND" in
      yes)
        _SYNC_WHICH=$(printf '%s\n' "$_SYNC_CHECK" | sed -n 2p)
        {
          echo "run_all.sh: tick shards are not ready, but this machine already has sync"
          echo "artifacts on disk (${_SYNC_WHICH}). Refusing to silently downgrade to"
          echo "async-only and leave them stale. Either repair the tick shards, or pass"
          echo "--async-only explicitly to proceed."
        } >&2
        exit 1
        ;;
      *) ;;   # "no", or garbage -- cannot tell; do not abort
    esac
  fi
fi

# An *explicitly requested* --full fails fast if the sync inputs aren't ready. Gated on
# MODE_EXPLICIT so the auto-detect fallback above (capability check failed -> assume
# full) stays reachable: re-running the same import here would raise the same error.
if [ "$MODE" = full ] && [ "$MODE_EXPLICIT" = 1 ] && [ -x .venv/bin/python ]; then
  .venv/bin/python -c "
from globalmacro.utils.capabilities import resolve_mode, shards_ready
resolve_mode('full', shards_ready(), 'run')
" || { echo "[run_all] --full check failed (see error above)" >&2; exit 1; }
fi

# dry-run job-id counter lives in a temp file: submit() runs inside $(...), so a
# plain variable would only increment in the subshell (every id would be DRY1).
# Appended to the shared _TMPFILES cleanup trap installed at the top of this
# script, not a second independent trap (which would replace, not add to, it).
_SEQ=$(mktemp); echo 0 > "$_SEQ"; _TMPFILES+=("$_SEQ")
# submit <deps-colon-list-or-empty> <sbatch-args...> ; echoes a job id (real or fake)
submit() {
  local deps="$1"; shift
  local dep=""
  [ -n "$deps" ] && dep="--dependency=afterok:${deps}"
  if [ "$DRY" = 1 ]; then
    local n=$(( $(cat "$_SEQ") + 1 )); echo "$n" > "$_SEQ"
    echo "  sbatch --parsable $dep $*" >&2
    echo "DRY${n}"
  else
    sbatch --parsable $dep "$@"
  fi
}
# Partition is dynamic (cpunormal,build) inside futures.sh/tickhistory.sh; here we
# only bump memory/cpus for the heavy tickhistory classes.
HEAVY_MEM="--mem=1450G --cpus-per-task=32"

# join non-empty colon-lists into one afterok list
join() { local out=""; for x in "$@"; do [ -n "$x" ] && out="${out}:${x}"; done; echo "${out#:}"; }

# --- optional download (6 WRDS branches, parallel); each consumer depends ONLY on
#     the branch(es) it actually reads, so downloads don't serialize consumers ---
declare -A DLID
if [ "$WITH_DL" = 1 ]; then
  # Public HTTPS inputs (Ken French rf, FRED ded3, OECD stir) -- fetched INLINE on the
  # login node (which has egress; compute-node HTTPS egress is not guaranteed) BEFORE any
  # sbatch job is submitted, so under set -e a fetch failure aborts before ANYTHING is
  # queued (no orphaned WRDS jobs). The files land on disk pre-submission, so rates/build
  # find them with no afterok dependency needed. skip-if-present: a machine that already
  # has them does zero network.
  if [ "$DRY" = 1 ]; then
    echo "  .venv/bin/python -m globalmacro.pipeline.download_public" >&2
  else
    .venv/bin/python -m globalmacro.pipeline.download_public
  fi
  for db in economics fx equities futures comp datastream_continuous; do
    DLID[$db]=$(submit "" $S/download.sh --database $db)
  done
fi
# dld <branch...> -> colon-list of those download job ids (empty when not downloading)
dld() { local d=""; for b in "$@"; do d="$(join "$d" "${DLID[$b]:-}")"; done; echo "$d"; }

# --- level 1 (parallel): rates ; equities ; instrumentlists ; futures x4 ---
jR=$(submit "$(dld economics)" $S/rates.sh)              # reads economics download
jE=$(submit "$(dld equities)" $S/equities.sh)            # reads equities download; independent of rates/fx
jI=$(submit "" $S/instrumentlists.sh)                    # reads only repo config -> NO dependency
FUT=""; FUT_SETTLE=""
for pt in settlement open; do
  for ct in ct cs; do
    id=$(submit "$(dld futures)" $S/futures.sh $pt $ct); FUT="${FUT}:${id}"   # reads futures download
    [ "$pt" = settlement ] && FUT_SETTLE="${FUT_SETTLE}:${id}"
  done
done
FUT="${FUT#:}"; FUT_SETTLE="${FUT_SETTLE#:}"

# --- fx (needs rates' 3month_libor_rates.csv, plus the fx/comp/futures downloads) ---
jFx=$(submit "$(join "$(dld fx comp futures)" "$jR")" $S/fx.sh)

# --- tickhistory x12 (after ALL 4 futures parquets); heavy classes bump memory ---
# Skipped entirely in async-only mode: no tick shards means nothing for these jobs to
# read, and researchers without LSEG tick data should never wait on (or be blocked by)
# a stage they can't run.
jLON1=""; jLON2=""; TICK=""
if [ "$MODE" = full ]; then
  # London currency runs (both tiers) go first: they write the same shared debug path
  # (data/tickhistory/debug/{tables,dates}/<sym>.csv, no sync_target in the path) as the
  # ET currency runs, so London->ET is serialized to avoid a write race and to leave the
  # ET (shipping) debug tables on disk afterward. validate's fx-vs-spot exercise also
  # needs the sync_london currency panels these produce.
  jLON1=$(submit "$FUT" $S/tickhistory.sh currency 1 london)
  jLON2=$(submit "$FUT" $S/tickhistory.sh currency 2 london)

  TICK="${jLON1}:${jLON2}"
  for c in commodity currency bond nonus_equity stir traditional us_equity volatility; do
    case "$c" in commodity|us_equity|nonus_equity) R="$HEAVY_MEM" ;; *) R="" ;; esac
    dep="$FUT"
    [ "$c" = currency ] && dep="$(join "$FUT" "$jLON1")"
    id=$(submit "$dep" $R $S/tickhistory.sh $c); TICK="${TICK}:${id}"
  done
  for c in currency equity; do
    dep="$FUT"
    [ "$c" = currency ] && dep="$(join "$FUT" "$jLON2")"
    id=$(submit "$dep" $S/tickhistory.sh $c 2); TICK="${TICK}:${id}"   # tier 2, standard mem
  done
  TICK="${TICK#:}"
fi

# --- build (full: needs equities + fx + all tickhistory, futures covered via
#     tickhistory; async-only: TICK is empty, so depend on the settlement futures
#     pair directly -- futures.sh requests 24h and the two `open` jobs produce
#     nothing an async-only run reads) ---
if [ "$MODE" = full ]; then
  jB=$(submit "${jE}:${jFx}:${TICK}" $S/build.sh --full)
else
  jB=$(submit "$(join "$jE" "$jFx" "$FUT_SETTLE")" $S/build.sh --async-only)
fi
jV=$(submit "$(join "$jB" "$(dld datastream_continuous)")" $S/validate.sh "--$MODE")   # +datastream_continuous benchmark

if [ "$WITH_DL" = 1 ]; then echo "downloads=[${DLID[*]}]"; else echo "downloads=[skipped]"; fi
echo "rates=$jR fx=$jFx equities=$jE instrumentlists=$jI"
echo "futures=[$FUT]"
if [ -n "$jLON1" ]; then echo "london_currency=[$jLON1:$jLON2]"; fi
echo "tickhistory=[$TICK]"
echo "build=$jB validate=$jV"
