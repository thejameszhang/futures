#!/bin/bash
# run_all.sh — submit the full globalmacro pipeline as a SLURM dependency DAG.
# This script runs ON THE LOGIN NODE (not via sbatch); it submits the per-stage
# jobs with sbatch and exits. Usage: slurm/run_all.sh [--with-download] [--dry-run]
#   --with-download : also run the WRDS download stage (OFF by default; it
#                     overwrites raw data and needs credentials).
#   --dry-run       : print sbatch commands + deps, submit nothing.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
S=slurm
DRY=0; WITH_DL=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --with-download) WITH_DL=1 ;;
    -h|--help) echo "usage: run_all.sh [--with-download] [--dry-run]"; exit 0 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

# dry-run job-id counter lives in a temp file: submit() runs inside $(...), so a
# plain variable would only increment in the subshell (every id would be DRY1).
_SEQ=$(mktemp); echo 0 > "$_SEQ"; trap 'rm -f "$_SEQ"' EXIT
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

# --- optional download (5 WRDS branches, parallel); each consumer depends ONLY on
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
  for db in economics fx equities futures comp; do
    DLID[$db]=$(submit "" $S/download.sh --database $db)
  done
fi
# dld <branch...> -> colon-list of those download job ids (empty when not downloading)
dld() { local d=""; for b in "$@"; do d="$(join "$d" "${DLID[$b]:-}")"; done; echo "$d"; }

# --- level 1 (parallel): rates ; equities ; instrumentlists ; futures x4 ---
jR=$(submit "$(dld economics)" $S/rates.sh)              # reads economics download
jE=$(submit "$(dld equities)" $S/equities.sh)            # reads equities download; independent of rates/fx
jI=$(submit "" $S/instrumentlists.sh)                    # reads only repo config -> NO dependency
FUT=""
for pt in settlement open; do
  for ct in ct cs; do
    id=$(submit "$(dld futures)" $S/futures.sh $pt $ct); FUT="${FUT}:${id}"   # reads futures download
  done
done
FUT="${FUT#:}"

# --- fx (needs rates' 3month_libor_rates.csv, plus the fx/comp/futures downloads) ---
jFx=$(submit "$(join "$(dld fx comp futures)" "$jR")" $S/fx.sh)

# --- tickhistory x10 (after ALL 4 futures parquets); heavy classes bump memory ---
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

# --- build (needs equities + fx + all tickhistory; futures covered via tickhistory) ---
jB=$(submit "${jE}:${jFx}:${TICK}" $S/build.sh)
jV=$(submit "$jB" $S/validate.sh)

if [ "$WITH_DL" = 1 ]; then echo "downloads=[${DLID[*]}]"; else echo "downloads=[skipped]"; fi
echo "rates=$jR fx=$jFx equities=$jE instrumentlists=$jI"
echo "futures=[$FUT]"
echo "london_currency=[$jLON1:$jLON2]"
echo "tickhistory=[$TICK]"
echo "build=$jB validate=$jV"
