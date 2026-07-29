#!/usr/bin/env bash
# Scaffold the isolated verification tree ~/futures_verify and symlink the
# raw prerequisite inputs from the real repo. Idempotent. Prints the
# env exports needed to run the pipeline against this tree.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
V="$HOME/futures_verify"
RD="$REPO/data"

mkdir -p "$V"/{datasets,validation}
mkdir -p "$V"/data/datastream/{futures,equities,fx,economics}
mkdir -p "$V"/data/comp "$V"/data/tickhistory

link() {  # link <real-relpath> <verify-relpath>
  local src="$RD/$1" dst="$V/data/$2"
  [ -e "$src" ] || { echo "MISSING prereq: $src" >&2; exit 1; }
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
}
# Tick pulls: symlink the two subdirs only (instrumentlists regenerates in-tree).
link tickhistory/trades tickhistory/trades
link tickhistory/quotes tickhistory/quotes
# Whole read-only prereq dirs.
link jkp jkp
link misc misc
# Files that share a directory with download outputs -> symlink individually.
link datastream/economics/ded3_wrds.csv datastream/economics/ded3_wrds.csv
link datastream/economics/oecd.csv      datastream/economics/oecd.csv
link datastream/futures/datastream_continuous_series.csv datastream/futures/datastream_continuous_series.csv

echo "scaffold ready at $V"
echo "--- to isolate a shell, export: ---"
echo "export FUTURES_DATA_ROOT=$V/data"
echo "export FUTURES_DATASETS_ROOT=$V/datasets"
echo "export FUTURES_VALIDATION_OUTPUT=$V/validation"
