#!/bin/bash
#SBATCH --job-name=gm-split
#SBATCH --output=slurm/logs/split-%j.out
#SBATCH --error=slurm/logs/split-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=640G
#SBATCH --partition=cpunormal,build
# --mem=640G: the pilot measured split+verify at 35 GB for a 25 GB file (streaming Gate-1
# digest); the 417 GB equity file is ~584 GB worst-case (linear) and almost certainly less
# (the streaming footprint is largely fixed). 640 G covers it AND fits an available node
# without waiting for a full ~1 TB slot to free. Incremental single-month adds can override
# lower (`sbatch --mem=64G ...`).
# One-time TickHistory monolith -> per-month Parquet split (+ Gate-1 verify). Also the
# path for future incremental single-month adds. Idempotent/resumable: shard dirs that
# already carry a `_GATE1_OK` marker are skipped, so a re-run continues where it stopped.
#
# Usage: sbatch slurm/split_tickhistory.sh <monolith.csv | a trades|quotes dir> [--full-check] [--force]
#   A dir target splits every *.csv under it, sequentially, so peak memory is one file's
#   split (the 417 GB equity file peaks ~584 GB with the streaming Gate-1 digest < 1000 G).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
ulimit -n 4096 || true   # ~340 month-writers held open at once on the big files
TARGET="${1:?usage: split_tickhistory.sh <monolith.csv | trades|quotes dir> [--full-check] [--force]}"
shift
echo "[split $TARGET] $(date)"
python -m globalmacro.pipeline.tickhistory_shards split "$TARGET" "$@"
echo "[split $TARGET] complete $(date)"
