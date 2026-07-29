#!/bin/bash
#SBATCH --job-name=gm-tick
#SBATCH --output=slurm/logs/tick-%j.out
#SBATCH --error=slurm/logs/tick-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=1000G
#SBATCH --partition=cpunormal,build
# Partition is dynamic (SLURM picks cpunormal or build, whichever frees first).
# HEAVY classes {commodity, us_equity, nonus_equity} read the ~400GB trades files
# and need more memory: submit with `sbatch --mem=1450G --cpus-per-task=32 slurm/tickhistory.sh <class>`
# (run_all.sh does this automatically; partition stays dynamic).
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
ASSET_CLASS="${1:?usage: tickhistory.sh <asset_class> [tier] [sync_target]}"
TIER="${2:-1}"
SYNC_TARGET="${3:-et}"
echo "[tickhistory $ASSET_CLASS tier$TIER sync=$SYNC_TARGET] $(date)"
globalmacro tickhistory --asset_class "$ASSET_CLASS" --tier "$TIER" --sync_target "$SYNC_TARGET"
echo "[tickhistory $ASSET_CLASS tier$TIER sync=$SYNC_TARGET] complete $(date)"
