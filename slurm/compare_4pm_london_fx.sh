#!/bin/bash
#SBATCH --job-name=gm-fx-london
#SBATCH --output=slurm/logs/fx-london-%j.out
#SBATCH --error=slurm/logs/fx-london-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=cpunormal,build

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
SYNC_TARGET="${1:-both}"
echo "[compare_4pm_london_fx] $(date)"
python scripts/compare_4pm_london_fx.py --sync_target "$SYNC_TARGET"
echo "[compare_4pm_london_fx] complete $(date)"
