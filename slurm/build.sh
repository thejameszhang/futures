#!/bin/bash
#SBATCH --job-name=gm-build
#SBATCH --output=slurm/logs/build-%j.out
#SBATCH --error=slurm/logs/build-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --partition=cpunormal,build
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
echo "[build] $(date)"
globalmacro build "$@"
echo "[build] complete $(date)"
