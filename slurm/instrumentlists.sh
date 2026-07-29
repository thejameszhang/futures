#!/bin/bash
#SBATCH --job-name=gm-instrumentlists
#SBATCH --output=slurm/logs/instrumentlists-%j.out
#SBATCH --error=slurm/logs/instrumentlists-%j.err
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=cpunormal,build
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
echo "[instrumentlists] $(date)"
globalmacro instrumentlists "$@"
echo "[instrumentlists] complete $(date)"
