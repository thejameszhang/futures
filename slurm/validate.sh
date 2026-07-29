#!/bin/bash
#SBATCH --job-name=gm-validate
#SBATCH --output=slurm/logs/validate-%j.out
#SBATCH --error=slurm/logs/validate-%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=cpunormal,build
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
echo "[validate] $(date)"
globalmacro validate "$@"
echo "[validate] complete $(date)"
