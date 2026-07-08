#!/bin/bash
#SBATCH --job-name=gm-download
#SBATCH --output=slurm/logs/download-%j.out
#SBATCH --error=slurm/logs/download-%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=cpunormal,build
source ~/futures/slurm/_common.sh || exit 1
echo "[download] $(date)"
globalmacro download "$@"
echo "[download] complete $(date)"
