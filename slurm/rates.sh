#!/bin/bash
#SBATCH --job-name=gm-rates
#SBATCH --output=slurm/logs/rates-%j.out
#SBATCH --error=slurm/logs/rates-%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=cpunormal,build
source ~/futures/slurm/_common.sh || exit 1
echo "[rates] $(date)"
globalmacro rates "$@"
echo "[rates] complete $(date)"
