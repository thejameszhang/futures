#!/bin/bash
#SBATCH --job-name=gm-equities
#SBATCH --output=slurm/logs/equities-%j.out
#SBATCH --error=slurm/logs/equities-%j.err
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --partition=cpunormal,build
source ~/futures/slurm/_common.sh || exit 1
echo "[equities] $(date)"
globalmacro equities "$@"
echo "[equities] complete $(date)"
