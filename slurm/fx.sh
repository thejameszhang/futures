#!/bin/bash
#SBATCH --job-name=gm-fx
#SBATCH --output=slurm/logs/fx-%j.out
#SBATCH --error=slurm/logs/fx-%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --partition=cpunormal,build
source ~/futures/slurm/_common.sh || exit 1
echo "[fx] $(date)"
globalmacro fx "$@"
echo "[fx] complete $(date)"
