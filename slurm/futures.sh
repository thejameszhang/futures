#!/bin/bash
#SBATCH --job-name=gm-futures
#SBATCH --output=slurm/logs/futures-%j.out
#SBATCH --error=slurm/logs/futures-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=640G
#SBATCH --partition=cpunormal,build
source ~/futures/slurm/_common.sh || exit 1
PRICE_TYPE="${1:?usage: futures.sh <settlement|open> <ct|cs>}"
CONTRACT="${2:?usage: futures.sh <settlement|open> <ct|cs>}"
case "$PRICE_TYPE" in settlement|open) ;; *) echo "bad price_type: $PRICE_TYPE" >&2; exit 2 ;; esac
CT_FLAG=""
case "$CONTRACT" in ct) CT_FLAG="--ct" ;; cs) CT_FLAG="" ;; *) echo "bad contract: $CONTRACT" >&2; exit 2 ;; esac
echo "[futures $PRICE_TYPE $CONTRACT] $(date)"
globalmacro futures --price_type "$PRICE_TYPE" $CT_FLAG
echo "[futures $PRICE_TYPE $CONTRACT] complete $(date)"
