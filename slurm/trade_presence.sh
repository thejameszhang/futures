#!/bin/bash
#SBATCH --job-name=gm-tradepresence
#SBATCH --output=slurm/logs/tradepresence-%j.out
#SBATCH --error=slurm/logs/tradepresence-%j.err
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=1000G
# HEAVY classes {commodity, us_equity, nonus_equity} read the ~400GB trades shards and
# need more memory: submit with
#   sbatch --mem=1450G --cpus-per-task=32 slurm/trade_presence.sh <tier> <asset_class> <sync_target>
#SBATCH --partition=cpunormal,build
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/_common.sh" || exit 1
TIER="${1:?usage: trade_presence.sh <tier> <asset_class> <sync_target>}"
ASSET_CLASS="${2:?usage: trade_presence.sh <tier> <asset_class> <sync_target>}"
SYNC_TARGET="${3:?usage: trade_presence.sh <tier> <asset_class> <sync_target>}"
echo "[trade_presence tier$TIER $ASSET_CLASS sync=$SYNC_TARGET] $(date)"
.venv/bin/python scripts/build_trade_presence.py \
  --tier "$TIER" --asset_class "$ASSET_CLASS" --sync_target "$SYNC_TARGET"
echo "[trade_presence tier$TIER $ASSET_CLASS sync=$SYNC_TARGET] complete $(date)"
