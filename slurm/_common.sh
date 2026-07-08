# slurm/_common.sh — shared setup, sourced by every stage script.
# NOT executed directly. Callers set their own #SBATCH headers.
export PYTHONUNBUFFERED=1
source ~/futures/.venv/bin/activate
set -euo pipefail
