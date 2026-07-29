# slurm/_common.sh — shared setup, sourced by every stage script.
# NOT executed directly. Callers set their own #SBATCH headers.
export PYTHONUNBUFFERED=1
# Repo root: honor FUTURES_ROOT, else derive from this file's own location
# (<repo>/slurm/_common.sh -> <repo>). No home-relative hardcode -> works from any clone.
_GM_REPO="${FUTURES_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$_GM_REPO/.venv/bin/activate"
set -euo pipefail
