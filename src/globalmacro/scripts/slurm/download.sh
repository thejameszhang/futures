#!/bin/bash
#
#SBATCH --job-name=DownloadDatastream      # Job name
#SBATCH --output=slurm-%j.out       # File for standard output
#SBATCH --error=slurm-%j.err        # File for standard error
#SBATCH --time=15:00:00             # Maximum runtime in HH:MM:SS
#SBATCH --ntasks=1                  # Number of tasks/processes
#SBATCH --cpus-per-task=32          # Number of CPU cores per task
#SBATCH --mem-per-cpu=45G           # Memory per CPU core
#SBATCH --partition=build           # The Slurm partition to use

export PYTHONUNBUFFERED=1
source ~/futures/.venv/bin/activate

echo "[download.py] Equities"
python3 download.py --database equities

echo "[download.py] Futures"
python3 download.py --database futures

echo "[download.py] Economics"
python3 download.py --database economics

echo "[download.py] Commodities"
python3 download.py --database commodities

echo "[download.py] Compustat Daily Exchange Rates"
python3 download.py --database comp

echo "Job complete"