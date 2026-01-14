#!/bin/bash
#
#SBATCH --job-name=FuturesData        # Job name
#SBATCH --output=futures-%j.out       # File for standard output
#SBATCH --error=futures-%j.err        # File for standard error
#SBATCH --time=12:00:00               # Maximum runtime in HH:MM:SS
#SBATCH --ntasks=1                    # Number of tasks/processes
#SBATCH --cpus-per-task=20            # Number of CPU cores per task
#SBATCH --mem-per-cpu=32G             # Memory per CPU core
#SBATCH --partition=cpunormal         # The Slurm partition to use

export PYTHONUNBUFFERED=1
source ~/futures/.venv/bin/activate

echo "[Settlement Prices] Running futures.py"      
python3 futures.py --price_type settlement --ct &
python3 futures.py --price_type settlement

wait 

echo "[Open Prices] Running futures.py"      
python3 futures.py --price_type open --ct &
python3 futures.py --price_type open

wait

echo "Job complete"