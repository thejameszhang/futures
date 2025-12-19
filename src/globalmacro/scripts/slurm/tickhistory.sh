#!/bin/bash
#
#SBATCH --job-name=TickHistory      # Job name
#SBATCH --output=slurm-%j.out       # File for standard output
#SBATCH --error=slurm-%j.err        # File for standard error
#SBATCH --time=1:00:00              # Maximum runtime in HH:MM:SS
#SBATCH --ntasks=1                  # Number of tasks/processes
#SBATCH --cpus-per-task=31          # Number of CPU cores per task
#SBATCH --mem-per-cpu=32G           # Memory per CPU core
#SBATCH --partition=build           # The Slurm partition to use

source ~/futures/.venv/bin/activate

echo "Running TickHistory for Commodities and Volatility Indices"
python3 tickhistory.py --asset_class commodity & 
python3 tickhistory.py --asset_class volatility

wait

echo "Running TickHistory for Currencies, STIRs, Bonds, and Traditional Assets"
python3 tickhistory.py --asset_class currency & 
python3 tickhistory.py --asset_class bond & 
python3 tickhistory.py --asset_class stir & 
python3 tickhistory.py --asset_class traditional

wait 

echo "Running TickHistory for Non-US Equities"
python3 tickhistory.py --asset_class nonus_equity

wait

echo "Running TickHistory for US Equities"
python3 tickhistory.py --asset_class us_equity

wait

echo "Job complete"