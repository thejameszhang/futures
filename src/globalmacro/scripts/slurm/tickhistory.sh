#!/bin/bash
#
#SBATCH --job-name=TickHistory      # Job name
#SBATCH --output=slurm-%j.out       # File for standard output
#SBATCH --error=slurm-%j.err        # File for standard error
#SBATCH --time=1:00:00              # Maximum runtime in HH:MM:SS
#SBATCH --ntasks=1                  # Number of tasks/processes
#SBATCH --cpus-per-task=32          # Number of CPU cores per task
#SBATCH --mem-per-cpu=45G           # Memory per CPU core
#SBATCH --partition=build           # The Slurm partition to use

source ~/futures/.venv/bin/activate

echo "Running TickHistory for Commodities"
python3 tickhistory.py --asset_class commodity

wait

echo "Running TickHistory for Sectors, Currencies, STIRs, and Bonds"
python3 tickhistory.py --asset_class sector &
python3 tickhistory.py --asset_class currency & 
python3 tickhistory.py --asset_class bond & 
python3 tickhistory.py --asset_class stir

wait

echo "Running TickHistory for Volatility Indices, Historical, and Traditional Assets"
python3 tickhistory.py --asset_class volatility &
python3 tickhistory.py --asset_class historical &
python3 trad.py --asset_class traditional

wait 

echo "Running TickHistory for Non-US Equities"
python3 tickhistory.py --asset_class nonus_equity

wait

echo "Running TickHistory for US Equities"
python3 tickhistory.py --asset_class us_equity

wait

echo "Job complete"