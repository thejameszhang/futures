# Global Macro Module

This package houses the pipelines that build the Global Macroeconomic futures data used throughout the project.

## Layout
- `scripts/`: Python entry points (e.g., `futures.py`) that ingest Datastream files, construct continuous series, splice contracts, and write tiered datasets.
- `slurm/`: Batch wrappers and logs for running the scripts on the cluster.

## Running the futures pipeline

Running this code produces returns datasets and characteristics for both the asynchronous and synchronous datasets. It does not build portfolios.

1. Install dependencies and activate the repo virtual environment (e.g., `uv sync && source .venv/bin/activate`).
2. Ensure Datastream and TickHistory CSV dumps exist under `data/datastream/futures/lseg/` and `data/tickhistory/{trades,quotes}/` and the tier configurations (`tier1.yaml`, `tier2.yaml`) are in the repo root.
3. Execute a job, e.g.:
   ```bash
   sbatch slurm/futures.sh
   ```
   This produces CSV/Parquet files in `datasets/tier*/async` and `characteristics/tier*/async`. For synchronous datasets, run 
   ```bash
   sbatch slurm/tickhistory.sh
   ```
   This produces CSV output files in `datasets/tier*/sync` and `characteristics/tier*/sync`.
