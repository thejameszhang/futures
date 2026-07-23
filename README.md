# Global Macroeconomic Data

## Overview

This repository contains the code to generate returns datasets
for global macro futures. It
is the global-macro counterpart to the dataset behind Jensen, Kelly, and Pedersen
(2023). Every panel is published in two forms: local currency (the primary form)
and U.S. dollars, the latter carrying a `_usd` suffix.

Follow this [link](https://www.overleaf.com/read/jrdvqrqmcwrt#d28fa4) for detailed documentation on the methodology and datasets.

## Data Usage

This package requires valid vendor licenses. The authors do not distribute proprietary LSEG Datastream, TickHistory, WRDS, or Compustat data. This tool orchestrates the user's own licensed downloads and transformations. Outputs generated locally are derived from your licensed data and remain subject to your respective vendor license terms.

## Instructions

### Prerequisites

- Obtain your WRDS credentials. You can set them up by running `globalmacro connect` interactively in your terminal, or non-interactively by setting the `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables. See [USAGE.md](USAGE.md) for more details.
- Ensure you have [uv](https://docs.astral.sh/uv/) installed on your system.
- Path overrides are set via a `.env` file (copy `.env.example` to get started).

```bash
uv sync
```

`uv sync` creates a `.venv/` folder (if one does not exist yet) and installs
`globalmacro` editable, along with every dependency. Activate the environment with `source .venv/bin/activate`, or run
commands through uv without activating using `uv run <command>`.

### Data Prerequisites

Before running the pipeline, required raw vendor inputs must be in place within the `data/` directory. Some files can be pulled via WRDS, while others must be acquired externally and pre-placed by hand. For exact file lists and locations, please refer to [USAGE.md](USAGE.md).

### Running the Pipeline

For comprehensive CLI usage, execution stages, and SLURM instructions, see [USAGE.md](USAGE.md).

## Outputs

```text
datasets/tier{1,2}/sync/sync_daily{_usd}.csv
datasets/tier{1,2}/async/async_monthly{_usd}.csv
```
