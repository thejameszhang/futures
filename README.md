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

Before running the pipeline, required raw vendor inputs must be in place within the `data/` directory. Most are fetched automatically — via WRDS (`globalmacro download`, which also pulls the Datastream continuous-futures benchmark) or public HTTPS (`globalmacro download-public`, no credentials). Only the licensed LSEG TickHistory and JKP extractions must be acquired externally and pre-placed by hand. For exact file lists and locations, see [USAGE.md](USAGE.md).

**A machine with no LSEG TickHistory data still produces a usable dataset.** `globalmacro run --with-download` auto-detects whether tick data is on disk and, if not, builds only the async datasets — no flag needed (`--async-only`/`--full` exist as explicit overrides, and `--full` fails fast rather than silently degrading). Run `globalmacro connect` first: it reports which datasets this machine can build and prints the next command. This does **not** remove two other prerequisites, required in every mode: the JKP sector file (`jkp/updated_daily_ind_gics.csv`) and a **Compustat** entitlement (`comp.exrt_dly`) — both feed the async build too, not just the sync one. See [USAGE.md](USAGE.md) for why and where to place them.

**Relocating `data/`.** The `data/` directory (especially the large LSEG TickHistory files) does not have to live inside the repo. Set `FUTURES_DATA_ROOT` in a `.env` at the repo root (or export it) to point it at other storage; per-vendor paths (`TICKHISTORY_PATH`, `DATASTREAM_PATH`, `COMPUSTAT_PATH`) and the repo root itself (`FUTURES_ROOT`) are overridable too. See `.env.example` for the full list.

### Running the Pipeline

For comprehensive CLI usage, execution stages, async-only mode, and SLURM instructions, see [USAGE.md](USAGE.md).

## Outputs

```text
datasets/tier{1,2}/sync/sync_daily{_usd}.csv    # requires tick data (full mode)
datasets/tier{1,2}/async/async_monthly{_usd}.csv
```
