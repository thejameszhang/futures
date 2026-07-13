# Global Macroeconomic Data

## Overview

`globalmacro` produces time-synced and standard settlement-price returns datasets
for global macro futures — commodities, government bonds, short-term interest
rates, currencies, equity indices, U.S. equity sectors, and volatility indices. It
is the global-macro counterpart to the dataset behind Jensen, Kelly, and Pedersen
(2023). Every panel is published in two forms: local currency (the primary form)
and U.S. dollars, the latter carrying a `_usd` suffix. The two datasets take their
exchange rates from different vendors, matched to when each is measured — the
settlement-price (async) dataset uses Datastream's end-of-day rates, the time-synced
dataset uses Compustat's WM/Reuters 4pm London rates.

Follow this [link](https://www.overleaf.com/read/jrdvqrqmcwrt#d28fa4) for detailed documentation on the dataset.

## Install

```bash
uv sync
```

`uv sync` creates a `.venv/` folder (if one does not exist yet) and installs
`globalmacro` editable, along with every dependency pinned in `pyproject.toml` /
`uv.lock`. Activate the environment with `source .venv/bin/activate`, or run
commands through uv without activating using `uv run <command>`. The project is
already initialized — do not run `uv init`.

## CLI usage

The single entry point is `globalmacro <stage>`. List all available stages with:

```bash
globalmacro --help
```

which prints all 10 stages (alphabetically): `build`, `download`, `equities`,
`futures`, `fx`, `instrumentlists`, `rates`, `run`, `tickhistory`, `validate`.
`globalmacro run` submits the whole pipeline to SLURM as a dependency graph
(see `slurm/run_all.sh`).

**Run the whole pipeline:** `globalmacro run` (optionally `--with-download`;
`--dry-run` to preview). It submits every stage to SLURM with the correct
dependencies and parallelism (`rates → fx`, `equities`/`instrumentlists`/`futures`
in parallel, `tickhistory` after `futures`, `build` after
`equities`+`fx`+`tickhistory`, then `validate`). To run a single stage manually,
`sbatch slurm/<stage>.sh` (from the repo root) or `globalmacro <stage>`.

Then run data QA with:

```bash
globalmacro validate
```

`globalmacro instrumentlists` generates the per-(tier, asset-class) RIC pull-lists
under `data/tickhistory/instrumentlists/` — these specify which RICs to pull from
LSEG TickHistory, and must be generated before the TickHistory `trades`/`quotes`
prerequisite files can be pulled (see
[Data prerequisites](#data-prerequisites) below).

Each stage forwards its own flags unchanged, e.g.:

```bash
globalmacro futures --price_type settlement --ct
```

## Data layout

Two artifact directories live at the repo root and are gitignored:

- `data/` — raw vendor inputs (downloaded and pre-placed; see
  [Data prerequisites](#data-prerequisites))
- `datasets/` — produced returns datasets (see
  [Datasets produced](#datasets-produced))

Path overrides are set via a `.env` file (copy `.env.example` to get started). With
no `.env` present, all of the above resolve relative to the repo root (the
directory containing `pyproject.toml`).

## Data prerequisites

`build` must not be run until the required `data/` inputs are in place. They fall
into two classes: files `globalmacro download --database <db>` can pull from WRDS,
and files that must be acquired externally and pre-placed by hand.

### Downloadable via `globalmacro download --database <db>` (WRDS)

| Branch | Library | Files → location |
|---|---|---|
| `futures` | `tr_ds_fut` | `dsf*` (dsfutclass, dsfutcontr, dsfuttrdcycle, dsfutcontrinfo, dsfutcontrval) → `data/datastream/futures/` |
| `equities` | `tr_ds_equities` | `ds2indexdata`, `ds2equityindex` → `data/datastream/equities/` |
| `fx` | `tr_ds_equities` | `ds2fxrate`, `ds2fxcode`, `ds2mktval`, `ds2primqt*`, `ds2scdqt*` → `data/datastream/fx/` |
| `economics` | `tr_ds_econ` | `eco*` incl. `ecodata.csv` → `data/datastream/economics/` |
| `comp` | `comp` | `exrt_dly` → `data/comp/` |

### Manual Prerequisite Data Files

In addition, some prerequisite files must be pre-placed manually in the `data/` folder. More details to follow.

## Datasets produced

The final outputs are the tier1/tier2, sync/async aggregate datasets:

```
datasets/tier{1,2}/sync/sync_daily.csv
datasets/tier{1,2}/async/async_daily.csv
datasets/tier{1,2}/async/async_monthly.csv
```
