# globalmacro

## Overview

`globalmacro` produces **tier1 and tier2 returns datasets** (sync and async) for
global macro futures — commodities, bonds, currencies, and equity indices — analogous
to the code behind "Is There a Replication Crisis in Finance?" (Jensen, Kelly, and
Pedersen, *Journal of Finance*, 2023), but for global macro assets.

Follow this [link](https://www.overleaf.com/read/jrdvqrqmcwrt#d28fa4) for detailed
documentation on the dataset.

Returns are in **local currency for non-FX assets**, with two by-construction
exceptions: the currency asset class carries FX (USD-base) returns, and spliced
synthetic-equity returns are net of a US risk-free rate. See
[Known residuals / currency basis](#known-residuals--currency-basis) below. Each
asset has a single return series (no `_local` suffix disambiguation). There is no
analysis component in this repository — `globalmacro` is a dataset producer only.

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

then run data QA with:

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

Three artifact directories live at the repo root and are gitignored:

- `data/` — raw vendor inputs (downloaded and pre-placed; see
  [Data prerequisites](#data-prerequisites))
- `datasets/` — produced returns datasets (see
  [Datasets produced](#datasets-produced))
- `characteristics/` — out of scope for this document

Path overrides are set via a `.env` file (copy `.env.example` to get started). With
no `.env` present, all of the above resolve relative to the repo root (the
directory containing `pyproject.toml`).

`data/` was reorganized to hold only the files the pipeline actually reads
(prerequisites, downloader outputs, and regenerable interims). Orphaned/scratch
inputs no longer read by any producer stage were moved to a gitignored
`old-data/` directory, and inputs used only by out-of-package analysis code
were moved into `analysis/` — nothing was deleted. The shareable data bundle
(`scripts/package_dropbox.py`) ships only raw inputs and pre-placed
prerequisites; regenerable interim files are excluded and are recreated
automatically on a full pipeline run.

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

### Prerequisites — must be pre-placed (no downloader; acquired externally)

| File(s) | Source | Location |
|---|---|---|
| `trades/`, `quotes/` tick pulls | LSEG TickHistory (via the `instrumentlists` RIC lists) | `data/tickhistory/{trades,quotes}/` |
| `updated_daily_ind_gics*.csv` | JKP | `data/jkp/` |
| `F-F_Research_Data_Factors_daily.csv` | Ken French Data Library | `data/misc/` |
| `VIX_History.csv` | Cboe | `data/misc/` |
| `oecd.csv` | OECD | `data/datastream/economics/` |
| `ded3_wrds.csv` | WRDS (manual pull; not a `download.py` branch) — read by `rates.py` (→ `3month_libor_rates.csv`), whose `ded3` column `fx.py` then uses as the USD funding rate | `data/datastream/economics/` |

## Datasets produced

The final outputs are the tier1/tier2, sync/async aggregate datasets:

```
datasets/tier{1,2}/sync/sync_daily.csv
datasets/tier{1,2}/async/async_daily.csv
datasets/tier{1,2}/async/async_monthly.csv
```

Each is a single return series per asset, in local currency for non-FX assets (see
[Known residuals / currency basis](#known-residuals--currency-basis)).

`build` consumes several per-asset-class intermediate returns files to produce
these aggregates, including:

- async: `daily_ret_1_{CT,CS}.csv` (the front-contract series `build` reads; `futures` also produces a second-contract `daily_ret_2_{CT,CS}.csv` that `build` does not consume)
- sync: `tier{1,2}/sync/{assetclass}_daily_returns.csv` — tier1: `commodity_daily_returns.csv`, `bond_daily_returns.csv`, `currency_daily_returns.csv`, `traditional_daily_returns.csv`, `us_equity_daily_returns.csv`, `nonus_equity_daily_returns.csv`, `volatility_daily_returns.csv`, `stir_daily_returns.csv`; tier2: `currency_daily_returns.csv`, `equity_daily_returns.csv`
- `spot_equity_returns.csv`, `synthetic_fx_returns.csv`

## Known residuals / currency basis

Non-FX asset returns are in **local currency** — the USD conversion step was
removed from every path. The final aggregate dataset nonetheless blends in two
by-construction exceptions:

1. **Currency asset class** (`synthetic_fx_returns`) carries **FX returns**
   (USD-base) — a currency has no "local" return.
2. **Synthetic spliced-equity** returns are net of a **US** risk-free rate
   (`local_equity − US_rf`, in `build.py`) — an accepted known residual (decision
   D22), not a US-dollar conversion of the underlying equity return itself.

The final dataset is therefore a **local + FX decomposition, not a single uniform
numeraire.**
