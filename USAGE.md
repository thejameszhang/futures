# Usage & Execution

## CLI usage

The single entry point is `globalmacro <stage>`. List all available stages with:

```bash
globalmacro --help
```

which prints all 12 stages (alphabetically):
- **`build`**: Aggregates all asset classes into the final tier 1 and tier 2 datasets.
- **`connect`**: Sets up and securely saves your WRDS credentials. Honors `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables for non-interactive authentication, and accepts `--reset` to clear saved credentials.
- **`download`**: Pulls the required tables from the WRDS API.
- **`download-public`**: Fetches the public, no-credentials inputs (Ken French factors, FRED interbank/eurodollar rates, OECD short-term rates). Skip-if-present; pass `--force` to refresh.
- **`equities`**: Processes equity indices and U.S. equity sector data.
- **`futures`**: Processes futures data for commodities, bonds, and volatility indices.
- **`fx`**: Processes daily currency exchange rates from Datastream.
- **`instrumentlists`**: Generates the specific RIC pull-lists required to pull data from LSEG TickHistory.
- **`rates`**: Processes short-term interest rates and Eurodollar data.
- **`run`**: Submits the whole pipeline to SLURM as a dependency graph (see `slurm/run_all.sh`).
- **`tickhistory`**: Parses the high-frequency tick data for trades and quotes.
- **`validate`**: Runs data QA pipelines to generate validation PDFs and CSVs.

For detailed flags on any specific stage, run `globalmacro <stage> --help`.

**Run the whole pipeline:**
1. Run `globalmacro connect` and input your WRDS credentials. **IMPORTANT**: When starting the code, you may be prompted to grant access to WRDS using two-factor authentication, for example via a Duo notification.
2. `globalmacro run --with-download` (`--dry-run` to preview). It submits every stage to SLURM with the correct dependencies and parallelism (`rates → fx`, `equities`/`instrumentlists`/`futures` in parallel, `tickhistory` after `futures`, `build` after `equities`+`fx`+`tickhistory`, then `validate`). To run a single stage manually, `sbatch slurm/<stage>.sh` (from the repo root) or `globalmacro <stage>`.

Then run data QA with:

```bash
globalmacro validate
```

## USD-converted datasets

- Sync `_usd` panels convert G10-denominated assets using the TickHistory FX-futures
  return (blended into `fx_sync` by `utils/sync_fx.build_sync_fx_panel`); EM/non-G10 and
  all async panels keep Compustat/Datastream spot FX.

## Detailed Data Prerequisites

`build` must not be run until the required `data/` inputs are in place. They fall
into two classes: files `globalmacro download --database <db>` can pull from WRDS,
and files that must be acquired externally and pre-placed by hand.

### Downloadable via `globalmacro download --database <db>` (WRDS)

> **Note on WRDS Credentials:** Before downloading WRDS data or submitting `globalmacro run --with-download` to the cluster, you must first configure your WRDS credentials. You can run `globalmacro connect` interactively in your terminal, or supply credentials non-interactively via the `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables. This securely saves them in a `~/.pgpass` file in your home directory (keyring is disabled for HPC compute-node compatibility) so SLURM background jobs don't hang waiting for input. You can clear saved credentials anytime using `globalmacro connect --reset`.
>
> **WARNING (HPC Cluster Batch Execution):** If your credentials are incorrect, the `wrds` library will silently swallow the first connection failure and attempt to re-prompt for a password via `input()`. In a SLURM batch job (where there is no TTY), this causes a fatal `EOFError` crash that halts your pipeline. Always verify your credentials by running `globalmacro connect` before submitting batch jobs.

| Branch | Library | Files → location |
|---|---|---|
| `futures` | `tr_ds_fut` | `dsfutclass`, `dsfutcontr`, `dsfuttrdcycle`, `dsfutcontrinfo`, `dsfutcontrval` → `data/datastream/futures/` |
| `equities` | `tr_ds_equities` | `ds2indexdata` → `data/datastream/equities/` |
| `fx` | `tr_ds_equities` | `ds2fxrate`, `ds2fxcode` → `data/datastream/fx/` |
| `economics` | `tr_ds_econ` | `ecodata` → `data/datastream/economics/` |
| `comp` | `comp` | `exrt_dly` → `data/comp/` |

### Manual Prerequisite Data Files

In addition, some prerequisite files must be pre-placed manually in the `data/` folder (they cannot be downloaded automatically via `globalmacro download`):
- **Datastream Continuous Futures Benchmark** (for validation QA):
  - `datastream/futures/datastream_continuous_series.csv`
- **JKP**: `jkp/updated_daily_ind_gics.csv`, `jkp/updated_daily_ind_gics_synced.csv`
- **TickHistory** (LSEG TickHistory extractions; use `globalmacro instrumentlists` to generate the RIC pull-lists, then place the resulting CSVs):
  - `tickhistory/trades/`: `tier1_bond_trades.csv`, `tier1_commodity_trades.csv`, `tier1_currency_trades.csv`, `tier1_equity_trades.csv`, `tier1_stir_trades.csv`, `tier1_traditional_trades.csv`, `tier1_volatility_trades.csv`, `tier2_cryptocurrency_trades.csv`, `tier2_currency_trades.csv`, `tier2_equity_trades.csv`, `tier2_housing_trades.csv`
  - `tickhistory/quotes/`: `tier1_bond_quotes.csv`, `tier1_commodity_quotes.csv`, `tier1_currency_quotes.csv`, `tier1_equity_quotes.csv`, `tier1_stir_quotes.csv`, `tier1_traditional_quotes.csv`, `tier1_volatility_quotes.csv`, `tier2_cryptocurrency_quotes.csv`, `tier2_currency_quotes.csv`, `tier2_equity_quotes.csv`, `tier2_housing_quotes.csv`
