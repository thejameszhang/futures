# Usage & Execution

## CLI usage

The single entry point is `globalmacro <stage>`. List all available stages with:

```bash
globalmacro --help
```

which prints all 12 stages (alphabetically):
- **`build`**: Aggregates all asset classes into the final tier 1 and tier 2 datasets. Auto-detects async-only vs. full mode from tick data on disk; `--async-only`/`--full` override (see [Async-only mode](#async-only-mode)).
- **`connect`**: Sets up and securely saves your WRDS credentials. Honors `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables for non-interactive authentication, and accepts `--reset` to clear saved credentials. Also reports which datasets this machine can build (see [Async-only mode](#async-only-mode)); `--check-lseg` additionally validates LSEG DataScope credentials live.
- **`download`**: Pulls the required tables from the WRDS API.
- **`download-public`**: Fetches the public, no-credentials inputs (Ken French factors, FRED interbank/eurodollar rates, OECD short-term rates). Skip-if-present; pass `--force` to refresh.
- **`equities`**: Processes equity indices and U.S. equity sector data.
- **`futures`**: Processes futures data for commodities, bonds, and volatility indices.
- **`fx`**: Processes daily currency exchange rates from Datastream and Compustat (both, unconditionally, every run -- see [Async-only mode](#async-only-mode)).
- **`instrumentlists`**: Generates the specific RIC pull-lists required to pull data from LSEG TickHistory.
- **`rates`**: Processes short-term interest rates and Eurodollar data.
- **`run`**: Submits the whole pipeline to SLURM as a dependency graph (see `slurm/run_all.sh`). Auto-detects async-only vs. full mode the same way `build` and `validate` do.
- **`tickhistory`**: Parses the high-frequency tick data for trades and quotes.
- **`validate`**: Runs data QA pipelines to generate validation PDFs and CSVs. In async-only mode, skips the checks that require the sync panels and names each one explicitly rather than omitting it (see [Async-only mode](#async-only-mode)).

For detailed flags on any specific stage, run `globalmacro <stage> --help`.

**Run the whole pipeline:**
1. Run `globalmacro connect` and input your WRDS credentials. **IMPORTANT**: When starting the code, you may be prompted to grant access to WRDS using two-factor authentication, for example via a Duo notification. `connect` also prints a capability report — see [Async-only mode](#async-only-mode) — ending in the exact next command to run.
2. `globalmacro run --with-download` (`--dry-run` to preview). It submits every stage to SLURM with the correct dependencies and parallelism (`rates → fx`, `equities`/`instrumentlists`/`futures` in parallel, `tickhistory` after `futures` in full mode only, `build` after `equities`+`fx`+`tickhistory`, then `validate`). To run a single stage manually, `sbatch slurm/<stage>.sh` (from the repo root) or `globalmacro <stage>`. Running `globalmacro run` without `--with-download` submits the same DAG with every download stage skipped, so on a machine that has not already populated `data/`, every stage then fails on its missing raw inputs — `--with-download` is what actually pulls them first.

Then run data QA with:

```bash
globalmacro validate
```

## Async-only mode

A machine with no LSEG TickHistory data on disk automatically builds only the **async**
datasets — no flag needed. The gate is **tick data on disk, not credentials**: LSEG
DataScope (DSS) credentials do not unlock anything today, because no automated
tick-data download stage exists yet. `DSS_USERNAME`/`DSS_PASSWORD` are consumed only by
`globalmacro connect --check-lseg` and its capability report (below) — they do not gate
`instrumentlists` or `tickhistory`, which read only local files.

- **`globalmacro connect`** reports what this machine can build and prints the exact next
  command. Its last two lines are one of:
  - `-> This machine can build the SYNC and ASYNC datasets.`
  - `-> This machine can build the ASYNC datasets.` / `Sync datasets need LSEG tick data on disk.`

  followed by `Next:  globalmacro run --with-download`.
- **`globalmacro build`**, **`globalmacro validate`**, and **`globalmacro run`** (via
  `slurm/run_all.sh`) all auto-detect the mode from tick-shard presence, and all accept
  explicit `--async-only` / `--full` overrides. `--full` is a demand, not a preference: if
  the sync inputs aren't ready it fails fast with a message rather than silently degrading
  to an async-only build.
- In async-only mode, **`globalmacro validate`** skips the checks that require the sync
  panels (currently the async-vs-sync consistency check, the futures-vs-Compustat-spot
  check, and the optional local-only external ground-truth check) and **names each one
  explicitly**, printed as `[SKIP] <name> SKIPPED (async-only run)` and recorded the same
  way in `VALIDATION_SUMMARY.md`, along with any invariants/figures dropped from checks
  that still run partially. Nothing is silently omitted.
- Async-only mode does **not** remove the two prerequisites below (the JKP sector file and
  a Compustat entitlement) — both are still required to build the async datasets. See
  [Detailed Data Prerequisites](#detailed-data-prerequisites).

### `connect --check-lseg`

`globalmacro connect --check-lseg` makes a live network call to validate your LSEG
DataScope credentials (`DSS_USERNAME`/`DSS_PASSWORD`) against LSEG's token endpoint, on top
of `connect`'s normal WRDS check. Without `--check-lseg`, `connect` still reports whether
DSS credentials are *present* in the environment, just not whether LSEG accepts them.
Either way, LSEG/DSS status is informational only: it never affects `connect`'s exit code,
which depends solely on the WRDS connection check — a WRDS failure still exits 1
regardless of LSEG credential status.

## USD-converted datasets

- Sync `_usd` panels convert G10-denominated assets using the TickHistory FX-futures
  return (blended into `fx_sync` by `utils/sync_fx.build_sync_fx_panel`); EM/non-G10 and
  all async panels keep Compustat/Datastream spot FX.

## Detailed Data Prerequisites

`build` must not be run until the required `data/` inputs are in place. They fall
into two classes: files `globalmacro download --database <db>` can pull from WRDS,
and files that must be acquired externally and pre-placed by hand.

> **Relocating the data directory.** These inputs default to `<repo>/data/`, but you can store them elsewhere (e.g. the large TickHistory files on a scratch/volume) by setting `FUTURES_DATA_ROOT` in a repo-root `.env` — or override individual vendor paths (`TICKHISTORY_PATH`, `DATASTREAM_PATH`, `COMPUSTAT_PATH`) and the repo root (`FUTURES_ROOT`). All knobs are listed in `.env.example`.

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
| `datastream_continuous` | `tr_ds_fut` | `dsfutcalcserval` ⋈ `dsfutcalcserinfo` (6-col validation slice) → `data/datastream/futures/datastream_continuous_series.csv` |

> **Compustat is a prerequisite for every mode, not just sync.** `comp.exrt_dly` is pulled
> like any other WRDS table above, but it requires its own Compustat entitlement, separate
> from your Datastream license. The `fx` stage's `__main__` builds the sync FX panel from
> it unconditionally on every run, before anything mode-aware exists in that stage; `build`'s
> `load_synthetic_returns` then reads that panel's derived synthetic-FX output before
> `build`'s own async/full branch runs. A researcher with Datastream but not Compustat
> therefore cannot build the **async** datasets either — they hit an unexplained
> `FileNotFoundError` on `COMPUSTAT_PATH/exrt_dly.csv`, not a graceful async-only build.
> Async-only mode does not remove this dependency; get Compustat access alongside
> Datastream.

### Manual Prerequisite Data Files

In addition, some prerequisite files must be pre-placed manually in the `data/` folder (they cannot be downloaded automatically via `globalmacro download`):
- **JKP**: `jkp/updated_daily_ind_gics.csv`, `jkp/updated_daily_ind_gics_synced.csv`.
  `updated_daily_ind_gics.csv` feeds 11 US Select Sector columns into the **async** panel
  (`load_sectors_async`) unconditionally, so async-only mode does not remove this
  dependency either. A separate upstream `jkp-data` pull request will make it downloadable;
  until then it must be requested and placed by hand, same as TickHistory below.
- **TickHistory** (LSEG TickHistory extractions) — two steps:
  1. Use `globalmacro instrumentlists` to generate the RIC pull-lists, pull the CSVs from
     LSEG DataScope, and place them under `tickhistory/trades/` and `tickhistory/quotes/`:
     - `tickhistory/trades/`: `tier1_bond_trades.csv`, `tier1_commodity_trades.csv`, `tier1_currency_trades.csv`, `tier1_equity_trades.csv`, `tier1_stir_trades.csv`, `tier1_traditional_trades.csv`, `tier1_volatility_trades.csv`, `tier2_cryptocurrency_trades.csv`, `tier2_currency_trades.csv`, `tier2_equity_trades.csv`, `tier2_housing_trades.csv`
     - `tickhistory/quotes/`: `tier1_bond_quotes.csv`, `tier1_commodity_quotes.csv`, `tier1_currency_quotes.csv`, `tier1_equity_quotes.csv`, `tier1_stir_quotes.csv`, `tier1_traditional_quotes.csv`, `tier1_volatility_quotes.csv`, `tier2_cryptocurrency_quotes.csv`, `tier2_currency_quotes.csv`, `tier2_equity_quotes.csv`, `tier2_housing_quotes.csv`
  2. **Split each monolith CSV into per-month Parquet shards and verify it (Gate 1).** The
     pipeline no longer reads the monolith CSVs directly — since the Parquet-shard
     migration it reads per-month shards instead:
     ```bash
     sbatch slurm/split_tickhistory.sh tickhistory/trades   # splits every CSV under the dir
     sbatch slurm/split_tickhistory.sh tickhistory/quotes   # (or point it at one file at a time)
     ```
     This writes `tickhistory/{trades,quotes}/<stem>_{trades,quotes}/<YYYY-MM>.parquet` — one
     shard directory per source CSV, one Parquet file per month — and drops a `_GATE1_OK`
     marker file in each shard directory once its row count and per-(RIC, month) content hash
     match the source CSV. `globalmacro connect` and the pipeline check for these shard
     directories and their markers, not the CSVs — placing only the monolith CSVs from step 1
     reports as "not ready" until this step runs. Re-running the script is idempotent: it
     skips any shard directory that already carries the marker, so an incremental
     single-month add only reprocesses what changed.
