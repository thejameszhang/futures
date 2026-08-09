# Usage & Execution

## CLI usage

The single entry point is `globalmacro <stage>`. List all available stages with:

```bash
globalmacro --help
```

which prints all 12 stages (alphabetically):
- **`build`**: Aggregates all asset classes into the final tier 1 and tier 2 datasets. Run standalone, auto-detects async-only vs. full mode from the tickhistory stage's own outputs on disk (not tick-shard presence — see [Async-only mode](#async-only-mode)); `--async-only`/`--full` override. Under `globalmacro run`, it instead receives the mode `run` already resolved, rather than detecting it itself.
- **`connect`**: Sets up and securely saves your WRDS credentials. Honors `WRDS_USERNAME` and `WRDS_PASSWORD` environment variables for non-interactive authentication, and accepts `--reset` to clear saved credentials. Also reports which datasets this machine can build (see [Async-only mode](#async-only-mode)); `--check-lseg` additionally validates LSEG DataScope credentials live.
- **`download`**: Pulls the required tables from the WRDS API.
- **`download-public`**: Fetches the public, no-credentials inputs (Ken French factors, FRED interbank/eurodollar rates, OECD short-term rates). Skip-if-present; pass `--force` to refresh.
- **`equities`**: Processes equity indices and U.S. equity sector data.
- **`futures`**: Processes futures data for commodities, bonds, and volatility indices.
- **`fx`**: Processes daily currency exchange rates from Datastream and Compustat (both, unconditionally, every run -- see [Async-only mode](#async-only-mode)).
- **`instrumentlists`**: Generates the specific RIC pull-lists required to pull data from LSEG TickHistory.
- **`rates`**: Processes short-term interest rates and Eurodollar data.
- **`run`**: Submits the whole pipeline to SLURM as a dependency graph (see `slurm/run_all.sh`). Auto-detects async-only vs. full mode from tick-shard presence, then passes that resolved mode down to `build` and `validate` as an explicit `--async-only`/`--full` flag — they do not re-detect it themselves (see [Async-only mode](#async-only-mode)).
- **`tickhistory`**: Parses the high-frequency tick data for trades and quotes.
- **`validate`**: Runs data QA pipelines to generate validation PDFs and CSVs. In async-only mode, skips the checks that require the sync panels and names each one explicitly rather than omitting it (see [Async-only mode](#async-only-mode)).

For detailed flags on any specific stage, run `globalmacro <stage> --help`.

**Run the whole pipeline:**
1. Run `globalmacro connect` and input your WRDS credentials. **IMPORTANT**: When starting the code, you may be prompted to grant access to WRDS using two-factor authentication, for example via a Duo notification. `connect` also prints a capability report — see [Async-only mode](#async-only-mode) — ending in the exact next command to run.
2. `globalmacro run --with-download` (`--dry-run` to preview). It submits every stage to SLURM with the correct dependencies and parallelism (`rates → fx`, `equities`/`instrumentlists`/`futures` in parallel, `tickhistory` after `futures` in full mode only, `build` after `equities`+`fx`+`tickhistory`, then `validate`). To run a single stage manually, `sbatch slurm/<stage>.sh` (from the repo root) or `globalmacro <stage>`. Running `globalmacro run` without `--with-download` submits the same DAG with every download stage skipped, so on a machine that has not already populated `data/`, every stage that needs raw vendor input then fails on its missing files — `--with-download` is what actually pulls them first. (`instrumentlists` is the one exception: it reads only the repo's own `tier1.yaml`/`tier2.yaml`, not `data/`, so it succeeds with an empty `data/` even here.)

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
  command. It checks exactly one file up front — the JKP sector file
  (`data/jkp/updated_daily_ind_gics.csv`), a genuine, manual, never-downloadable
  prerequisite for both modes — never a Compustat entitlement (no file on disk can prove
  one before the first download; see "Compustat is a prerequisite for every mode" below),
  so it only ever reminds the researcher of that one, never claims to have verified it.

  When the JKP file is present and shards are ready, it ends with three lines:
  - `-> This machine can build the SYNC and ASYNC datasets.`
  - `   Also requires a Compustat entitlement (comp.exrt_dly) -- see USAGE.md.`
  - `Next:  globalmacro run --with-download`

  When the JKP file is present but shards are not ready, it ends with four lines instead:
  - `-> This machine can build the ASYNC datasets.`
  - `   Sync datasets need LSEG tick data on disk.`
  - `   Also requires a Compustat entitlement (comp.exrt_dly) -- see USAGE.md.`
  - `Next:  globalmacro run --with-download`

  When the JKP file is missing — regardless of shard state, since it gates
  `load_sectors_async()`, which `build` calls unconditionally in every mode — it ends with
  a fourth shape instead, naming the missing file:
  - `-> This machine needs one more file before it can build the ASYNC datasets:`
  - `   <path to the missing file>`
  - `   See USAGE.md's Detailed Data Prerequisites section.`
  - `Next:  globalmacro run --with-download`
- **Mode detection is not one predicate — each entry point asks a different question**
  (`src/globalmacro/utils/capabilities.py`: "shard presence is not build-readiness").
  `globalmacro run` (via `slurm/run_all.sh`) keys on **tick-shard presence**
  (`shards_ready()`), then hands the mode it resolves down to `build` and `validate` as an
  explicit `--async-only`/`--full` flag — they do not re-detect it in that path. Run
  **standalone**, `globalmacro build` instead keys on the **tickhistory stage's own
  outputs** on disk (`sync_stage_outputs_ready()`: the 10
  `datasets/tier{1,2}/sync/{class}_daily_returns.csv` files), and `globalmacro validate`
  keys on the **shipped sync panels** (`sync_panels_ready()`: the 3 final sync panels,
  `tier{1,2}/sync/sync_daily.csv` and `tier2/sync/currency_daily_returns.csv`). The three
  predicates can disagree: shards present but the `tickhistory` stage not yet run makes
  standalone `build` resolve async-only despite shards being on disk, and shards deleted
  after a full run still leave `build` resolving full because the stage outputs remain on
  disk. All three stages accept explicit `--async-only` / `--full` overrides. `--full` is a
  demand, not a preference: if the relevant inputs for that stage aren't ready it fails
  fast with a message rather than silently degrading to an async-only build.
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
> therefore cannot build the **async** datasets either — but which symptom they see
> depends on how they run it. Running `globalmacro fx` standalone, they hit an unexplained
> `FileNotFoundError` on `COMPUSTAT_PATH/exrt_dly.csv`, not a graceful async-only build. On
> the documented first-run path (`globalmacro run --with-download`), they instead see a
> stalled DAG: the `comp` download job fails first (no Compustat entitlement to query it
> with), and because `fx.sh` depends on it via SLURM's `afterok`, `fx` — and everything
> downstream of it — never launches at all, rather than printing that `FileNotFoundError`.
> Async-only mode does not remove this dependency; get Compustat access alongside
> Datastream.

### Manual Prerequisite Data Files

In addition, some prerequisite files must be pre-placed manually in the `data/` folder (they cannot be downloaded automatically via `globalmacro download`):
- **JKP**: `jkp/updated_daily_ind_gics.csv`, `jkp/updated_daily_ind_gics_synced.csv`.
  `updated_daily_ind_gics.csv` feeds 11 US Select Sector columns into the **async** panel
  (`load_sectors_async`) unconditionally, so async-only mode does not remove this
  dependency either. `updated_daily_ind_gics_synced.csv`, by contrast, is read only by the
  sync path (`build_synced_dataset`) — an async-only researcher does not need to request
  it. A separate upstream `jkp-data` pull request will make the async file downloadable;
  until then it must be requested and placed by hand, same as TickHistory below.
- **TickHistory** (LSEG TickHistory extractions) — two steps:
  1. Use `globalmacro instrumentlists` to generate the RIC pull-lists, pull the CSVs from
     LSEG DataScope, and place them under `data/tickhistory/trades/` and
     `data/tickhistory/quotes/` (or `$TICKHISTORY_PATH/{trades,quotes}/` if you've
     relocated the data directory — see "Relocating the data directory" above):
     - `data/tickhistory/trades/`: `tier1_bond_trades.csv`, `tier1_commodity_trades.csv`, `tier1_currency_trades.csv`, `tier1_equity_trades.csv`, `tier1_stir_trades.csv`, `tier1_traditional_trades.csv`, `tier1_volatility_trades.csv`, `tier2_cryptocurrency_trades.csv`, `tier2_currency_trades.csv`, `tier2_equity_trades.csv`, `tier2_housing_trades.csv`
     - `data/tickhistory/quotes/`: `tier1_bond_quotes.csv`, `tier1_commodity_quotes.csv`, `tier1_currency_quotes.csv`, `tier1_equity_quotes.csv`, `tier1_stir_quotes.csv`, `tier1_traditional_quotes.csv`, `tier1_volatility_quotes.csv`, `tier2_cryptocurrency_quotes.csv`, `tier2_currency_quotes.csv`, `tier2_equity_quotes.csv`, `tier2_housing_quotes.csv`
  2. **Split each monolith CSV into per-month Parquet shards and verify it (Gate 1).** The
     pipeline no longer reads the monolith CSVs directly — since the Parquet-shard
     migration it reads per-month shards instead. Run this from the repo root: neither
     this script nor `slurm/_common.sh` change directories, so the target below resolves
     relative to wherever you invoke `sbatch` from, not relative to `data/`.
     ```bash
     sbatch slurm/split_tickhistory.sh data/tickhistory/trades   # splits every CSV under the dir
     sbatch slurm/split_tickhistory.sh data/tickhistory/quotes   # (or point it at one file at a time)
     ```
     This writes `data/tickhistory/{trades,quotes}/<stem>_{trades,quotes}/<YYYY-MM>.parquet`
     — one shard directory per source CSV, one Parquet file per month — and drops a
     `_GATE1_OK` marker file in each shard directory once its row count and per-(RIC,
     month) content hash match the source CSV. `globalmacro connect` and the pipeline
     check for these shard directories and their markers, not the CSVs — placing only the
     monolith CSVs from step 1 reports as "not ready" until this step runs. Re-running the
     script is idempotent at whole-shard-directory granularity: it skips any shard
     directory that already carries the marker. That means appending a new month to an
     already-split monolith is **not** picked up by a bare re-run — the existing marker
     causes the whole file to be skipped — and `--force` re-splits and re-verifies the
     entire monolith from scratch, not just the new month.
