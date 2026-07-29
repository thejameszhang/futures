# globalmacro data bundle

Raw inputs and prerequisites to run the `globalmacro` pipeline end-to-end, plus
everything the `download.py` WRDS pulls would produce (so the bundle is
self-describing). Unpack so that a `data/` directory sits at the repo root
(next to `pyproject.toml`).

## Included
- `data/tickhistory/{trades,quotes}/` — LSEG TickHistory pulls (large).
- `data/datastream/{futures,equities,fx,economics}/` — Datastream WRDS pulls
  (all `dsf*` / `eco*` / the equities & fx table sets), the public
  `ded3_wrds.csv` and `oecd.csv` (fetchable via `globalmacro download-public`), and
  `datastream_continuous_series.csv` (fetchable via `globalmacro download --database datastream_continuous`;
  used by the datastream validation cross-check).
- `data/comp/exrt_dly.csv`, `data/jkp/updated_daily_ind_gics*.csv`,
  `data/misc/F-F_Research_Data_Factors_daily.csv` — prerequisite (F-F fetchable via `globalmacro download-public`).

## Not included (regenerated or out of scope)
- Regenerable interims (`*.parquet`, `spot_equity_returns.csv`,
  `synthetic_fx_returns_async.csv`, `synthetic_fx_returns_sync.csv`,
  `3month_libor_rates.csv`, `instrumentlists/`) — recreated on a full pipeline run.
- Debug output, stale/duplicate files, out-of-scope analysis datasets, and any
  confidential third-party reference series.

## Running
See the repo `README.md`: `uv sync`, then `globalmacro run` (or per-stage).
No interim files are required to be present.
