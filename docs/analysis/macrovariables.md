# Macro Panel Configuration (Economics)

This note documents the design choices behind `economics.yaml` and the
macroeconomic series included in the curated panel.

## Scope and data sources

- **Scope**: U.S. macroeconomic series only (initial build).
- **Source**: Datastream Economics tables, using the metadata in
  `src/data/datastream/economics/ecoinfo_monthly.csv` and
  `src/data/datastream/economics/ecoinfo_quarterly.csv`.
- **Frequencies**:
  - Most series are **monthly**.
  - Quarterly national accounts series (GDP and components, external NIPA,
    GDP deflator) are included with `freq: "quarterly"`.

## Seasonality

Where Datastream provides seasonally adjusted variants, the panel prefers
those (SA). This is consistent with typical macro-finance practice and
reduces the need for downstream seasonal adjustment.

## Panel structure and schema

The `economics.yaml` file uses a country key (currently `US`) so the schema
can be extended to other countries later.

Each entry includes:
- `ecoseriesid`, `dsnumber`, `name` (Datastream identifiers and label)
- `country`, `category`
- `unit` (human-readable unit derived from Datastream `UnitCode` and
  `ScaleCode`)
- `transform` (canonical analysis transform)
- `freq` (monthly or quarterly)

## Category choices

Series are grouped into:
`real_activity`, `labor`, `prices`, `monetary_credit`, `fiscal`, `external`,
`surveys`, `demographics`, `composite`, and `asset_prices`.

This aligns with macro-finance conventions while keeping the panel broad
enough for business-cycle analysis and asset-pricing applications.

## Key design decisions and rationale

- Keep the panel broad and primarily monthly for business-cycle timing, but
  add quarterly NIPA anchors (GDP and components) where no monthly analog
  exists.
- Prefer seasonally adjusted series to avoid ad hoc seasonal adjustment
  downstream and to match standard macro-finance practice.
- Retain both monthly proxies and quarterly anchors (no replacements) to
  preserve information richness across frequencies.
- Include a small set of asset-pricing indicators (CAPE, industrial share
  prices, term spread) to capture risk appetite and financial conditions
  alongside macro fundamentals.
- Record `freq` and `transform` explicitly to keep economic definitions
  transparent and reproducible.

## Transform choices (canonical, documented)

The `transform` field is the intended analysis transform. It is **not**
applied in the YAML; it is a documented choice for downstream processing.

Rules used:

- **Real activity**:
  - `log_diff` for levels (production, sales, orders, income, PCE).
  - `level` for ratios and rates (capacity utilization, inventories/sales).
- **Labor**:
  - `level` for unemployment rate.
  - `log_diff` for employment levels, labor force, job openings, earnings.
- **Prices**:
  - `yoy_pct` for price indices (CPI, PCE, PPI, import/export prices,
    house prices).
  - `level` for expectations/surprise indices (Michigan 1y expectations,
    Citi inflation surprise).
- **Monetary/credit**:
  - `level` for rates and spreads (policy rate, T-bill, yields, term spread,
    mortgage rate).
  - `log_diff` for quantities (money aggregates, bank credit, consumer credit).
- **Fiscal**:
  - `level` for balances, revenues, debt.
- **External**:
  - `log_diff` for exports/imports and effective FX indices.
  - `level` for balances and reserves.
- **Surveys / composite**:
  - `level` for indices (PMI, ISM, confidence, LEI/CLI).
- **Demographics**:
  - `log_diff` for population.
- **Asset prices**:
  - `log_diff` for price indices.
  - `level` for valuation ratios (e.g., CAPE).

These choices are intended to align with standard macro-finance usage and
keep the economic object explicit for reproducibility.

## Quarterly additions

Quarterly series were added where no monthly analog exists:
- Real GDP (real, SAAR).
- GNP.
- PCE, private fixed investment, government consumption & investment,
  change in private inventories.
- NIPA exports and imports.
- Current account balance.
- GDP deflator (implicit price deflator of GDP).

Quarterly series are **not** replacements; they complement the monthly
proxies.

## Asset-pricing indicators

Despite the macro focus, a few asset-pricing series are included for
business-cycle and risk-appetite context:
- S&P 500 CAPE ratio.
- Share prices (industrials).
- Term spread (10y Treasury minus Fed funds).

## Known gaps (not in Datastream monthly/quarterly economics tables)

Some requested series are not present in the Datastream economics
metadata and may need other sources:
- Labor force participation rate / employment-population ratio.
- Average weekly hours (manufacturing or total private).
- Initial and continuing unemployment claims.
- Corporate credit spreads (BAA/AAA or high yield).
- Long-horizon inflation expectations (5–10y).
- Output gap / NAIRU / potential GDP.

These can be added later from non-economics Datastream sources or other
vendors if needed.
