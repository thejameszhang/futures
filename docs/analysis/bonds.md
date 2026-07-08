# Bonds Data Sources

This document summarizes the US and non-US bond data sources available in the
project and how to locate them.

## 1) US Bonds (CRSP US Treasury)

Source: CRSP US Treasury databases on WRDS. These are **US-only** (Treasury
securities). We pull both daily and monthly datasets so we can construct term
structure factors, returns, carry/roll-down, and yield curve signals at either
frequency.

Daily US Treasury (CRSP):
- Master / issue-level: `BMHEADER`, `BMPAYMTS`, `BMQUOTES`, `BMYIELD`, `BMDEBT`
- Cross-sectional / index-level: `BXCALIND`, `BXDLYIND`, `BXQUOTES`, `BXYIELD`
- Fixed-term indices (constant-maturity): `TFZ_IDX`, `TFZ_DLY_FT`
  - Note: WRDS “tfz_ft” is a join of `tfz_idx` and `tfz_dly_ft`.
- Synthetic futures returns for ZT/ZF/ZN/ZB use a duration–yield approximation
  with roll-down: r_{t+1} = −D_t × (y_{t+1} − y_t) + D_t × (y_t − y_{t-})/365,
  where D_t is modified duration (from `tdduratn`) and y_{t-} is the next
  shorter maturity yield in `TFZ_DLY_FT`. For ZN we use the 5Y bucket as the
  short leg; for ZB we interpolate a 15Y bucket from 10Y and 20Y to improve
  long-end roll-down. Output: `synthetic_bond_excess_daily.csv`.

Monthly US Treasury (CRSP):
- Calendar / master / cross-sectional: `MBI`, `MBMDAT`, `MBMHDR`, `MBX`, `MBXID`
- Yields: `YLDASK06`, `YLDASK12`, `YLDAVE06`, `YLDAVE12`, `YLDBID06`, `YLDBID12`
- Prices: `PRIASK06`, `PRIASK12`, `PRIAVE06`, `PRIAVE12`, `PRIBID06`, `PRIBID12`
- Forward rates: `FWDASK06`, `FWDASK12`, `FWDAVE06`, `FWDAVE12`, `FWDBID06`, `FWDBID12`
- Holding-period returns: `HLDASK06`, `HLDASK12`, `HLDAVE06`, `HLDAVE12`, `HLDBID06`, `HLDBID12`
- Fixed-term indices (monthly): `BXMTHIND`
- Risk-free: `RISKFREE`
- Fama-Bliss: `FBPRI`, `FBYLD`
- Maturity portfolio returns: `BNDPRT06`, `BNDPRT12`
- Fixed-term indices (constant-maturity): `TFZ_IDX`, `TFZ_MTH_FT`
  - Note: WRDS “tfz_ft” is a join of `tfz_idx` and `tfz_mth_ft`.

## 2) Non-US Bonds (Datastream Economics)

Source: Datastream Economics tables under `src/data/datastream/economics/`.
Key tables:
- `ecoinfo.csv`: series metadata (country, currency, frequency, description).
- `ecodata.csv`: time-series values (by `EcoSeriesID` and `PeriodDate`).
- `ecocode.csv`: code lookup for MktCode, CurrCode, UnitCode, FreqCode, etc.
- `ecoclscode.csv`: classification labels (e.g., Interest Rates, Bond Market).

Useful series families for foreign bond data (identified via `ecoinfo.csv`):

1) Government bond yields (long-term, typically 10-year):
   - Pattern: `dsmnemonic` contains `GBOND` (e.g., `AUGBOND.`, `JPGBOND.`).
   - Typical metadata: `unitcode = 241` (Percentage), `freqcode = MONT`.
   - Best starting point for long-end yields and term-structure factors.

2) Government securities rates (bonds + bills):
   - Pattern: `desc_english` contains
     `INTEREST RATES: GOVERNMENT SECURITIES`.
   - Includes both Treasury bills (short-end) and government bonds (long-end).
   - Tenor often only appears in the description text.

3) Money market rates (short-end proxy):
   - Pattern: `desc_english` contains `INTEREST RATES: MONEY MARKET RATE`.

4) Central bank policy rates:
   - Pattern: `desc_english` contains `CENTRAL BANK POLICY RATE`.

5) Treasury bill yields:
   - Pattern: `desc_english` contains `TREASURY BILL` or `TREASURY BILLS`.

Frequency coverage:
- Mostly **monthly** (`MONT`), with some **quarterly** (`QUAR`) and **annual**
  (`ANNL`).
- A small number of series are **weekly** (`WWE`, `WFR`) or **weekday/daily**
  (`DWY`), but these are the exception rather than the rule.

Units and codes:
- `UnitCode = 241` maps to **Percentage** (common for yield series).
- `MktCode` maps to country/region (see `ecocode.csv`, `series_type = 5`).
- `CurrCode` maps to currency (see `ecocode.csv`, `series_type = 3`).

Forward yields:
- Economics “FORWARD RATE” series typically carry **currency units** and appear
  to be **FX forward rates**, not bond forward yields. For bond forwards, we will
  likely need to **derive forwards from multi-tenor yields**.
