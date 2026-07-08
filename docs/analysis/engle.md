# Engle (1982) ARCH Test for Volatility Clustering

This document describes the procedure for documenting time-varying volatility and volatility clustering in the futures dataset using Engle's (1982) test for autoregressive conditional heteroskedasticity (ARCH). The goal is to quantify the extent of volatility clustering by asset class.

---

## 1) Objective

- **Primary**: Document the presence and strength of volatility clustering across tier1 futures.
- **Secondary**: Stratify results by asset class (commodity, bond, currency, equity, volatility, stir) to compare clustering behavior across major asset classes.

Volatility clustering refers to the empirical fact that large returns tend to be followed by large returns (of either sign) and small returns by small returns—i.e., squared returns (or absolute returns) are positively autocorrelated.

---

## 2) Method: Engle (1982) LM Test

### 2.1 Null and Alternative Hypotheses

- **Null**: No ARCH effects. The conditional variance of the residual is constant; squared residuals are uncorrelated with their lags.
- **Alternative**: ARCH effects exist. Past squared residuals help predict current squared residuals (volatility clustering).

### 2.2 Test Procedure

1. **Obtain residuals** $\hat{\epsilon}_t$ from a mean model. For daily returns, we use a constant mean:
   $$
   r_t = \mu + \epsilon_t
   $$
   so $\hat{\epsilon}_t = r_t - \bar{r}$ (demeaned returns).

2. **Auxiliary regression**: Regress squared residuals on a constant and $q$ lagged squared residuals:
   $$
   \hat{\epsilon}_t^2 = \alpha_0 + \alpha_1 \hat{\epsilon}_{t-1}^2 + \cdots + \alpha_q \hat{\epsilon}_{t-q}^2
   $$

3. **Test statistic**: Under the null, $T' R^2 \sim \chi^2(q)$ where $T' = T - q$ is the number of observations in the auxiliary regression.

4. **Decision**: Reject the null at 5% significance if $T' R^2 > \chi^2_{0.95}(q)$.

### 2.3 Implementation

- **Function**: `statsmodels.stats.diagnostic.het_arch` (verified against R `FinTS::ArchTest`).
- **Input**: 1D array of residuals (demeaned log returns).
- **Parameters**:
  - `resid`: demeaned log returns.
  - `nlags`: $q$ (number of ARCH lags).
  - `ddof`: 1 (constant mean estimated).

### 2.4 Lag Order

- **Main specification**: $q = 12$ (approximately 2.5 trading weeks).
- Robustness checks (different $q$) are deferred for later.

---

## 3) Data Sources and Preprocessing

### 3.1 Source

- **Path**: `DATASETS_ROOT/tier1/async/async_daily.csv`.
- **Frequency**: Daily (required for volatility clustering analysis; monthly would understate clustering).

### 3.2 Conversion

- **Raw data**: Simple returns $R_t$ in asset columns.
- **Conversion**: Log returns $r_t = \ln(1 + R_t)$.
- **Residuals**: For each asset, $\hat{\epsilon}_t = r_t - \bar{r}$ (sample mean over non-null observations).

### 3.3 Sample Filters

- **Cutoff date**: Keep only assets whose first non-null observation is on or before **2005-12-31** (≥20 years of history for inference).
- **Minimum observations**: Skip assets with fewer than $q + 50$ valid (non-null) observations after filtering. With $q = 12$, this implies at least 62 observations.

### 3.4 Asset Universe and Mapping

- **Universe**: Tier 1 futures from `tier1.yaml`.
- **Asset class mapping**: Build symbol → asset class from `tier1.yaml`. For symbols with multiple classes (e.g. `[commodity, historical]`), use the primary class (first non-historical, non-traditional entry) for stratification. Examples:
  - `commodity`, `bond`, `currency`, `equity`, `volatility`, `stir`
  - Equity subclasses (`us_equity`, `nonus_equity`) may be grouped as `equity` for stratification.

### 3.5 Missing Data

- Per-asset: Drop rows where that asset's return is null. The Engle test is run on each asset's valid series independently.
- No cross-asset imputation.

---

## 4) Target Output

### 4.1 Per-Asset

| Field    | Description                                      |
|----------|--------------------------------------------------|
| `symbol` | Asset symbol (e.g. `ES`, `ZN`, `CL`)            |
| `asset_class` | Primary asset class for stratification     |
| `n_obs`  | Number of valid observations used for the test   |
| `lm_stat`| LM test statistic $T' R^2$                        |
| `p_value`| P-value for the $\chi^2(q)$ distribution         |
| `reject_5pct` | True if null rejected at 5% significance   |

### 4.2 Per-Asset-Class Summary

| Field    | Description                                      |
|----------|--------------------------------------------------|
| `asset_class` | Asset class label                          |
| `n_assets` | Number of assets in that class               |
| `n_reject` | Number of assets rejecting at 5%             |
| `reject_rate` | Fraction rejecting (n_reject / n_assets)   |
| `mean_pvalue` | Mean p-value across assets in class  |

---

## 5) Notebook Workflow

The notebook `src/analysis/notebooks/04_engle.ipynb` will implement the following steps.

### Step 1: Setup

- Import: `polars`, `numpy`, `statsmodels.stats.diagnostic.het_arch`, `utils.config.load_config`, `utils.paths.DATASETS_ROOT`, `utils.paths.PROJECT_ROOT`.
- Plotting: Reuse matplotlib style from `04_vratio.ipynb` if desired (optional).

### Step 2: Load Data

- Read `DATASETS_ROOT/tier1/async/async_daily.csv`.
- Parse `date` column as `pl.Date`.
- Convert simple returns to log returns: $r_t = \ln(1 + R_t)$.
- Filter to assets with first non-null on or before 2005-12-31.
- Drop assets with $< q + 50$ valid observations.

### Step 3: Build Asset Class Map

- Load `tier1.yaml` via `load_config(PROJECT_ROOT / "tier1.yaml")`.
- For each symbol, extract primary asset class (first non-historical, non-traditional entry).
- Handle symbols with `asset_class: [commodity, historical]` → use `commodity`.

### Step 4: Run Engle Test

- For each asset column:
  - Extract valid (non-null) log returns.
  - Demean: $\hat{\epsilon}_t = r_t - \bar{r}$.
  - Call `het_arch(resid, nlags=12, ddof=1)`.
  - Store: `lm_stat`, `p_value`, `reject_5pct` (p_value < 0.05).

### Step 5: Build Summary Tables

- **Per-asset table**: `symbol`, `asset_class`, `n_obs`, `lm_stat`, `p_value`, `reject_5pct`.
- **Per-asset-class table**: `asset_class`, `n_assets`, `n_reject`, `reject_rate`, `mean_pvalue`.

### Step 6: Figures

- **Figure 1**: Bar chart of rejection rate by asset class.
- **Figure 2**: Distribution of p-values by asset class (e.g. histogram or box plot).

### Step 7: Optional Output

- Write per-asset results to `src/analysis/notebooks/tables/engle_results.csv` for reproducibility.

---

## 6) Interpretation

- **Rejection** at 5%: Evidence of ARCH effects (volatility clustering) in that asset's returns.
- **Higher rejection rate** in an asset class: That class exhibits more pronounced volatility clustering on average.
- **Low p-values** in many assets: Consistent with the well-documented stylized fact that financial returns display volatility clustering.

---

## 7) References

- Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. *Econometrica*, 50(4), 987–1007.
- statsmodels `het_arch`: https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_arch.html

---

## 8) Proposed Implementation Checklist

Before implementation, the following are specified:

| Item | Specification |
|------|---------------|
| Test | Engle (1982) LM test via `statsmodels.stats.diagnostic.het_arch` |
| Data | `DATASETS_ROOT/tier1/async/async_daily.csv` |
| Return type | Log returns $r_t = \ln(1 + R_t)$ |
| Residuals | Demeaned log returns: $\hat{\epsilon}_t = r_t - \bar{r}$ |
| Lag order | $q = 12$ |
| Cutoff date | First non-null on or before 2005-12-31 |
| Min observations | $q + 50$ (≥ 62 with $q = 12$) |
| Asset classes | From `tier1.yaml`; primary class for stratification |
| Output | Per-asset table + per-asset-class summary + figures |
| Output path | `src/analysis/notebooks/tables/engle_results.csv` (optional) |

**Deferred for later**: Robustness with different $q$ (e.g. 5, 10), multiple-testing adjustments, subperiod analysis.

---

## 9) Proposed Changes (Summary)

The following changes will be made upon approval:

### 9.1 Files to Create

| File | Action |
|------|--------|
| `docs/engle.md` | **Created** (this document) |

### 9.2 Files to Implement

| File | Action | Content |
|------|--------|---------|
| `src/analysis/notebooks/04_engle.ipynb` | **Populate** (currently empty) | Full notebook implementing §5 (Setup → Load Data → Asset Class Map → Engle Test → Summary Tables → Figures). |

### 9.3 Files Possibly Created by Notebook

| File | When |
|------|------|
| `src/analysis/notebooks/tables/engle_results.csv` | When user runs the notebook and saves results (optional step). |

### 9.4 No Other Changes

- No modifications to `utils/`, `globalmacro/`, or other analysis modules.
- No new dependencies (statsmodels already in `pyproject.toml`).
