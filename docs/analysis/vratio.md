# Variance Ratio Test Procedure

This document describes the exact procedure from our dataset to heteroskedasticity-robust variance ratio test results. The variance ratio test assesses the random walk hypothesis: under the null, the variance of $q$-period returns equals $q$ times the variance of one-period returns, so $\text{VR}(q) = 1$.

---

## 1) Target Output

For each asset and each holding period $q$:

- **Variance ratio** $\text{VR}(q)$: ratio of variance of $q$-period returns to $q \cdot \text{Var}(r_t)$. Under the null, $\text{VR}(q) = 1$. VR $> 1$ suggests momentum/trending; VR $< 1$ suggests mean reversion.
- **Heteroskedasticity-robust test statistic** $\psi^*(q)$ (z*): asymptotically standard normal under the null.
- **P-value**: two-sided p-value for the test statistic.
- **Rejection decision**: whether the random walk null is rejected at 5% significance.

We use the heteroskedasticity-robust statistic throughout because financial returns exhibit time-varying volatility; the homoskedastic test can spuriously reject due to heteroskedasticity alone.

---

## 2) Data Sources and Preprocessing

### 2.1 Futures (cross-asset analysis)

- **Source**: `DATASETS_ROOT/tier1/async/async_monthly.csv` (or `async_daily.csv` if weekly aggregation is used).
- **Frequency**: Monthly for main analysis.
- **Conversion**: Simple returns → log returns via $r_t = \ln(1 + R_t)$.
- **Missing data**: Keep only columns whose first non-null observation is on or before a cutoff date (e.g., 2005-12-31) to ensure adequate history for inference.
- **Asset universe**: Tier 1 futures from `tier1.yaml`; symbol-to-asset-class mapping for stratification.

### 2.2 CRSP (Lo–MacKinlay 1988 Table 1a replication)

- **Source**: `CRSP_PATH/crsp_a_index.csv` (or equivalent CRSP index file).
- **Frequency**: Weekly, aggregated from daily.
- **Weekly aggregation** (Lo and MacKinlay 1988, Section 2): Two-point return $R_t = \log P_{\text{end}} - \log P_{\text{start}}$. Sequential chaining: this week's start = last week's chosen endpoint (no re-resolving). Wed preferred, else Thu, else Tue. Log prices only at observed trade dates. Output: weekly log returns. MATLAB vratio.m receives log returns and converts to log prices. Sparse = per-asset, weeks where that asset had no Wed/Thu/Tue price.
- **Sample**: 1962-09-06 to 1985-12-26 (full sample, nq=1216).

---

## 3) What Does MATLAB vratiotest Expect? (Critical)

### 3.1 The MATLAB `vratiotest` function

MATLAB's `vratiotest` **expects data in levels**, not returns. From the official documentation:

> **y — Univariate time series data in levels**  
> Univariate time series data **in levels**, specified as a numeric vector. Each element of `y` represents an observation.
>
> **Note:** `vratiotest` assumes that the specified input data is in levels. **To convert a return series `r` to levels**, define `y(1)` appropriately and let **`y = cumsum([y(1) r])`**.

The test internally computes one-period returns as:

$$r_t = y_t - y_{t-1}$$

So the input `y` must be a series whose first difference yields the return series of interest.

### 3.2 What input types work?

| Input type        | Correct? | Why |
|-------------------|----------|-----|
| **Log prices**    | ✓ Yes    | $r_t = \log P_t - \log P_{t-1}$ = log return. This is what the Lo–MacKinlay test uses. |
| **Cumsum of log returns** | ✓ Yes | $y_t = \sum_{s=1}^t r_s$ with $y_0=0$ gives log prices; $r_t = y_t - y_{t-1}$ recovers the log return. |
| **Raw (nominal) prices** | ⚠ Technically valid but different | $r_t = P_t - P_{t-1}$ = simple return. The variance ratio literature (Lo–MacKinlay, CLM) uses **log returns**; using raw prices would test a different object. |
| **Log returns directly** | ✗ Wrong | vratiotest would treat them as levels and compute $r_t = \text{logreturn}_t - \text{logreturn}_{t-1}$, which is meaningless. |
| **Simple returns directly** | ✗ Wrong | Same issue; differencing returns gives changes in returns, not returns. |

### 3.3 Our `vratio.m` wrapper

`vratio.m` is designed to accept **log returns** (the natural input when we have return data) and converts them internally:

```matlab
log_prices = [0; cumsum(y_valid)];   % y_valid = log returns
[h, p, vstat, ~, vr] = vratiotest(log_prices, 'Period', k, 'IID', false);
```

So:

- **vratio.m expects**: **Log returns** (each column = one asset's log return series).
- **vratiotest receives**: Log prices (cumsum of log returns with initial 0).

### 3.4 Our CSV format

The CSV we export from Python must contain **log returns** in the asset columns. The first column is dates (dropped by vratio.m). Do **not** pass prices or simple returns.

---

## 4) Input Format for vratio

The MATLAB function `vratio.m` expects:

- **Input**: Either (a) a $T \times N$ numeric matrix of **log returns** (each column is an asset), or (b) a CSV file path. If CSV, the first column is assumed to be dates and is dropped; remaining columns are numeric **log returns**.
- **NaN handling**: MATLAB's `vratiotest` removes missing observations internally. Per-asset, only valid (non-NaN) returns are used.

---

## 5) Variance Ratio Computation

We use MATLAB's `vratiotest` via the wrapper `vratio.m`:

1. **Conversion**: Log returns $r_t$ are converted to log prices $y_t$ via $y_t = y_0 + \sum_{s=1}^t r_s$ (with $y_0 = 0$). `vratiotest` expects price levels.
2. **Test specification**: `vratiotest(log_prices, 'Period', q, 'IID', false)`.
   - `IID = false`: heteroskedasticity-robust test (recommended for financial data).
   - `IID = true`: homoskedastic test; not used.
3. **Periods**:
   - **Monthly data**: $q = 2, 3, \ldots, 61$ (up to ~5 years).
   - **Weekly data**: $q = 2, 3, \ldots, 53$ (up to ~1 year).
4. **Minimum sample**: Skip assets/periods where $n < q + 10$ observations.

---

## 6) Output Format

`vratio.m` returns (and writes to CSV):

| Field    | Description                                      |
|----------|--------------------------------------------------|
| `assets` | Cell array of asset names                        |
| `periods`| Vector of periods $q$ tested                      |
| `stat`   | $N \times K$ matrix of $\psi^*(q)$ (z*)          |
| `pvalue` | $N \times K$ matrix of p-values                  |
| `reject` | $N \times K$ logical (true if reject at 5%)      |
| `ratio`  | $N \times K$ matrix of variance ratios VR(q)     |

The summary table written to CSV includes columns: `Asset`, `zstar_2`, `zstar_3`, …, `pval_2`, `pval_3`, …, `vr_2`, `vr_3`, …, and `n_rejections`.

---

## 7) Python–MATLAB Workflow (CSV Round-Trip)

We use a CSV round-trip because MATLAB cannot be run on the HPC (licensing). The workflow is manual: run Python on HPC, copy files to a machine with MATLAB, run vratio locally, copy results back.

### Step 1: Load and prepare data (Python, on HPC)

- Load the dataset (futures or CRSP) as described in §2.
- Convert to log returns if needed.
- Apply any filters (cutoff dates, sparse weeks, etc.).
- Result: a Polars DataFrame with `date` column + one column per asset (log returns).

### Step 2: Export to CSV (Python, on HPC)

- Write the DataFrame to CSV: first column `date`, remaining columns are asset symbols (log returns).
- Files are saved to `src/analysis/notebooks/tables/` (e.g. `clm_table1a.csv` for the CLM full-sample replication).
- **Download the CSV** to your local machine (or wherever MATLAB is available).

### Step 3: Run vratio in MATLAB (locally)

Place `vratio.m` and the input CSV in the same directory (or use full paths). In MATLAB:

```matlab
vratio('clm_table1a.csv', [], [2 4 8 16], 'clm_table1a_out.csv');
```

For weekly data (CLM Table 1a), use periods `[2 4 8 16]`. For monthly, use `2:61`.

### Step 4: Upload output CSV back to HPC

- Copy `clm_table1a_out.csv` to `src/analysis/notebooks/tables/` on the HPC.

### Step 5: Load output and reshape (Python, on HPC)

- Python reads the output CSV and reshapes from wide (columns `zstar_2`, `vr_2`, …) to long format for plotting: `asset`, `k`, `z_star_stat`, `variance_ratio`.

### Step 6: Analysis and plotting (Python)

- Use the reshaped Polars DataFrame for tables, rejection summaries, and figures (e.g. VR by asset, ψ* term structure by asset class).

---

## Futures Workflow (Weekly Data)

For tier1 futures, the notebook `05_matlab_vratio.ipynb`:

1. Loads `async_daily.csv`, filters to assets with first non-null on or before 2005-12-31 (≥20 years history).
2. Aggregates to weekly (Wed-Wed, same sparse-week rule as CRSP).
3. Exports `futures_weekly.csv` (log returns). Asset columns are prefixed with `s_` (e.g. `s_160120001`, `s_ZW`) so MATLAB `readtable` preserves headers; Python strips the prefix when loading vratio output.
4. MATLAB: `vratio('futures_weekly.csv', [], [2 4 8 16], 'futures_weekly_out.csv');`
5. Upload `futures_weekly_out.csv`, then plot ψ*(k) by asset class (2×3 grid). Rejected box shows full future names from tier1.

---

## 8) References

- Lo, A. W., & MacKinlay, A. C. (1988). Stock market prices do not follow random walks: Evidence from a simple specification test. *The Review of Financial Studies*, 1(1), 41–66.
- Campbell, J. Y., Lo, A. W., & MacKinlay, A. C. (1997). *The Econometrics of Financial Markets*. Princeton University Press. Chapter 12, equations 2.4.42–2.4.44.
- MATLAB `vratiotest` documentation: https://www.mathworks.com/help/econ/vratiotest.html

---

## 9) Implementation Notes

- **vratio.m**: Returns `stat`, `pvalue`, `ratio`, `reject` and writes all to CSV. Accepts optional `outpath` argument.
- **Column naming**: Notebook plotting expects `asset`, `k`, `variance_ratio`, `z_star_stat`, `p_value_z_star`. A mapping layer converts vratio output to this schema.
