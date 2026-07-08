# MOP Figure 1 Replication Plan

This note sets the immediate priority: replicate Fig. 1 from Moskowitz, Ooi,
and Pedersen (2012) with our data pipeline before extending the cross-asset
risk-return design.

---

## 1) Target Output

Replicate the three components of Fig. 1:

- Panel A: pooled $t$-statistics by lag month $h$ from return-on-return regression.
- Panel B: pooled $t$-statistics by lag month $h$ from sign regression.
- Panel C: sign-regression $t$-statistics by lag month $h$ within each asset class
  (commodity, equity, bond, currency).

Use $h=1,\dots,60$ months.

---

## 2) Exact Regression Definitions

Use MOP-style volatility scaling throughout.

### What "pooled" means (explicit stacking and dimensions)

For each lag $h$, we form one stacked cross-section/time panel over all eligible
instruments and months in the sample window.

Let:

- $\mathcal{S}$ be the instrument set used in replication (commodity, equity,
  bond, currency symbols only).
- $\mathcal{T}$ be monthly dates from $1985\text{-}01$ to $2009\text{-}12$.
- $\mathcal{D}_h \subseteq \mathcal{S}\times\mathcal{T}$ be rows where both the
  dependent variable and the lag-$h$ regressor are observed.

Then for each $h$:

$$
N_h = |\mathcal{D}_h|
$$

and we estimate one OLS on the stacked sample:

$$
y_h = X_h\theta_h + \varepsilon_h, \quad
y_h \in \mathbb{R}^{N_h}, \; X_h \in \mathbb{R}^{N_h \times 2}
$$

where column 1 of $X_h$ is an intercept and column 2 is the lag regressor.
So "pooled" means we do **not** run separate regressions per symbol and average;
we run one regression per lag on all stacked $(s,t)$ observations together.

In our current implementation, this is an unbalanced panel (different symbols
have different start dates), so $N_h$ varies by lag.

### Panel A (all asset classes pooled)

For each lag $h$:

$$
\frac{r^s_t}{\sigma^s_{t-1}}
= \alpha_h + \beta_h \frac{r^s_{t-h}}{\sigma^s_{t-h-1}} + \varepsilon^s_t
$$

Record the $t$-statistic of $\beta_h$.

### Panel B (all asset classes pooled)

For each lag $h$:

$$
\frac{r^s_t}{\sigma^s_{t-1}}
= \alpha_h + \beta_h\,\mathrm{sign}(r^s_{t-h}) + \varepsilon^s_t
$$

Record the $t$-statistic of $\beta_h$.

### Panel C (by asset class)

For each asset class $c$ and lag $h$, run Panel B within class $c$ only and
record the $t$-statistic of $\beta_h$.

---

## 3) Ex-Ante Volatility Construction (Daily Data)

From `DATASETS_ROOT/tier1/async/async_daily.csv`, compute instrument-level daily
ex-ante volatility using:

$$
\sigma^{2,s}_t
= 261 \sum_{j=0}^{\infty} (1-\delta)\delta^j\left(r^s_{t-1-j}-\bar r^s_t\right)^2,
\quad
\frac{\delta}{1-\delta}=60
$$

with $\bar r^s_t$ the exponentially weighted mean of lagged daily returns under
the same weights.

Strict timing:

- Daily $\sigma^s_t$ uses information through day $t-1$ only.
- Monthly scaled return for month $m$ uses previous month-end volatility,
  i.e. $r^s_m / \sigma^s_{m-1}$.

---

## 4) Data Assembly Steps

1. Load monthly returns from `DATASETS_ROOT/tier1/async/async_monthly.csv`.
2. Load daily returns from `DATASETS_ROOT/tier1/async/async_daily.csv`.
3. Build symbol-to-asset-class map from `tier1.yaml`.
4. Keep only four classes: commodity, equity, bond, currency.
5. Compute daily ex-ante volatility per symbol using the MOP EWMA definition.
6. Collapse daily volatility to month-end $\sigma^s_m$.
7. Create monthly scaled returns $r^s_m / \sigma^s_{m-1}$.
8. Build lagged RHS variables for each $h=1,\dots,60$.
9. Run pooled regressions with month-clustered standard errors.
10. Save the three panel series and plot in Fig. 1 layout.

---

## 5) Inference and Estimation Details

- Use pooled panel OLS on stacked instrument-month rows:
  one regression per lag $h$ (Panels A/B) or per lag-class pair $(h,c)$ (Panel C).
- Include an intercept in every regression.
- Cluster standard errors by month (time clustering), matching MOP.
- If $g_h$ is the month label vector for the $N_h$ stacked rows, then
  $g_h \in \mathcal{T}^{N_h}$ and clustering is applied on $g_h$.
- Use the coefficient $t$-statistic on $\beta_h$ as plotted value.
- No macro controls in this replication exercise.

---

## 6) Sample Window and Comparability

Primary replication window:

$$
1985\text{-}01 \text{ to } 2009\text{-}12
$$

to align with the paper figure. We can run a second extended-sample version
after the baseline replication is complete.

Expected differences vs the paper are still possible due to contract set,
construction details in our async pipeline, and instrument coverage.

---

## 7) Code Changes We Will Make

- Add a dedicated replication block in
  `src/analysis/notebooks/06_risk_return_tradeoff.ipynb` for Fig. 1.
- Keep the existing cross-asset pairwise regression code separate so the Fig. 1
  replication remains auditable.
- Export replication artifacts:
  - `src/analysis/notebooks/figures/mop_fig1_replication.pdf`
  - `src/analysis/notebooks/mop_fig1_tstats.csv`

