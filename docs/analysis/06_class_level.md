# Cross-Asset Pricing: Does Asset Class X Price Asset Class Y?

## Research Blueprint

A unified, SDF-based framework for testing whether factors from one futures asset class price the returns of another. This outline is explicitly **factor-pricing / SDF–oriented**, centered on **payoff space**, and designed to be **tractable** with real global futures data. The focus is **cross-sectional only**: does asset class X price the cross-section of asset class Y?

---

## 1. Motivation and Contribution

### 1.1 Research Question

> In a unified and systematic way: **Does asset class X price asset class Y?**

The literature documents many pairwise facts (equity–bond, FX–commodity, volatility–equity, etc.) but lacks a common framework to:

- Compare pricing ability across all ordered pairs (X, Y) of asset classes.
- Ground conclusions in **SDF / linear factor models** rather than VAR spillovers or variance decomposition.
- Use a **purely statistical** factor construction (PCA) that is mechanical and replicable.

### 1.2 Contribution

1. **Conceptual clarity:** Rigorous definition of “X prices Y” at the cross-sectional level, grounded in factor-pricing theory and economic interpretation (shared SDF exposure, risk sharing).
2. **PCA-based factor architecture:** Class-specific PCs only (F_X, F_Y). No global factor; no EW indices; no single-contract factors. Simple and tractable.
3. **Cross-sectional pricing tests:** Fama–MacBeth / GMM with proper econometric inference (Newey–West, Shanken, GRS). OOS validation via expanding/rolling window to guard against regime dependence.
4. **Pricing network:** Directed graph summarizing which classes price which, with interpretable “hubs” and “price takers.”

---

## 2. Conceptual Framework: What Does “X Prices Y” Mean?

### 2.1 SDF Foundation

Under no-arbitrage, there exists a stochastic discount factor $m_{t+1}$ such that for any excess return $r_{i,t+1}$:

$$
\mathbb{E}_t[m_{t+1} r_{i,t+1}] = 0.
$$

A **linear factor model** posits:

$$
m_{t+1} = 1 - b' (f_{t+1} - \mathbb{E}[f_{t+1}]),
$$

where $f_{t+1}$ is a vector of factor returns. This implies:

$$
\mathbb{E}[r_{i,t+1}] = \lambda' \beta_i,
$$

where $\lambda$ are factor risk premia and $\beta_i$ are factor loadings. Equivalently, in time-series form:

$$
r_{i,t+1} = \alpha_i + \beta_i' f_{t+1} + \varepsilon_{i,t+1},
$$

with $\alpha_i = 0$ if the model prices asset $i$ exactly.

### 2.2 Cross-Sectional Definition

**Definition:** For the cross-section of contracts in Y, we say **“asset class X prices the cross-section of Y”** if:

1. **X-only model performance:** A model with X’s native factors (F_X) alone achieves cross-sectional $R^2$ and HJ distance comparable to (or better than) a Y-native model (F_Y).
2. **Marginal contribution of Y:** The combined model (F_X + F_Y) yields only marginal improvement over the X-only model in terms of average squared alpha, mean absolute alpha, or HJ distance.
3. **Formal test:** We cannot reject that the cross-sectional alphas from the X-only model are jointly zero for the contracts in Y.

**Economic interpretation:** “X prices Y” means that the cross-section of Y’s returns is exposed to the same risk factor(s) that drive X. Y’s expected returns are then determined by exposure to X’s factor—i.e., shared SDF exposure. Y is a “price taker” in the sense that its risk premia are subsumed by X’s factor structure. This is a statement about **risk sharing** and **factor exposure**, not merely correlation.

Operationally: we compare Model A (F_Y), Model B (F_X), and Model C (F_X + F_Y). If B performs nearly as well as A and C adds little beyond B, we conclude that X prices the cross-section of Y.

---

## 3. Data and Frequency

### 3.1 Asset Universe

Asset classes (from `tier1.yaml` / `tier2.yaml`):

| Class       | Examples |
|------------|----------|
| Equity     | ES, NQ, Z, FESX, Nikkei, etc. |
| Bond       | ZN, ZB, ZT, FGBL, FGBS, JGB, etc. |
| Commodity  | CL, BRN, GC, HG, ZC, ZW, etc. |
| Currency   | 6E, 6J, 6B, 6A, etc. |
| STIR       | SR3, I, TIFEY, L, etc. |
| Volatility | VX, FVS |

All returns in **USD** (or common numéraire) via Datastream/TickHistory pipelines. Use existing `async_monthly.csv` as the primary frequency (see below).

### 3.2 Primary Frequency: Monthly

**Choice:** **Monthly** as the primary frequency.

**Justification:**

1. **Literature standard:** Cross-asset work (VME, Time Series Momentum, Asness–Moskowitz–Pedersen) uses monthly.
2. **Holiday alignment:** "Last trading day of month" is well-defined across exchanges; monthly largely avoids the Wednesday-holiday mismatch that plagues weekly (see Section 3.5).
3. **Signal vs. noise:** Monthly aggregates away microstructure and transitory flows; the pricing relationship (shared SDF exposure) is typically cleaner.
4. **Foundation for extensions:** Once monthly foundations are laid, higher frequencies can be explored as robustness or extensions.

**Construction:**

- $r_{i,m}^{monthly} = (P_{i,\tau_i(m)} / P_{i,\tau_i(m-1)}) - 1$, where $\tau_i(m)$ = last trading day of contract $i$ in month $m$.
- Use existing rolling rules from the futures pipeline.

### 3.3 Synchronization and Identification

- Align all returns to a **common month** (last trading day of month).
- **Baseline:** Use contemporaneous factors. **Economic interpretation:** Contemporaneous exposure means Y’s returns are priced by the same period’s realization of X’s factor—i.e., shared risk exposure in the SDF. This is the standard identification for factor models (Cochrane 2005).
- **Robustness:** Exclude announcement months or use lagged factors if needed.

### 3.4 Volatility Scaling

**Why it matters:** Asset classes have very different return volatilities. Scaling by ex-ante volatility puts returns on a comparable "risk unit" basis.

**MOP-style ex-ante volatility** (as in `risk_return.md`): For each contract $i$, compute

$$
\sigma^2_{i,t} = 261 \sum_{j=0}^{\infty}(1-\delta)\delta^j\left(r_{i,t-1-j}-\bar r_{i,t}\right)^2
$$

with $\delta/(1-\delta)=60$, using **daily** returns. At monthly frequency, $\sigma_{i,t}$ is the ex-ante volatility as of the start of month $t$.

**When to use:** Treat volatility scaling as a **robustness check**. Primary tests use raw returns.

### 3.5 Handling Holiday and Exchange Asynchrony

**At monthly frequency, the problem is much reduced.** "Last trading day of month" is typically well-defined across exchanges.

**Recommended approach:** For each contract $i$ and month $m$, let $\tau_i(m)$ = last trading day of contract $i$ in month $m$. Monthly return: $r_{i,m}^{monthly} = (P_{i,\tau_i(m)} / P_{i,\tau_i(m-1)}) - 1$.

---

## 4. Factor Construction: PCA-Only Architecture

All factors are constructed via **principal component analysis** on return panels. No EW indices; no single-contract factors. This keeps the design mechanical, replicable, and comparable across asset classes.

**Main specification:** 1 class PC per class (F_X, F_Y). That is, $K_X = 1$, $K_Y = 1$. No global factor—omitted for simplicity to keep the design tractable and focused on cross-class pricing. This parsimonious baseline tests whether the dominant factor from X prices the dominant factor from Y; additional PCs can be explored as robustness.

### 4.1 Literature Background

PCA for factor extraction in asset returns has a long tradition:

- **Connor & Korajczyk (1988):** Asymptotic PCA for large cross-sections of returns; factors are linear combinations of returns that capture common variation.
- **Stock & Watson (2002), Bai & Ng (2002):** Factor number selection (e.g., Bai–Ng criteria, eigenvalue scree).
- **Lettau & Pelger (2020), Kelly–Pruitt–Su (2019):** Risk-premium PCA (RP-PCA) incorporates mean returns into the decomposition; standard correlation PCA is the baseline.
- **Kozak, Nagel & Santosh (2020):** "Shrinking the Cross-Section"—PCA of factors; shows that a few statistical factors can span the cross-section.

For this project, **standard correlation PCA** is the baseline: demean returns, compute the correlation matrix, extract eigenvectors. Factor returns are the time series of portfolio returns with weights given by the eigenvectors. This is implemented in `src/analysis/notebooks/pca_utils.py`.

### 4.2 Class-Specific Factors (F_X, F_Y)

**Construction:** For each asset class X (and similarly for Y), run PCA on the **returns of contracts in X only**.

- Input: $N_X \times T$ matrix of returns for contracts $i \in X$.
- Extract the first $K_X$ principal components. **Main spec:** $K_X = 1$. Robustness: $K_X \in \{2, 3, \ldots\}$.
- Factor returns $f^X_t = [f^{X1}_t, \ldots, f^{XK_X}_t]$ are the class-X PC scores.

**Output:** For each class X, $f^X_t$ (and similarly $f^Y_t$ for class Y). These represent the payoff space of that class.

**Economic interpretation of PCs:** Add a brief economic interpretation of each class’s first PC (e.g., bond PC ≈ level/duration, equity PC ≈ market, commodity PC ≈ broad commodity index). This aids interpretability when we ask “does X price Y?”—we can describe what risk is being shared.

### 4.3 Factor Architecture Summary

| Block   | Contents |
|---------|----------|
| **X-native (F_X)** | $K_X$ PCs from the cross-section of contracts in asset class X (main: $K_X = 1$) |
| **Y-native (F_Y)** | $K_Y$ PCs from the cross-section of contracts in asset class Y (main: $K_Y = 1$) |

**Models:**

| Model | Factors |
|-------|---------|
| **A (Y-native)** | F_Y |
| **B (X-only)**  | F_X |
| **C (Combined)**| F_X + F_Y |

---

## 5. Cross-Sectional Pricing Tests

### 5.1 Competing Models

For each ordered pair (X, Y), define:

| Model | Factors |
|-------|---------|
| **A (Y-native)** | F_Y |
| **B (X-only)**  | F_X |
| **C (Combined)**| F_X + F_Y |

### 5.2 Fama–MacBeth Procedure

**Stage 1 (time-series):** For each contract $i \in Y$, run:

$$
r_{i,t} = \alpha_i + \beta_i' f_t + \varepsilon_{i,t},
$$

where $f_t$ is the factor vector for the model (A, B, or C). Collect $\hat{\alpha}_i$, $\hat{\beta}_i$.

**Stage 2 (cross-section):** For each $t$, run:

$$
r_{i,t} = \gamma_{0,t} + \gamma_t' \hat{\beta}_i + \eta_{i,t},
$$

or equivalently use $\hat{\alpha}_i$ as the dependent variable to test $\mathbb{E}[\alpha_i] = 0$.

**Inference (econometric):**

- **Stage 1 (time-series):** **Newey–West** HAC standard errors (e.g., 12 lags for monthly data) to allow for autocorrelation and heteroskedasticity in residuals. This is standard for factor model inference (Cochrane 2005).
- **Stage 2 (cross-section):** **Shanken-adjusted** standard errors for risk premia $\hat{\lambda}$. The adjustment multiplies the Fama–MacBeth variance by $(1 + \hat{\lambda}' \hat{\Sigma}_f^{-1} \hat{\lambda})$ to account for the fact that $\hat{\beta}_i$ is estimated. Without this, inference is overstated. References: Shanken (1992); Cochrane (2005) Ch. 12.
- **Joint test of $\alpha = 0$:** **GRS test** (Gibbons, Ross, Shanken 1989). Under the null that the model prices all assets, the statistic is $F_{N, T-N-K}$ distributed. This is the primary **statistical test** for “does the model price the cross-section?” For small $N_Y$ (e.g., $N_Y < 20$), note possible size distortions; block bootstrap is an alternative for inference.

**Metrics:**

- **Average squared alpha:** $\overline{\alpha^2} = \frac{1}{N_Y} \sum_i \hat{\alpha}_i^2$.
- **Mean absolute alpha:** $\overline{|\alpha|} = \frac{1}{N_Y} \sum_i |\hat{\alpha}_i|$.
- **Cross-sectional $R^2$:** From Stage 2 regression.

**Multiple testing:**

- With 30 ordered pairs (X, Y), control for multiple comparisons.
- **Benjamini–Hochberg (FDR):** Sort p-values $p_{(1)} \leq \ldots \leq p_{(30)}$; reject hypotheses $1, \ldots, k$ where $k$ is the largest index with $p_{(k)} \leq (k/30) \cdot 0.05$. Controls the expected fraction of false rejections.
- **Holm–Bonferroni:** Reject $H_{(1)}$ if $p_{(1)} < 0.05/30$, $H_{(2)}$ if $p_{(2)} < 0.05/29$, etc. More powerful than Bonferroni; controls family-wise error rate.
- **Report:** Raw p-values for each pair; apply Benjamini–Hochberg at 5% (or 10%) and report which pairs survive; add Holm as a conservative check.

### 5.3 GMM / HJ Distance (Optional but Recommended)

Estimate the SDF parameters $b$ by minimizing the HJ distance:

$$
\delta^2 = \min_b \mathbb{E}[(m_{t+1} r_{t+1})^2],
$$

subject to $\mathbb{E}[m_{t+1}] = 1$, where $r_{t+1}$ is the vector of excess returns for contracts in Y.

**Compare:** HJ distance for Model A vs. B vs. C. Lower distance = better pricing.

### 5.4 Decision Rule

We say **“X prices the cross-section of Y”** if:

1. **X-only competitive:** $\overline{\alpha^2}_B \leq 1.5 \times \overline{\alpha^2}_A$.
2. **Marginal Y:** $\overline{\alpha^2}_C - \overline{\alpha^2}_B \leq 0.5 \times \overline{\alpha^2}_B$.
3. **Formal test:** We cannot reject $H_0: \alpha_i = 0 \ \forall i \in Y$ in Model B at the 5% level (GRS test or joint Wald test on alphas).

The 1.5× and 0.5× thresholds are heuristic; the **primary statistical criterion** is (3). Vary thresholds in robustness.

### 5.5 Out-of-Sample Validation

**Rationale:** In-sample fit can be regime-dependent. A single train/test split (e.g., 70/30) is sensitive to which period is held out—pre- vs post-2008, COVID, etc. OOS validation ensures that “X prices Y” is not an artifact of a particular subsample.

**Primary procedure: Expanding window (pseudo–out-of-sample).**

1. **Initial estimation window:** $t = 1, \ldots, T_0$ with **minimum training length 10 years** (120 months). No OOS forecasts before $T_0$.
2. **For each forecast origin** $t = T_0, T_0+1, \ldots, T-1$:
   - Estimate PCA on returns in X and Y using data from $1$ to $t$ only (no look-ahead).
   - Construct F_X, F_Y from these in-sample PCs.
   - Estimate betas and alphas for contracts in Y using data $1, \ldots, t$.
   - Compute OOS pricing metrics (e.g., mean squared alpha, HJ distance) on month $t+1$ only.
3. **Aggregate:** Average OOS metrics across all forecast origins. Compare Model A vs B vs C on OOS performance.
4. **OOS test statistic:** Use a formal test (e.g., Diebold–Mariano or Clark–West) to test whether Model B’s OOS pricing performance differs significantly from Model A’s. This provides a statistical test of “does X price Y?” in OOS space, not just point estimates.

**Alternative: Rolling window.** Same logic, but use a fixed-length training window (e.g., 60 months) that slides forward. More robust to regime shifts but uses less data. Report as robustness.

**What we do *not* use:**

- **Bootstrap:** Appropriate for **inference** (standard errors, confidence intervals, p-values), not for OOS validation. Bootstrap resamples the sample; it does not create new future data.
- **Random K-fold cross-validation:** Breaks temporal order and induces data leakage (train on future, test on past). Inappropriate for time series.
- **Blocked K-fold:** Can work for stationary series (Bergmeir et al. 2018) but is less reliable for non-stationary finance data (Cerqueira et al. 2020). Expanding/rolling window is the finance standard (Welch–Goyal, Rapach et al.).

**Econometric implications:** OOS validation provides a **distribution-free** check that pricing relationships hold in unseen periods. Report both in-sample (full sample) and OOS results; the formal statistical tests (GRS, Shanken) remain in-sample, while OOS quantifies robustness across regimes.

---

## 6. Implementation Details

### 6.1 Data Pipeline

1. **Monthly returns:** Use `async_monthly.csv` directly, or construct from daily/settlement prices (last trading day of month per Section 3.5).
2. **Currency:** Convert all to USD.
3. **PCA input:** Build a balanced or unbalanced panel; document handling of missing values (pairwise vs. listwise).

### 6.2 Factor Construction Order

1. For each asset class X: build **F_X** (PCs from returns of contracts in X).
2. Same for Y: build **F_Y**.

### 6.3 Computational Scope

- **Pairs:** All ordered (X, Y) with X ≠ Y. With 6 classes, that’s 30 pairs.
- **Feasibility:** PCA is O(min(N²T, NT²)); Fama–MacBeth is O(N × T). With ~100 contracts and ~400 months, this is tractable.

---

## 7. Pricing Network

### 7.1 Construction

- **Nodes:** Asset classes (equity, bond, commodity, currency, STIR, volatility).
- **Directed edges:** X → Y with weight $w_{XY}$.

**Weight (cross-sectional only):**

$$
w_{XY} = 1 - \frac{\overline{\alpha^2}_B}{\overline{\alpha^2}_A}
$$

(fraction of Y’s pricing error explained by X’s factors). Alternative: use HJ distance ratio.

### 7.2 Interpretation

- **Pricing hubs:** Classes with high **out-degree** price many other classes.
- **Price takers:** Classes with high **in-degree** are priced by many others.
- **Pairwise facts:** Equity–bond, volatility–equity, FX–commodity should appear as edges.

### 7.3 Visualization

- Directed graph with edge thickness = weight.
- Color nodes by asset class.

---

## 8. Robustness and Extensions

### 8.1 Robustness

1. **Subperiods:** Pre-2008 vs. post-2008; COVID exclusion. Tests whether pricing is regime-dependent.
2. **Number of PCs:** Vary $K_X$, $K_Y$ (e.g., 2 or 3 per class); check stability of network.
3. **PCA variant:** Correlation vs. covariance; RP-PCA (Lettau–Pelger) if incorporating means.
4. **Inference:** Block bootstrap confidence intervals for alphas and network weights (bootstrap for inference, not OOS).
5. **Volatility scaling:** Re-run with scaled returns; compare raw vs. scaled.
6. **OOS rolling window:** Compare expanding vs. fixed-length rolling window; rolling is more robust to regime shifts.

### 8.2 Extensions

1. **Higher frequency (weekly):** After monthly foundations, explore weekly returns.
2. **Conditional models:** Interact factors with macro state (recession dummy, VIX level).

---

## 9. Paper Structure (Proposed)

1. **Introduction:** Motivation, research question, contribution.
2. **Conceptual framework:** SDF, definition of “X prices Y” (cross-sectional), economic interpretation.
3. **Data and factor construction:** Universe, frequency, class-specific PCA, economic interpretation of PCs.
4. **Econometric framework:** Fama–MacBeth, inference (Newey–West, Shanken, GRS), OOS validation.
5. **Cross-sectional results:** Model comparison (A vs. B vs. C), HJ distance, GRS tests, OOS performance.
6. **Pricing network:** Construction, visualization, interpretation (hubs, takers).
7. **Robustness:** Subperiods, PC count, volatility scaling, rolling-window OOS.
8. **Conclusion:** Summary, limitations, future work.

---

## 10. Key Equations Reference

| Object | Equation |
|--------|----------|
| Linear SDF | $m_{t+1} = 1 - b'(f_{t+1} - \mathbb{E}[f])$ |
| Time-series | $r_{i,t} = \alpha_i + \beta_i' f_t + \varepsilon_{i,t}$ |
| X-native factors | F_X = first $K_X$ PCs of returns in class X (main: $K_X = 1$) |
| Y-native factors | F_Y = first $K_Y$ PCs of returns in class Y (main: $K_Y = 1$) |
| Model A | $f_t = f^Y$ |
| Model B | $f_t = f^X$ |
| Model C | $f_t = [f^X, f^Y]$ |
| Cross-sectional criterion | $\overline{\alpha^2}_B \leq 1.5 \overline{\alpha^2}_A$, $\overline{\alpha^2}_C - \overline{\alpha^2}_B \leq 0.5 \overline{\alpha^2}_B$, cannot reject $\alpha = 0$ in B |

---

## 11. Implementation Checklist

- [ ] Load or construct monthly returns (`async_monthly.csv` or last-trading-day-of-month).
- [ ] Build **class-specific PCs** (F_X, F_Y) for each asset class.
- [ ] Run Fama–MacBeth for models A, B, C for each (X, Y).
- [ ] Compute HJ distance (optional).
- [ ] **OOS validation:** Expanding-window pseudo-OOS (min 10 years training); re-estimate PCA and betas at each origin; Diebold–Mariano or Clark–West for OOS test; no look-ahead.
- [ ] Build pricing network; visualize.
- [ ] Robustness: subperiods, PC count, volatility scaling, rolling-window OOS.
- [ ] Extension (later): weekly frequency.

---

## 12. PCA Literature (Selected)

| Paper | Contribution |
|-------|--------------|
| Connor & Korajczyk (1988) | Asymptotic PCA for asset returns; factors as linear combinations |
| Bai & Ng (2002) | Determining number of factors in large panels |
| Stock & Watson (2002) | Factor extraction for macro/finance |
| Kelly, Pruitt & Su (2019) | "Characteristics Are Covariances"; RP-PCA |
| Lettau & Pelger (2020) | Risk-premium PCA (RP-PCA) |
| Kozak, Nagel & Santosh (2020) | "Shrinking the Cross-Section"; few factors span cross-section |
| Gibbons, Ross & Shanken (1989) | GRS test for joint $\alpha = 0$ |
| Shanken (1992) | Correction for estimated betas in Fama–MacBeth |
| Cerqueira et al. (2020) | OOS vs. CV for time series; holdout repeated in multiple periods |
| Bergmeir et al. (2018) | Blocked CV for stationary series; expanding/rolling for non-stationary |

---

## 13. Referee Comments (Anticipated)

*Potential concerns a referee might raise, and how the design addresses them:*

1. **Look-ahead bias in OOS:** PCA and betas must be re-estimated at each forecast origin using only data through $t$. No future information in factor construction or estimation. Explicit in Section 5.5.

2. **Single split regime dependence:** Addressed by expanding-window OOS with many forecast origins. Each month is tested OOS; results are averaged across regimes.

3. **Economic mechanism:** “X prices Y” is defined as shared SDF exposure (Section 2.2). The economic content is risk sharing and factor exposure, not mere correlation. PCA factors are statistical; the pricing test asks whether they span the cross-section.

4. **Power of GRS test:** With small $N_Y$, power may be low. Report effect sizes ($\overline{\alpha^2}$, HJ distance) alongside p-values. Multiple-testing adjustment (BH, Holm) controls for 30 pairs.

5. **Why PCA and not economic factors?** PCA is mechanical and replicable; it avoids the circularity of testing “does X price Y?” with factors built from Y. The design is deliberately statistical to allow a clean cross-class comparison.

6. **Survivorship / selection bias:** Document the contract universe and any exclusions. Use tier1/tier2 config; report coverage by class and time period.

7. **Pricing vs. predictability:** We test **pricing** (shared SDF exposure, $\alpha = 0$), not return predictability. Contemporaneous factor exposure identifies risk premia; we do not require X to forecast Y. The GRS test and HJ distance are pricing metrics.

8. **Decision-rule thresholds (1.5×, 0.5×):** Heuristic; the formal GRS test is primary. Report sensitivity to alternative thresholds in robustness.

---

*This outline is designed to be directly translatable into code and paper structure. All definitions, equations, and decision rules are explicit and reproducible.*
