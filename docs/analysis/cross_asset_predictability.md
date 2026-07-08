# Cross-Asset Predictability Research Outline (MOP-Compatible)

This note is the source of truth for the cross-asset predictability experiment.
Core question:

> Does market $x$ contain incremental information about next-month returns in
> market $y$, beyond $y$'s own lagged return information?

Design principle: change one thing at a time relative to MOP.  
Here, the only new ingredient in baseline tests is cross-market regressors.

---

## 1) Scope and Philosophy

- Keep MOP-style scaling, timing, and lag structure.
- Baseline has no macro controls.
- Start with representative contracts for transparency:
  `ES` (equity), `ZN` (bond), `CL` (commodity), `6J` (Japanese Yen).
- Separate two concepts cleanly:
  - same-asset-class predictability (e.g., commodity $\to$ commodity),
  - cross-asset-class predictability (e.g., bond $\to$ commodity).

---

## 2) Data Inputs

- Monthly returns: `DATASETS_ROOT/tier1/async/async_monthly.csv`.
- Daily returns for ex-ante volatility:
  `DATASETS_ROOT/tier1/async/async_daily.csv`.

All predictors and scalers must be known at forecast origin.

---

## 3) Ex-Ante Volatility (MOP-Style)

For each contract $i$:

$$
\sigma^2_{i,t}
= 261 \sum_{j=0}^{\infty}(1-\delta)\delta^j\left(r_{i,t-1-j}-\bar r_{i,t}\right)^2
$$

with:

- $\delta/(1-\delta)=60$ (so $\delta=60/61$),
- $\bar r_{i,t}$ as the exponentially weighted mean under the same weights.

Then:

$$
\sigma_{i,t}=\sqrt{\sigma^2_{i,t}}
$$

Timing:

- Monthly return in $t+1$ is scaled by $\sigma_{i,t}$.
- Lagged predictor at horizon $h$ uses $\sigma_{i,t-h-1}$.

---

## 4) Predictor and Baseline Regression Objects

For lag horizon $h$:

$$
z^{(h)}_{i,t}=\frac{r_{i,t-h}}{\sigma_{i,t-h-1}}
$$

Primary cross-asset equation for each ordered pair $(x \to y)$:

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}
= \alpha_{xy,h}
+ \beta_{xy,h} z^{(h)}_{x,t}
+ \gamma_{xy,h} z^{(h)}_{y,t}
+ \varepsilon_{xy,t+1}
$$

- $\beta_{xy,h}$: incremental cross-market predictability.
- $\gamma_{xy,h}$: own-market control.

Optional sign-based MOP diagnostic:

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}
= \alpha_{xy,h}
+ \beta^{\mathrm{sign}}_{xy,h}\mathrm{sign}(r_{x,t-h})
+ \gamma^{\mathrm{sign}}_{xy,h}\mathrm{sign}(r_{y,t-h})
+ \varepsilon_{xy,t+1}
$$

---

## 5) Apples-to-Apples Rules

To claim comparability with MOP-style outputs, hold fixed:

1. Sample dates (same start/end months; Stage A1 uses 1985-01 to 2009-12).
2. Dependent-variable scaling ($r_{t+1}/\sigma_t$).
3. Regressor scaling/timing ($r_{t-h}/\sigma_{t-h-1}$).
4. Horizon $h$.
5. Inference method (HAC vs clustered) within a comparison block.
6. Row set used in estimation (common non-missing sample when comparing models).

What is **not** apples-to-apples:

- Comparing a univariate MOP-style own-lag $t$-stat directly to a multivariate
  own-lag $t$-stat without qualification.
- Comparing coefficients across models that use different row sets.
- Comparing HAC-based $t$-stats to clustered-$SE$ $t$-stats as if equivalent.

---

## 6) Best Comparison Framework (Nested and Tractable)

For each target $y$, source $x$, and horizon $h$, build one common-sample panel
$\mathcal{D}_{xyh}$ that has all variables needed by all nested models.

Estimate the following on the **same** $\mathcal{D}_{xyh}$:

### Model M0 (Own-Only Benchmark)

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}=\alpha+\phi z^{(h)}_{y,t}+u_{t+1}
$$

### Model M1 (Cross-Only Benchmark)

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}=\alpha+\psi z^{(h)}_{x,t}+u_{t+1}
$$

### Model M2 (Joint Incremental Test)

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}=\alpha+\beta z^{(h)}_{x,t}+\gamma z^{(h)}_{y,t}+u_{t+1}
$$

Primary incremental question:

$$
H_0:\beta=0 \quad \text{in M2}
$$

Report for M0/M1/M2:

- coefficients and $t$-stats,
- adjusted $R^2$,
- $\Delta R^2$ from M0 $\to$ M2,
- $n_{obs}$ (same across M0/M1/M2 by construction).

Why this is tractable:

- only 3 models per $(x,y,h)$,
- fixed regressor count (max 2 + intercept),
- no macro block in baseline,
- no model-search over many controls.

---

## 7) Exact Experiment Procedure (Checklist)

Horizons: $h \in \{1, 2, \dots, 60\}$ (inclusive, MOP-consistent).

---

### Stage A0: MOP Replication Anchor

- [ ] Replicate MOP Fig. 1 style regressions and plotting (already implemented).
- [ ] Confirms scaling, timing, and inference machinery.

---

### Stage A1: Representative-Contract Cross Extension

- [ ] Sample period: 1985-01 to 2009-12 (MOP sample).
- [ ] Universe: `ES`, `ZN`, `CL`, `6J` (Japanese Yen).
- [ ] Pairs: all ordered $(x \to y)$, $x \ne y$ (12 equations).
- [ ] Horizons: $h = 1, 2, \dots, 60$.
- [ ] For each $(x, y, h)$: build common sample $\mathcal{D}_{xyh}$, run M0/M1/M2.
- [ ] Report coefficients, $t$-stats, adjusted $R^2$, $\Delta R^2$, $n_{obs}$.
- [ ] Plot by pair: horizons on $x$-axis; overlay $t(\phi)$, $t(\beta)$, $t(\gamma)$ (left axis) and $\Delta R^2$ (right axis).

---

### Stage A2: Same-Class vs Cross-Class Decomposition

- [ ] For each target $y$: choose one same-class source $x_{\text{same}}$ (if available).
- [ ] For each target $y$: choose one cross-class source $x_{\text{cross}}$ (pre-specified).
- [ ] Run M2 for same-class pair $(x_{\text{same}} \to y)$.
- [ ] Run M2 for cross-class pair $(x_{\text{cross}} \to y)$.
- [ ] Compare $\beta$ magnitudes and significance across same vs cross.

---

### Stage A3 (Optional): Pooled Panel Stage

- [ ] Proceed only after A1/A2 are stable.
- [ ] Expand to broader contract set; run pooled class-level regressions.

---

## 8) Should We Keep Representative Contracts?

Yes, for baseline identification and interpretability.

- Pros: easy to audit, low-dimensional, clean economic narrative.
- Cons: can miss within-class heterogeneity.

Resolution: keep representative contracts in Stage A1, then add heterogeneity in
Stage A2/A3 in a controlled way.

---

## 9) Potential Pooled Stage (When Expanding Universe)

Pooled cross-asset estimation is feasible when we have many target instruments.

Define class-level source predictor:

$$
z^{(h)}_{x,t,\mathrm{class}}
= \frac{1}{N_{x,t}}\sum_{i \in x}\frac{r_{i,t-h}}{\sigma_{i,t-h-1}}
$$

Stack target instruments $j$ in class $y$:

$$
\frac{r_{j,t+1}}{\sigma_{j,t}}
= \alpha_j
+ \beta_{x\to y,h} z^{(h)}_{x,t,\mathrm{class}}
+ \gamma z^{(h)}_{j,t}
+ u_{j,t+1}
$$

- $\alpha_j$: instrument fixed effects.
- $\beta_{x\to y,h}$: pooled cross-asset-class effect.

Inference:

- cluster by month at minimum,
- optionally two-way cluster by month and instrument.

---

## 10) Interpretation Guide

Given M2:

$$
\frac{r_{y,t+1}}{\sigma_{y,t}}=\alpha+\beta z^{(h)}_{x,t}+\gamma z^{(h)}_{y,t}+u
$$

- $\beta$ significant, $\gamma$ significant: both cross and own channels matter.
- $\beta$ significant, $\gamma$ weak: target is mostly externally predictable.
- $\beta$ weak, $\gamma$ significant: little incremental cross-asset evidence.
- $\beta$ changes sign/size materially from M1 to M2: strong overlap between
  source and target lag signals (interpret as conditional, not marginal, effect).

### Comparing $t(\phi)$ vs $t(\gamma)$

- **$\phi$** (M0): coefficient on target's own lag; *unconditional* own effect.
- **$\gamma$** (M2): coefficient on target's own lag *controlling for* cross-asset predictor; *conditional* own effect.

**When $t(\phi) \approx t(\gamma)$ despite adding a new predictor:**

1. **Near-orthogonality of predictors.** If $z^{(h)}_x$ and $z^{(h)}_y$ are nearly uncorrelated, adding $z^{(h)}_x$ barely changes the coefficient or standard error of the own predictor. So $\phi \approx \gamma$ and $t(\phi) \approx t(\gamma)$. The cross predictor contributes information without "stealing" the own signal.

2. **Collinearity with offsetting effects.** If $z^{(h)}_x$ and $z^{(h)}_y$ are correlated, $\gamma$ can differ from $\phi$. But if the coefficient change is offset by a change in the standard error (e.g., both shrink), the $t$-stat can stay similar. Similar $t$-stats do *not* imply similar coefficients.

3. **Weak cross predictor.** If $z^{(h)}_x$ has little explanatory power, the projection of $z^{(h)}_y$ onto it is small. The residual in the Frisch–Waugh step is close to $z^{(h)}_y$, so $\gamma \approx \phi$ and $t(\gamma) \approx t(\phi)$.

**Implications when $t(\phi) \approx t(\gamma)$:**

- The own predictor's marginal contribution (per unit of variation) is similar with or without the cross predictor. The cross predictor is not absorbing the own signal in a way that changes the $t$-stat.
- **Caveat:** Similar $t$-stats do *not* imply similar coefficients. With collinearity, $\gamma$ can be much smaller than $\phi$ while $t(\gamma) \approx t(\phi)$ if the SE of $\gamma$ falls proportionally.
- **Practical check:** Compare $\phi$ and $\gamma$ directly. If $\phi \approx \gamma$ and $t(\phi) \approx t(\gamma)$, predictors are roughly orthogonal. If $\phi \neq \gamma$ but $t(\phi) \approx t(\gamma)$, collinearity is affecting the coefficient but not the $t$-stat.

**When $t(\phi)$ and $t(\gamma)$ differ:**

- **$t(\phi)$ large, $t(\gamma)$ small:** The cross predictor absorbs most of the own signal; strong overlap.
- **$t(\phi)$ small, $t(\gamma)$ large:** The own effect appears mainly when controlling for the cross predictor; the cross predictor acts as a confound or suppressor.

### Same-class vs cross-class tests

- stronger same-class $\beta$: within-class transmission dominates,
- stronger cross-class $\beta$: true cross-asset spillover channel.

---

## 11) Baseline Output Requirements

For every $(x,y,h)$ and model (M0/M1/M2), output:

- coefficient and $t$-stat,
- adjusted $R^2$,
- $n_{obs}$,
- model label and horizon.

Also output a compact comparison table with:

- $\Delta R^2$ (M0 $\to$ M2),
- incremental-test $p$-value for $H_0:\beta=0$,
- same-class vs cross-class flags where relevant.

---

## 12) Deferred Items

- Trading strategy construction and portfolio backtests (later stage).
- Macro controls (CPI YoY first robustness).
- Large predictor menus or kitchen-sink variants.
