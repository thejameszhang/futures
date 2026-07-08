You are a sophisticated research assistant for an empirical asset pricing project in financial economics.

## High-level goal

I’m writing a paper that asks, in a unified and systematic way:

> “Does asset class X price asset class Y?”

I want you to **frame the entire research outline from scratch**: conceptual foundations, empirical design, factor construction, tests, and interpretation. Do not just give a generic asset pricing template; tailor everything to the details below.

The paper should be:

- Explicitly **SDF / factor-pricing–oriented**, not VAR/connectedness–oriented.
- Centered on **payoff space / returns space**, not firm-level characteristics.
- **Tractable** and implementable with real data and reasonable computation.
- Structured to handle both:
  - **Individual contract–level** questions (“does the S&P 500 future price the 10‑year Treasury future?”),
  - And **cross-sectional** questions (“do equity index futures factors price the cross-section of bond futures?”).

## Data and setting

Assume I have a **large global futures dataset** spanning multiple macro/financial asset classes, including (but not necessarily limited to):

- Equity index futures (e.g., S&P 500, EuroStoxx, Nikkei, EM equity indices).
- Government bond futures (e.g., U.S. Treasury futures across maturities, Bunds, JGBs).
- Short rate / money market futures.
- Currency futures (G10, possibly EM).
- Commodity futures (energy, metals, agriculture, etc.).
- Possibly volatility futures (e.g., VIX) if helpful.

For each contract I can construct **weekly, daily, and monthly excess returns** in a common currency (e.g., USD), with standard mechanical rolling rules and careful handling of contract expiry. You should **choose and justify a primary frequency** for the analysis (likely weekly) and explain how to handle time zones, nonsynchronous trading, and announcement timing in a way consistent with the identification of “pricing” relationships.

## Core conceptual question

I want you to define, carefully and rigorously, what it means for:

> “Asset class X to price asset class Y”

at **two distinct levels**:

1. **Individual contract level**  
   For a given contract i in asset class Y (e.g., the 10‑year U.S. Treasury future), and a given asset class X (e.g., U.S. equity index futures), what does it mean empirically to say “factors from X price asset i”?

2. **Cross-sectional level within an asset class**  
   For the entire cross-section of futures in Y (e.g., all gov bond futures), what does it mean to say “asset class X prices the cross-section of Y”?

You should ground these definitions in **SDF / linear factor models** and related tests (alphas, HJ distance, etc.), not in VAR spillover or pure forecast variance decomposition.

## Factor construction requirements

I want a **tractable, systematic way** to construct candidate factors that can be used to test cross-asset pricing, subject to these constraints:

1. Factors should be based on **payoffs and returns**, not firm-level characteristics.
2. For each asset class X, I want **“native” factors** that represent that class’s payoff space:
   - Examples I have in mind:
     - An **equal-weighted index** of all futures in X (class-level “market” factor).
     - A **first principal component (PC1)** extracted from the cross-section of residual returns within X (e.g., regress each contract on the class EW index and run PCA on residuals).
   - You can extend or refine this construction if it improves robustness or interpretability, but keep it **mechanical and replicable**.

3. In addition, for some specific asset class pairs, I want factors that correspond to **“individual cross-asset facts”** that the literature highlights. For example:
   - For equities ↔ bonds:
     - Use the **S&P 500 futures return** explicitly as a candidate factor for Treasury futures.
     - Use the **10‑year Treasury futures return** explicitly as a candidate factor for equity index futures.
   - For other pairs where the literature documents clear time-series predictability or cross-asset pricing, allow analogous “single-contract factors” to enter.

4. I am **not** asking you to define factors via characteristics like size, value, momentum at the stock level. Factors should be:
   - Class-level payoff aggregates (EW indices, PCs over returns),
   - Or mechanically constructed strategies over the contracts themselves (e.g., time-series momentum portfolios, equal-weight long-only or long-short portfolios),
   - Or specific, economically important contracts (like S&P 500 future, 10‑year Treasury future) treated as standalone factors in pairwise tests.

Your job is to propose a **systematic factor architecture** that:

- Gives each asset class a set of “native” factors,
- Allows the inclusion of **specific individual contracts** as additional factors for particular pairs where economically justified,
- And remains tractable for a large number of contracts and classes.

## Asset pricing tests: what I want

I want you to design **two layers of tests**, both grounded in linear factor models:

### 1. Individual contract–level tests

For each ordered pair (X, Y) of asset classes and each contract i in Y:

- Specify a **benchmark model** that includes:
  - A set of **global factors** (constructed from returns/payoffs across all classes, e.g., a global market factor, global bond factor, etc.), and possibly Y’s own native factors.
- Specify an **X-augmented model** that adds X’s native factors and any relevant single-contract factors from X (e.g., S&P 500 future when X = equities).
- Define precise criteria for when we say:
  - “Asset class X helps price contract i in Y,” in terms of:
    - Reduction in alpha magnitude for i,
    - Improvement in R² or pricing error metrics,
    - Statistical significance handled in a way that respects multiple testing across many contracts and pairs.

I want you to be explicit about:

- The regression structure (time-series step),
- How alphas and betas are estimated,
- How standard errors are computed (e.g., HAC, Newey–West),
- And how to aggregate these results across contracts to summarize individual-level evidence.

### 2. Cross-sectional pricing tests within Y

For each ordered pair (X, Y):

- Specify a **set of competing factor models** for the cross-section of contracts in Y, such as:
  - Model A: global factors + Y’s native factors (Y-native model).
  - Model B: global factors + X’s native factors (X-only model).
  - Model C: global factors + X’s native factors + Y’s native factors (combined model).
  - Optionally include the relevant single-contract factors where economically justified (e.g., S&P 500 future, 10‑year future).

- Describe how to conduct **Fama–MacBeth or GMM** style tests to compare these models in terms of:
  - Average squared alpha and mean absolute alpha,
  - HJ distance or related distance measures,
  - Cross-sectional R².

I want a clear, operational decision rule for:

- When we conclude that “X prices the cross-section of Y,” in the sense that X’s factors **alone** come close to Y’s own native factors, and the combined model yields only marginal additional improvement.

## Frequency, synchronization, and identification

Please:

- Choose a **primary frequency** (likely weekly) and justify it carefully.
- Explain how you would:
  - Construct returns (e.g., Wednesday-to-Wednesday),
  - Handle nonsynchronous trading and time zones,
  - Deal with macro announcement timing and potential lead–lag effects (e.g., using lagged factors in the regressions).

Make sure the chosen frequency and synchronization strategy is coherent with interpreting coefficients and alphas as evidence of **pricing** relationships, rather than purely mechanical timing effects.

## Network and interpretation

Once the tests are defined:

- Propose how to summarize the results as a **directed “pricing network”**:
  - Nodes = asset classes,
  - Directed edge X → Y weighted by a **pricing ability score** based on the cross-sectional tests (and possibly informed by individual-level evidence).

Explain how to interpret this network in economic terms:

- Which classes emerge as **“pricing hubs”** (high out-degree)?
- Which are primarily **“price takers”**?
- How do known pairwise facts (equity–bond interactions, cross-asset TSM, FX risk, etc.) appear as **special cases** within this broader network?

## What I want from you

Given all of the above, please:

1. **Frame the full research outline**:
   - Motivation and contribution,
   - Conceptual framework for “X prices Y,”
   - Data and frequency choices,
   - Detailed factor construction scheme,
   - Individual contract–level asset pricing tests,
   - Cross-sectional asset pricing tests,
   - Construction and interpretation of the pricing network,
   - Robustness checks and extensions.

2. Make the outline **as concrete and operational as possible**:
   - Explicit equations for the key regressions,
   - Clear definitions of factor blocks,
   - Specific decision rules for what counts as “pricing” at individual and cross-sectional levels.

3. Keep everything **tractable**:
   - Avoid models that would be computationally prohibitive on a large global futures panel.
   - Use standard tools (linear factor models, Fama–MacBeth, GMM, PCA) in a disciplined way.

Your final output should read like a **research blueprint** that I could directly translate into code and, ultimately, into the structure of the paper.