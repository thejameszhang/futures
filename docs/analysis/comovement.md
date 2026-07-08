# Comovement Research Outline

## Paper Angle: Stylized Facts

This project documents time-varying comovement patterns across asset classes and regions. The goal is to present facts and patterns rather than to provide a single unifying answer. With the breadth of asset pairs, regions, and methods covered, the paper will emphasize interpretable, tractable results.

---

## 1. Daily Comovement: Core Framework

### 1.1 Methodology (Campbell 2025 style)

- **Regression:** Regress log return of asset Y on log return of asset X (OLS with constant).
- **Rolling window:** At each quarter-end, use the past 90 trading days.
- **Smoothing:** 12-quarter backward-looking moving average, requiring at least 6 quarters.
- **Output:** Time series of betas (comovement coefficients).

### 1.2 Implemented Pairs

| Pair Type       | Regions                                      | Status   |
|-----------------|----------------------------------------------|----------|
| Bond–Stock      | US, UK, Eurozone, Japan, Switzerland         | Done     |
| Currency–Stock  | UK, Eurozone, Japan, Switzerland             | Done     |
| Commodity–Stock | US, UK, Eurozone, Japan, Switzerland (WTI, Brent, NG, Gold) | Done |
| Volatility–Equity | US (VX/ES), Eurozone (FVS/FESX)              | Done     |
| Volatility–Bond   | US (VX/ZN), Eurozone (FVS/FGBL)              | Done     |
| Volatility–Gold/Silver | US (VX/GC, VX/SI)                         | Done     |
| Short-Rate–Equity | US (SR3/ES), Eurozone (I/FESX), UK (SO3/Z)  | Done     |

### 1.3 Figure Observations

Figures are saved to `src/analysis/notebooks/figures/` as `comovement_{name}.png`.

- **Bond–Stock:** US bond beta turns negative post-2000 in synchronous data (flight-to-quality), while UK/Eurozone/Switzerland/Japan remain positive. All regions show a notable upward trend in bond betas in the 2020s. Async vs sync choice materially affects US betas.
- **Currency–Stock:** Async betas are consistently positive; sync betas can turn negative (Japan stays negative 2004–2022). UK/Eurozone/Switzerland track each other; Japan diverges.
- **WTI–Stock:** Strong shift from low/negative pre-2000 to positive post-2005. Betas peak around GFC and COVID-19. US/UK highest; Japan lowest. Sync data shows a more negative US beta in the early 2000s.
- **Brent–Stock:** Similar to WTI: surge in positive comovement from mid-2000s, peaks around 2010–2012 and 2020. Sync betas slightly higher in magnitude. Japan and Switzerland dip negative post-2020.
- **Natural Gas–Stock:** More volatile and region-specific. US beta often negative (early 2000s, 2014–2019); Eurozone/Switzerland show early-2000s peaks in sync data. Recession impact varies by region.
- **Gold–Stock:** Betas mostly positive; US shows the largest swings and periods of negative beta (early 1990s, 2000, mid-2010s). GFC coincides with a positive spike across regions. Gold behaves as risk-on during GFC, safe-haven in other episodes.
- **Volatility–Equity:** Betas consistently negative (vol rises when equity falls). US more negative than Eurozone. Both weaken (less negative) during COVID-19. Async and sync plots are very similar.
- **Volatility–Bond:** Betas generally positive (flight-to-quality: vol and bond returns rise together in stress). US beta much larger and more volatile than Eurozone; peaks at GFC and COVID-19. Post-2020, US beta drops sharply and briefly turns negative.
- **Volatility–Gold/Silver:** Regime shift around 2015: from negative to strong positive gold–vol beta (2015–2019), then back toward zero/negative. Gold’s positive phase is stronger than silver’s. 2008 recession: more negative; 2020: brief rebound toward zero.

### 1.4 Extensions: Commodity–Stock Comovement

Commodity–stock betas capture how equity markets respond to commodity price moves (input costs, inflation, sector composition).

#### Domestic / Region-Specific Pairs

| Region    | Commodity | Equity   | Rationale                                                                 |
|-----------|-----------|----------|---------------------------------------------------------------------------|
| **US**    | CL (WTI)  | ES       | Oil as input cost, inflation proxy; US energy sector exposure             |
| **US**    | GC (Gold) | ES       | Flight to safety, inflation hedge; gold often negatively correlated      |
| **US**    | HG (Copper) | ES     | "Dr. Copper" – leading indicator of economic activity                     |
| **UK**    | BRN (Brent) | Z      | UK energy-heavy; Brent is UK/Europe benchmark                             |
| **UK**    | GC (Gold) | Z         | UK mining sector; gold as safe haven                                     |
| **Eurozone** | BRN       | FESX     | Eurozone energy exposure; Brent as regional benchmark                     |
| **Eurozone** | GC       | FESX     | Gold as safe haven                                                       |
| **Japan** | CL        | Nikkei   | Japan as energy importer; oil sensitivity                                 |
| **Japan** | HG (Copper) | Nikkei | Japan manufacturing; copper as demand indicator                           |
| **Switzerland** | GC     | FSMI     | Swiss gold sector; gold as safe haven                                   |

#### Cross-Country / International

Use a single global commodity as benchmark and regress each equity index on it:

| Commodity | Equities | Interpretation |
|-----------|----------|----------------|
| **CL (WTI)** or **BRN (Brent)** | ES, Z, FESX, Nikkei, FSMI | Oil sensitivity across countries; energy vs oil-importing |
| **GC (Gold)** | ES, Z, FESX, Nikkei, FSMI | Flight-to-safety across markets |
| **HG (Copper)** | ES, Z, FESX, Nikkei, FSMI | Cyclical demand sensitivity |

#### Additional Commodities (Optional)

- **Silver (SI)** – industrial metal, often tracks gold in stress periods
- **Natural gas (NG)** – US-specific; different drivers than oil
- **Agricultural** (e.g. wheat ZW, corn ZC) – food inflation, less standard for equity

---

## 2. Regime Splits

- **Recession vs expansion:** Compute betas separately for NBER recession and expansion months.
- **Interpretation:** Compare comovement strength and sign across regimes.
- **Implementation:** Use NBER dates already in the notebook; split sample by regime.

---

## 3. Comovement Matrix (Optional)

- **Concept:** Pairwise betas for a small set of assets (e.g. 5–6) at each quarter.
- **Output:** Time series of comovement matrices; identify which pairs flip sign or change structure.
- **Use:** Stylized facts on regime shifts (e.g. pre/post 2000, 2008).

---

## 4. Cross-Asset Predictability (Later)

- **Goal:** Move from contemporaneous comovement to lead–lag predictability.
- **Method:** \( r_{Y,t} = \alpha + \beta \, r_{X,t-k} \) for \( k = 1, 5, 22 \).
- **Pairs:** US equity → non-US equity; volatility → returns; bond → equity.
- **Extension:** Granger causality in rolling windows.

---

## 5. Spillovers and VARs (Later)

- **Goal:** Quantify how shocks propagate across assets.
- **Method:** Diebold–Yilmaz spillover (VAR-based variance decomposition).
- **Output:** Total spillover index; directional spillovers; pairwise spillovers.
- **Scope:** Start with 3–5 assets (e.g. US equity, US bond, UK equity, oil, gold).
- **Interpretation:** Time-varying spillover network; crisis vs calm periods.

---

## 6. Data Sources

- **Returns:** `async_daily.csv` (main), `sync_daily.csv` (comparison).
- **Regime:** NBER recession dates.
- **Asset universe:** `tier1.yaml` (futures symbols).

---

## 7. Implementation Order

1. **Commodity–stock pairs** – Add domestic and cross-country commodity–stock betas.
2. **Regime splits** – Recession vs expansion betas.
3. **Comovement matrix** – If time permits.
4. **Predictability** – Lagged regressions.
5. **Spillovers** – Diebold–Yilmaz.

---

## 8. Literature

- Campbell (2025) – Bond–stock comovement
- Diebold & Yilmaz (2009, 2012, 2014) – Spillovers
- Forbes & Rigobon (2002) – Contagion vs interdependence
- Longin & Solnik (2001) – Correlation in crises
