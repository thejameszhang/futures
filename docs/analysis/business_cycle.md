# Business Cycle Dating and Phase Construction

This note documents how the business cycle phases are built in
`src/analysis/notebooks/02_heterogeneity.ipynb` and how the conditional
returns table is computed.

## Data sources and conventions

- NBER monthly business cycle turning points (peaks and troughs) are used.
  The notebook embeds the NBER monthly list that matches the FRED recession
  bars dates.
- NBER convention:
  - The peak month is the last month of an expansion.
  - The trough month is the last month of a recession.
  - Therefore recession months are the month after the peak through the
    trough month inclusive.
- The monthly peak/trough dates are treated as month-level dates. Returns
  are monthly and timestamped at month-end. We map each return to its
  calendar month via `YYYY-MM` (e.g., 1981-07-31 -> 1981-07) and then apply
  the phase label for that month.

## Expansion and recession months

Given consecutive NBER turning points (peak, trough, next peak):

- Recession months:
  - Start: month after the peak (`peak + 1 month`, month-start).
  - End: trough month inclusive.
- Expansion months:
  - Start: month after the trough (`trough + 1 month`, month-start).
  - End: next peak month inclusive.

This aligns with the NBER definition that the peak month is expansion and
the trough month is recession.

## Early and late phases

Following Gorton and Rouwenhorst (2006, Figure 6 and Section 8):

- Each recession interval (peak -> trough) is split into two equal halves:
  - Early Recession: first half of recession months.
  - Late Recession: second half of recession months.
- Each expansion interval (trough -> next peak) is split into two equal halves:
  - Early Expansion: first half of expansion months.
  - Late Expansion: second half of expansion months.

When the number of months in an interval is odd, the extra month is assigned
to the late phase (i.e., late gets `ceil(n/2)`, early gets `floor(n/2)`).

Only months that fall within a fully defined NBER interval are included.
Months after the last known trough or before the first known peak are not
used in phase splits.

## Series included in the table

The conditional returns table uses the same column labels as the risk
premium and moments tables:

- `S&P500`: `ES` monthly returns from `dataset` when present.
- `US 10Y`: `ZN` monthly returns from `dataset` when present.
- `Equity`, `Bond`, `Commodity`, `FX`, `Rate`, `Volatility`: equal-weighted
  asset-class returns from `equally_weighted` in the notebook.

Columns are shown only when the underlying series is present.

## Computation of table entries

For each phase (rows) and each series (columns):

- Compute the mean of monthly returns.
- Annualize the mean as `mean_monthly * 12 * 100` (percent).

