# src/globalmacro/validation/spot_fx.py
"""Exercise (c): the two spot FX panels against each other.

Every USD return depends on these. If a panel is stale, mis-oriented or has a
redenomination break, the USD identity r_usd = (1+r_local)(1+r_fx)-1 STILL HOLDS PERFECTLY
and every USD return is silently wrong -- so the identity is not the thing to check. The
panels are.

Exercise (a) reaches only the currencies with a CIP synthetic. HKD and SGD have no currency
future at all, and CAD has one (6C) but no synthetic, so cross-vendor comparison is the only
independent check those will ever get.

Correlation cannot validate a peg -- HKD's returns are almost pure snapshot noise (0.5%
annualised vol), so its 0.765 is meaningless, not alarming. The LEVEL check is what
validates it.

What the level gate does and does not catch. It is the MEDIAN of |async/sync - 1| over a
currency's whole history, against a 0.5% tolerance, and observed deviations run 0.00008 to
0.0019 -- three orders of margin. That reliably catches anything structural: an inverted
quote (USD-per-unit vs unit-per-USD), a units error (major vs minor unit, 100x), a stale or
wrong column, a redenomination spanning most of a series. It CANNOT catch a break confined
to less than half a currency's dates -- a 1000x error over 45% of JPY's history passes.

Gating per-year instead would not fix that; it would break the check. A per-year median at
0.5% breaches on seven currencies for entirely legitimate reasons: EUR 3.96% in 1991
(synthesised pre-EMU), MYR 11.05% in 1999 (capital controls), THB 5.96% in 2007, plus
KRW/MXN/TWD/CNY in the 80s-90s. Those are real vendor disagreements during currency crises,
not defects, and gating on them would false-alarm on true history. So the gate stays a
full-history median, and the worst year is REPORTED (worst_annual_dev, worst_year) for a
human to read rather than used to fail the run.
"""
from __future__ import annotations

import polars as pl

from globalmacro.build import build_currency_map, compute_monthly_returns
from globalmacro.utils.config import load_config
from globalmacro.utils.paths import FX_PATH, PROJECT_ROOT
from globalmacro.validation.base import Check, Invariant
from globalmacro.validation.synthetic import load_panel, shipped_panel

NAME = "Spot FX panels"
SLUG = "spot_fx"

PEG_VOL_THRESHOLD = 0.02     # annualised; HKD is 0.005, next-lowest (SGD) is 0.045
LEVEL_TOLERANCE = 0.005      # median |async/sync - 1|; currently 0.00008 - 0.0019

# The peg filter removes a currency from the graded median, so it is a denominator that
# adjusts ITSELF: a currency whose feed went stale has ~zero vol, would be reclassified a
# peg, and would leave the graded set with a green tick. Pin the set. A NEW peg is not a
# reason to grade less -- it is a reason to stop and look.
EXPECTED_PEGS = frozenset({"HKD"})


def _shipped_currencies() -> list[str]:
    """The foreign currencies the shipped USD panels actually convert through.

    TIER 2, not tier 1: save_usd_datasets converts all four panels -- tier1 AND tier2, sync
    and async -- through these same two FX files, and tier 2 is the superset. Reading tier 1
    here checked 11 of the 21 currencies, leaving BRL, CNY, DKK, MXN, MYR, NOK, PLN, THB,
    TWD and ZAR unchecked -- precisely the currencies where an orientation or unit error is
    most likely to exist and least likely to be noticed.
    """
    futures = load_config(PROJECT_ROOT / "tier1.yaml") + load_config(PROJECT_ROOT / "tier2.yaml")
    ccy = build_currency_map(futures)
    symbols = [c for c in shipped_panel("async", tier=2).columns if c != "date"]
    return sorted({ccy[s] for s in symbols if s in ccy} - {"USD"})


def _panels() -> tuple[pl.DataFrame, pl.DataFrame]:
    # fx_sync.csv is read unconditionally here, and this check stays in the async-only set
    # (it is not requires_sync=True). That is deliberate, not an oversight: Compustat is a
    # pipeline-wide prerequisite, not a tick-data one -- build.load_synthetic_returns reads
    # its derivative (synthetic_fx_returns_sync.csv) in EVERY mode, async-only included.
    # `globalmacro build` itself, in every mode, cannot succeed without
    # that file, which fx.py writes to this same directory strictly AFTER fx_sync.csv in one
    # invocation -- so by the time a validate run has anything built to grade, fx_sync.csv is
    # already on disk. A missing-file error here would mean Compustat access itself is
    # broken, which should fail loudly, not be silently downgraded through a second
    # capability axis.
    return load_panel(FX_PATH / "fx_async.csv"), load_panel(FX_PATH / "fx_sync.csv")


def _metrics() -> pl.DataFrame:
    fx_async, fx_sync = _panels()
    rows = []
    for ccy in _shipped_currencies():
        if ccy not in fx_async.columns or ccy not in fx_sync.columns:
            continue
        j = (
            fx_async.select("date", pl.col(ccy).alias("a"))
            .join(fx_sync.select("date", pl.col(ccy).alias("s")), on="date", how="inner")
            .filter(pl.col("a").is_not_null() & pl.col("s").is_not_null() & (pl.col("s") != 0))
            .sort("date")
        )
        if j.height < 500:
            continue

        # LEVELS: same quantity (USD per unit), different snapshot time.
        dev = ((pl.col("a") / pl.col("s")) - 1).abs()
        level_dev = j.select(dev.median()).item()

        # The gate above is a full-history median and is blind to a partial-period break;
        # the worst YEAR is therefore reported, never gated (see the module docstring --
        # per-year gating false-alarms on EUR pre-EMU, MYR 1999, THB 2007).
        annual = (
            j.group_by(pl.col("date").dt.year().alias("year"))
            .agg(dev.median().alias("d"))
            .drop_nulls("d")
            .sort("d", descending=True)
        )
        worst_annual_dev = annual.get_column("d")[0] if annual.height else None
        worst_year = int(annual.get_column("year")[0]) if annual.height else None

        # RETURNS: gap-aware is unnecessary here -- both panels are forward-filled dense.
        r = j.with_columns(
            [(pl.col(c) / pl.col(c).shift(1) - 1).alias(f"r{c}") for c in ("a", "s")]
        ).drop_nulls(["ra", "rs"])
        vol = r.select(pl.col("ra").std()).item() * (252 ** 0.5)

        ma = compute_monthly_returns(r.select("date", pl.col("ra").alias("x")))
        ms = compute_monthly_returns(r.select("date", pl.col("rs").alias("y")))
        m = (
            ma.select("year_month", "x")
            .join(ms.select("year_month", "y"), on="year_month", how="inner")
            .filter(pl.col("x").is_not_null() & pl.col("y").is_not_null())
        )
        corr_m = m.select(pl.corr("x", "y")).item() if m.height >= 24 else None
        corr_d = r.select(pl.corr("ra", "rs")).item()

        is_peg = vol < PEG_VOL_THRESHOLD
        rows.append(
            {
                "instrument": ccy,
                "correlation": corr_m,
                "n_obs": m.height,
                "corr_daily": corr_d,
                "level_dev": level_dev,
                "worst_annual_dev": worst_annual_dev,   # reported, never gated
                "worst_year": worst_year,               # reported, never gated
                "ann_vol": vol,
                "is_peg": is_peg,
                # Pegs are reported but not graded: correlating near-zero returns is
                # meaningless. The level check is what validates them.
                "used": (not is_peg) and corr_m is not None,
            }
        )
    if not rows:
        # Explicit schema: pl.DataFrame([]) has NO columns, so run.py's filter(pl.col("used"))
        # and _invariants() below would raise ColumnNotFoundError instead of reporting 0/0.
        return pl.DataFrame(
            schema={
                "instrument": pl.Utf8, "correlation": pl.Float64, "n_obs": pl.Int64,
                "corr_daily": pl.Float64, "level_dev": pl.Float64,
                "worst_annual_dev": pl.Float64, "worst_year": pl.Int64,
                "ann_vol": pl.Float64, "is_peg": pl.Boolean, "used": pl.Boolean,
            }
        )
    return pl.DataFrame(rows).sort("correlation", descending=True, nulls_last=True)


def _invariants() -> list[Invariant]:
    m = _metrics()
    worst = m.select(pl.col("level_dev").max()).item()
    breaches = m.filter(pl.col("level_dev") > LEVEL_TOLERANCE)
    ok = m.height > 0 and breaches.height == 0
    detail = (
        f"max {worst * 100:.3f}% over {m.height} currencies (tol {LEVEL_TOLERANCE * 100:.1f}%)"
        if ok
        else f"BREACH: {sorted(breaches['instrument'].to_list())}"
    )
    pegs = set(m.filter(pl.col("is_peg"))["instrument"].to_list())
    unexpected = sorted(pegs - EXPECTED_PEGS)
    absent = sorted(EXPECTED_PEGS - pegs)
    peg_detail = ", ".join(sorted(pegs)) if pegs else "none"
    if unexpected:
        peg_detail += f" (UNEXPECTED: {unexpected} -- stale feed?)"
    if absent:
        peg_detail += f" (MISSING: {absent})"
    return [
        Invariant(
            check=NAME,
            name="the two FX panels agree in level, per currency",
            value=detail,
            passed=ok,
        ),
        Invariant(
            check=NAME,
            name=f"the graded median excludes exactly the pinned pegs {sorted(EXPECTED_PEGS)}",
            value=peg_detail,
            passed=pegs == set(EXPECTED_PEGS),
        ),
    ]


spot_fx_check = Check(
    name=NAME,
    slug=SLUG,
    run=_metrics,
    invariants=_invariants,
)
