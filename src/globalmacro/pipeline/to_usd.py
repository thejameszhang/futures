"""Convert local-currency return panels to USD as a post-process.

r_usd = (1 + r_local) * (1 + r_fx) - 1, with r_fx the return of the USD-per-local spot
over the symbol's OWN observation interval (gap-aware, so foreign holidays don't smear
the FX move). USD-denominated symbols pass through unchanged.

The two legs must span the SAME interval. r_local for day t is priced off the symbol's
previous OBSERVATION, so r_fx has to be measured from that same date -- not from the
previous calendar row, which would smear a foreign holiday into the FX move.

Which date that is used to be inferred from return nullity alone: a null return meant no
observation, because a symbol with no price had no row. That inference is no longer safe.
futures.py's cross-contract guard nulls returns on dates where the row AND the price still
exist, and it is not alone -- a leading partial month (filter_dataset_by_monthly_returns),
a hand-set start date (keep_after_date) and a symbol's own first price all produce the same
shape. In every one of those cases the NEXT day's local return is still a one-day return,
so widening the FX leg to the last surviving return puts the two legs on different
intervals. Hence `traded`: an explicit per-symbol observation mask supplied by the caller.
"""
from __future__ import annotations

import polars as pl

TRADED_PREFIX = "__traded__"


def _fx_return_over_observations(symbol: str, ccy: str, *, has_traded: bool) -> pl.Expr:
    level = pl.col(f"__fx__{ccy}")
    observed_here = pl.col(symbol).is_not_null()
    if has_traded:
        # A union, never a replacement. `traded` covers only the symbols whose returns come
        # from the Datastream futures parquet; the same panels also carry JKP sector columns
        # and a synthetic pre-listing backfill (spot equity / CIP FX) whose observations that
        # mask knows nothing about and reports as False. Or-ing keeps those on the old
        # return-nullity inference, and makes the correction one-directional: an FX interval
        # can only ever be narrowed towards a real observation, never widened.
        #
        # RESIDUAL: this union is also the source of the 16 (of ~1.17M) tier2 cells that
        # still land on the wrong FX interval after this fix (down from 1,686, measured
        # against ground truth date.shift(1).over('clscode')). 5 of the 16 are cells the OLD
        # return-nullity-only rule got RIGHT and this union gets WRONG: HG 2006-07-05,
        # SI 1988-04-04, SI 2002-12-02, SO3 2008-05-27, SO3 2010-05-04. All five are CT/CS
        # asymmetry -- `traded` is unioned across BOTH cycles, but the symbol's return on
        # these dates is priced by only ONE cycle (verified: CT), off THAT cycle's own row
        # grid. The other cycle (CS) prices an intermediate date CT does not (e.g. HG
        # 2006-07-03), and the union marks that date "traded" anyway -- a false intermediate
        # observation on a grid the actual return was never computed from, narrowing the FX
        # interval short of the true span.
        observed_here = observed_here | pl.col(f"{TRADED_PREFIX}{symbol}").fill_null(False)
    observed = pl.when(observed_here).then(level).otherwise(None)
    prev_observation = observed.forward_fill().shift(1)
    denom = pl.coalesce([prev_observation, level.shift(1)])
    return level / denom - 1


def _traded_symbols(symbols: list[str], traded: pl.DataFrame | None) -> set[str]:
    """The subset of `symbols` for which `traded` actually carries an observation mask.

    Partial coverage is the normal case, not an error: the mask is built from the futures
    parquet and the panels also carry sector and pre-listing-synthetic columns.
    """
    if traded is None:
        return set()
    if "date" not in traded.columns:
        raise ValueError("traded panel must carry a `date` column")
    # A duplicated date would fan the left join out and silently multiply panel rows.
    if traded.get_column("date").n_unique() != traded.height:
        raise ValueError("traded panel has duplicate dates")
    return {s for s in symbols if s in traded.columns}


def usd_panel(
    local: pl.DataFrame,
    symbol_to_ccy: dict[str, str],
    fx_levels: pl.DataFrame,
    traded: pl.DataFrame | None = None,
) -> pl.DataFrame:
    symbols = [c for c in local.columns if c != "date"]
    unmapped = sorted(set(symbols) - set(symbol_to_ccy))
    if unmapped:
        raise ValueError(f"No currency mapping for columns: {unmapped}")
    needed = sorted({symbol_to_ccy[s] for s in symbols} - {"USD"})
    absent = [c for c in needed if c not in fx_levels.columns]
    if absent:
        raise ValueError(f"fx_levels is missing currency columns: {absent}")

    local = local.with_columns([pl.col(s).cast(pl.Float64, strict=False) for s in symbols]).sort("date")
    if needed:
        fx = (fx_levels.select(["date"] + needed).rename({c: f"__fx__{c}" for c in needed})
              .with_columns([pl.col(f"__fx__{c}").cast(pl.Float64, strict=False) for c in needed]))

        # Seed the FX grid with the single most recent fx_levels row STRICTLY
        # BEFORE the panel's first date, so level.shift(1) has something to
        # resolve to on the panel's very first row. Sync panels begin exactly at
        # their first observation (no prior grid row) even though fx_levels may
        # carry the currency's history much further back; without this seed that
        # first observation's denom is null and the whole row drops to null USD.
        # The seed row carries no symbol values, so `observed` is null there — it
        # only ever supplies the level.shift(1) denominator, never a return.
        panel_start = local.select(pl.col("date").min()).item()
        seed = fx.filter(pl.col("date") < panel_start).sort("date").tail(1)
        if seed.height:
            seed_local = pl.DataFrame({"date": seed["date"]}).with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(s) for s in symbols])
            local_ext = pl.concat([seed_local.select(["date"] + symbols), local.select(["date"] + symbols)],
                                   how="vertical")
        else:
            local_ext = local

        frame = (local_ext.join(fx, on="date", how="left")
                 .with_columns([pl.col(f"__fx__{c}").forward_fill() for c in needed]))
        traded_symbols = _traded_symbols(symbols, traded)
        if traded is not None and traded_symbols:
            marks = traded.select(
                ["date"] + [pl.col(s).cast(pl.Boolean, strict=False).alias(f"{TRADED_PREFIX}{s}")
                            for s in sorted(traded_symbols)])
            frame = frame.join(marks, on="date", how="left")
    else:
        frame = local
        panel_start = None
        traded_symbols = set()

    out = []
    for s in symbols:
        ccy = symbol_to_ccy[s]
        if ccy == "USD":
            out.append(pl.col(s))
        else:
            r_fx = _fx_return_over_observations(s, ccy, has_traded=s in traded_symbols)
            out.append(((1.0 + pl.col(s)) * (1.0 + r_fx) - 1.0).alias(s))
    frame = frame.with_columns(out)
    if panel_start is not None:
        # Drop the seed row now that it has served its only purpose (supplying
        # the level.shift(1) denominator for the panel's real first row).
        frame = frame.filter(pl.col("date") >= panel_start)
    return frame.select(["date"] + symbols)
