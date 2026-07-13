"""Convert local-currency return panels to USD as a post-process.

r_usd = (1 + r_local) * (1 + r_fx) - 1, with r_fx the return of the USD-per-local spot
over the symbol's OWN observation interval (gap-aware, so foreign holidays don't smear
the FX move). USD-denominated symbols pass through unchanged.
"""
from __future__ import annotations

import polars as pl


def _fx_return_over_observations(symbol: str, ccy: str) -> pl.Expr:
    level = pl.col(f"__fx__{ccy}")
    observed = pl.when(pl.col(symbol).is_not_null()).then(level).otherwise(None)
    prev_observation = observed.forward_fill().shift(1)
    denom = pl.coalesce([prev_observation, level.shift(1)])
    return level / denom - 1


def usd_panel(local: pl.DataFrame, symbol_to_ccy: dict[str, str], fx_levels: pl.DataFrame) -> pl.DataFrame:
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
    else:
        frame = local
        panel_start = None

    out = []
    for s in symbols:
        ccy = symbol_to_ccy[s]
        if ccy == "USD":
            out.append(pl.col(s))
        else:
            r_fx = _fx_return_over_observations(s, ccy)
            out.append(((1.0 + pl.col(s)) * (1.0 + r_fx) - 1.0).alias(s))
    frame = frame.with_columns(out)
    if panel_start is not None:
        # Drop the seed row now that it has served its only purpose (supplying
        # the level.shift(1) denominator for the panel's real first row).
        frame = frame.filter(pl.col("date") >= panel_start)
    return frame.select(["date"] + symbols)
