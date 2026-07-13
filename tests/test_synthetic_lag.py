from datetime import date

import polars as pl

from globalmacro.build import (
    AMERICAS_CASH_INDICES,
    lag_one_session,
    load_synthetic_returns,
    load_rf,
    load_symbols,
)


def test_lag_uses_previous_own_observation_not_previous_row():
    # X does not trade on the 2nd or the 4th. Its lagged value on the 3rd must be
    # the 1st's value (its previous OWN observation), not the 2nd's null.
    df = pl.DataFrame({
        "date": [date(2020, 1, d) for d in (1, 2, 3, 4, 5)],
        "X": [0.01, None, 0.02, None, 0.03],
    })
    out = lag_one_session(df, ["X"])
    assert out["X"].to_list() == [None, None, 0.01, None, 0.02]


def test_lag_never_turns_a_null_into_zero():
    # THE cardinal rule: a missing observation must stay missing. A 0.0 here would be
    # a fabricated "no move" that no downstream consumer could detect.
    df = pl.DataFrame({
        "date": [date(2020, 1, d) for d in (1, 2, 3)],
        "X": [None, None, 0.05],
    })
    out = lag_one_session(df, ["X"])
    assert out["X"].to_list() == [None, None, None]
    assert 0.0 not in [v for v in out["X"].to_list() if v is not None]


def test_lag_preserves_null_mask_except_first_observation():
    df = pl.DataFrame({
        "date": [date(2020, 1, d) for d in (1, 2, 3, 4)],
        "X": [0.01, None, 0.02, 0.03],
    })
    out = lag_one_session(df, ["X"])
    before = [v is not None for v in df["X"].to_list()]
    after = [v is not None for v in out["X"].to_list()]
    # exactly one observation lost (the first); none invented
    assert sum(before) - sum(after) == 1
    for b, a in zip(before, after):
        assert not (a and not b), "lag invented an observation on a non-trading date"


def test_lag_leaves_other_columns_untouched():
    df = pl.DataFrame({
        "date": [date(2020, 1, d) for d in (1, 2, 3)],
        "X": [0.01, 0.02, 0.03],
        "Y": [0.10, 0.20, 0.30],
    })
    out = lag_one_session(df, ["X"])
    assert out["Y"].to_list() == [0.10, 0.20, 0.30]


def _synth_frames():
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    equities = [f for f in (t1f + t2f) if f.dsindexcode is not None]
    return load_synthetic_returns(load_rf(), equities)


def test_async_synthetic_is_not_lagged_for_americas():
    # The async panel is settlement-timed. Lagging it would BREAK it (same-day
    # correlation 0.96-0.98 vs ~-0.05 at lag-1).
    from globalmacro.utils.paths import EQUITIES_PATH
    spot = pl.read_csv(EQUITIES_PATH / "spot_equity_returns.csv", infer_schema_length=0)
    spot = spot.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    a, _ = _synth_frames()
    for sym in AMERICAS_CASH_INDICES:
        if sym not in spot.columns or sym not in a.columns:
            continue
        j = (spot.select("date", pl.col(sym).cast(pl.Float64, strict=False).alias("raw"))
                 .join(a.select("date", pl.col(sym).alias("got")), on="date", how="inner")
                 .filter(pl.col("raw").is_not_null()))
        bad = j.filter((pl.col("got") - pl.col("raw")).abs() > 1e-12).height
        assert bad == 0, f"{sym}: async synthetic was lagged ({bad} rows differ)"


def test_sync_synthetic_is_lagged_for_americas_only():
    a, s = _synth_frames()
    equity_cols = [c for c in a.columns if c in s.columns and c != "date"]
    for sym in equity_cols:
        j = (a.select("date", pl.col(sym).alias("x"))
              .join(s.select("date", pl.col(sym).alias("y")), on="date", how="inner"))
        differs = j.filter(
            (pl.col("x").is_not_null() | pl.col("y").is_not_null())
            & (pl.col("x").is_null() | pl.col("y").is_null()
               | ((pl.col("x") - pl.col("y")).abs() > 1e-12))
        ).height
        if sym in AMERICAS_CASH_INDICES:
            assert differs > 0, f"{sym}: sync synthetic should be lagged but matches async"
        elif sym in ("NOK", "SEK", "6N", "6A"):
            continue          # FX columns legitimately differ (Datastream vs Compustat)
        else:
            assert differs == 0, f"{sym}: sync synthetic must NOT be lagged ({differs} rows differ)"
