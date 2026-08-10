from datetime import date

import polars as pl
import pytest

from globalmacro.build import (
    build_currency_map,
    load_symbols,
    load_symbols_to_save,
    load_synthetic_returns,
    save_usd_datasets,
)


def test_load_synthetic_returns_uses_source_specific_fx():
    from globalmacro.build import load_rf
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    equities = [f for f in (t1f + t2f) if f.dsindexcode is not None]
    a, s = load_synthetic_returns(load_rf(), equities)
    # async and sync must NOT be identical clones now (different FX source)
    for col in ["NOK", "SEK", "6N", "6A"]:
        if col in a.columns and col in s.columns:
            m = a.select(["date", pl.col(col).alias("x")]).join(
                s.select(["date", pl.col(col).alias("y")]), on="date", how="inner").drop_nulls()
            # same rate, different snapshot -> highly but not perfectly correlated
            if m.height > 500:
                c = m.select(pl.corr("x", "y")).item()
                assert 0.5 < c < 0.9999, f"{col}: async/sync synthetic corr={c} (expected same-rate-diff-time)"
                return
    raise AssertionError("no comparable currency column found")


def test_currency_map_covers_every_published_symbol():
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    ccy = build_currency_map(t1f + t2f)
    published = set(load_symbols_to_save(t1f)) | set(load_symbols_to_save(t2f))
    assert sorted(published - set(ccy)) == []
    assert ccy["6E"] == "USD" and ccy["XAE"] == "USD" and ccy["FDAX"] == "EUR"


def test_save_usd_datasets_uses_async_vs_sync_fx(tmp_path):
    d = [date(2020, 1, 1), date(2020, 1, 2)]
    def panel():
        return pl.DataFrame({"date": d, "ES": [None, 0.01], "FGBL": [None, 0.01]})
    fx_async = pl.DataFrame({"date": d, "EUR": [1.10, 1.20]})   # +9.1%
    fx_sync = pl.DataFrame({"date": d, "EUR": [1.10, 1.15]})    # +4.5%
    ccy = {"ES": "USD", "FGBL": "EUR"}
    save_usd_datasets(panel(), panel(), panel(), panel(), ccy, fx_async, fx_sync, out_root=tmp_path)
    for rel in ["tier1/async/async_daily_usd.csv", "tier1/sync/sync_daily_usd.csv",
                "tier1/async/async_monthly_usd.csv", "tier2/async/async_daily_usd.csv",
                "tier2/sync/sync_daily_usd.csv", "tier2/async/async_monthly_usd.csv"]:
        assert (tmp_path / rel).is_file(), rel
    a = pl.read_csv(tmp_path / "tier1/async/async_daily_usd.csv")["FGBL"].to_list()[1]
    s = pl.read_csv(tmp_path / "tier1/sync/sync_daily_usd.csv")["FGBL"].to_list()[1]
    assert a == pytest.approx((1 + 0.01) * (1.20 / 1.10) - 1)   # async used fx_async
    assert s == pytest.approx((1 + 0.01) * (1.15 / 1.10) - 1)   # sync used fx_sync


# ---------------------------------------------------------------------------
# load_traded_panel / the traded mask's route through save_usd_datasets
# ---------------------------------------------------------------------------


def _write_settlement_parquet(root, ct, rows):
    """rows: (symbol, date, [settlement_1..5]) -- the columns load_traded_panel reads."""
    frame = pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            **{f"settlement_{slot}": [r[2][slot - 1] for r in rows] for slot in range(1, 6)},
        },
        schema_overrides={f"settlement_{slot}": pl.Float64 for slot in range(1, 6)},
    )
    frame.write_parquet(root / f"datastream_futures_settlement_{ct}.parquet")


def test_load_traded_panel_reads_every_settlement_slot_and_both_cycles(tmp_path, monkeypatch):
    """Three load-bearing choices, one fixture each:

    d2 -- slot 1 is null while slot 3 prices. The finalised ret_1 is priced off ret_temp_2 as
    well as ret_temp_1, so a front-slot-only mask would call this date untraded and hand back
    exactly the widened FX interval the mask exists to prevent. Kills
    `any_horizontal(settlement_1..5)` -> `settlement_1`.

    d3 -- present in CT only. load_async_dataset coalesces CT over CS, so a panel value can
    come from either parquet and the mask must union both. Kills dropping either cycle.

    d4 -- every slot null. The class did not price: this is the "genuinely absent" state that
    must stay False, or the mask would pin every FX interval to one day. Kills
    `is_not_null()` -> `is_not_null().not_()` and any blanket-True mask.

    d5 -- the SAME date in both cycles, priced in CT and blank in CS. The two cycles must
    combine by ANY, not by ALL: the panel value could have come from either parquet, so one
    cycle pricing the class is enough to make the date an observation. d3 cannot catch this
    (it exists in one cycle only, where any and all agree). Kills `any()` -> `all()`.
    """
    import globalmacro.build as build_module

    d1, d2, d3, d4, d5 = (date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3),
                          date(2020, 1, 6), date(2020, 1, 7))
    monkeypatch.setattr(build_module, "FUTURES_PATH", tmp_path)
    _write_settlement_parquet(tmp_path, "CS", [
        ("FGBL", d1, [100.0, 101.0, None, None, None]),
        ("FGBL", d2, [None, None, 102.0, None, None]),
        ("FGBL", d4, [None, None, None, None, None]),
        ("FGBL", d5, [None, None, None, None, None]),
        ("ZZZZ", d1, [7.0, None, None, None, None]),
    ])
    _write_settlement_parquet(tmp_path, "CT", [
        ("FGBL", d3, [103.0, None, None, None, None]),
        ("FGBL", d5, [104.0, None, None, None, None]),
    ])

    panel = build_module.load_traded_panel(["FGBL"])
    assert "ZZZZ" not in panel.columns, "the mask must not carry symbols nobody asked for"
    got = dict(zip(panel["date"].to_list(), panel["FGBL"].to_list(), strict=True))
    assert got[d1] is True
    assert got[d2] is True, "a price in slot 3 is still a price"
    assert got[d3] is True, "the CT cycle must be unioned in"
    assert got[d4] is False, "no slot priced -- the class did not trade"
    assert got[d5] is True, "priced in either cycle is traded -- union by ANY, not ALL"


def test_load_traded_panel_survives_a_missing_parquet(tmp_path, monkeypatch):
    """An async-only or partially built tree must degrade to the old inference, not crash."""
    import globalmacro.build as build_module

    monkeypatch.setattr(build_module, "FUTURES_PATH", tmp_path)
    assert build_module.load_traded_panel(["FGBL"]).columns == ["date"]


def test_save_usd_datasets_gives_the_traded_mask_to_async_but_not_sync(tmp_path, monkeypatch):
    """The mask is built from Datastream settlement dates, which are the async panel's own
    observation dates. The sync panels are sampled from LSEG tick data at 09:31 ET, so those
    dates are not their observations and handing them the mask would be wrong, not merely
    useless. Kills passing `traded_async` to the sync call sites."""
    import globalmacro.build as build_module
    from globalmacro.pipeline import to_usd

    seen = []
    real = to_usd.usd_panel

    def spy(local, ccy, fx, traded=None):
        seen.append(traded is not None)
        return real(local, ccy, fx, traded)

    d = [date(2020, 1, 1), date(2020, 1, 2)]
    panel = pl.DataFrame({"date": d, "FGBL": [None, 0.01]})
    fx = pl.DataFrame({"date": d, "EUR": [1.10, 1.20]})
    mask = pl.DataFrame({"date": d, "FGBL": [True, True]})
    monkeypatch.setattr(build_module, "usd_panel", spy)
    build_module.save_usd_datasets(
        panel, panel, panel, panel, {"FGBL": "EUR"}, fx, fx,
        out_root=tmp_path, traded_async=mask)
    # call order in save_usd_datasets: tier1 async, tier1 sync, tier2 async, tier2 sync
    assert seen == [True, False, True, False]


def test_build_main_wires_the_traded_mask_into_save_usd_datasets():
    """`build.main` is the only place the mask is loaded and handed on, and it cannot be
    called from a test: it rebuilds every panel from the full Datastream tree. So assert the
    wiring structurally rather than by grep -- find main's `save_usd_datasets` call, take its
    `traded_async=` keyword, and prove the name it passes was bound to a `load_traded_panel`
    call in the same function.

    Without this, `traded_async = None` in main is a silent no-op: every behavioural test in
    the repo still passes while the shipped panels keep the defect.
    """
    import ast
    import inspect

    import globalmacro.build as build_module

    tree = ast.parse(inspect.getsource(build_module))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    save = next(c for c in ast.walk(main)
                if isinstance(c, ast.Call) and getattr(c.func, "id", None) == "save_usd_datasets")
    kw = next((k for k in save.keywords if k.arg == "traded_async"), None)
    assert kw is not None, "main does not pass traded_async to save_usd_datasets"
    assert isinstance(kw.value, ast.Name), "traded_async must be passed a bound name"
    bound = [n for n in ast.walk(main)
             if isinstance(n, ast.Assign)
             and any(getattr(t, "id", None) == kw.value.id for t in n.targets)]
    assert bound, f"{kw.value.id} is never assigned in main"
    assert any(isinstance(a.value, ast.Call)
               and getattr(a.value.func, "id", None) == "load_traded_panel"
               for a in bound), f"{kw.value.id} is not bound to load_traded_panel(...)"
