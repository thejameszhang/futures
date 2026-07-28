import polars as pl
import pytest

from globalmacro.build import (
    build_currency_map,
    load_rf,
    load_symbols,
    load_synthetic_returns,
    read_csv,
)
from globalmacro.pipeline.fx import SYMBOL_TO_CURCDD_MAPPING
from globalmacro.pipeline.to_usd import usd_panel
from globalmacro.utils.paths import DATASETS_ROOT, FX_PATH
from globalmacro.utils.sync_fx import build_sync_fx_panel
from globalmacro.validation.synthetic import pre_splice_panel

TOLERANCE = 1e-10

# The FX columns load_synthetic_returns joins onto the equity synthetics.
_FX_SYNTHETIC_COLS = ("NOK", "SEK", "6N", "6A")


def _panel(rel: str) -> pl.DataFrame:
    df = pl.read_csv(str(DATASETS_ROOT / rel), infer_schema_length=0)
    df = df.with_columns(pl.col("date").str.strptime(pl.Date, strict=False))
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in df.columns if c != "date"]
    )


def _fx_panel(dataset: str, fx_file: str) -> pl.DataFrame:
    """The FX level panel that actually produced the `dataset` `_usd` panel.

    Async uses the raw on-disk Datastream file. SYNC applies the in-memory G10
    FX-futures blend (`build_sync_fx_panel`) that `build.main` layers on `fx_sync`
    before `save_usd_datasets` -- the blended panel is never written to disk, so it is
    rebuilt here from the same inputs the pipeline uses (`pre_splice_panel("sync")` ==
    `build.main`'s captured `pre_splice_synced`), giving an exact reconstruction.
    """
    fx = read_csv(FX_PATH / fx_file)
    if dataset == "sync" and fx_file == "fx_sync.csv":
        fx = build_sync_fx_panel(fx, pre_splice_panel("sync"), SYMBOL_TO_CURCDD_MAPPING)
    return fx


def _ccy_map():
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    return build_currency_map(t1f + t2f)


def _max_deviation(published: pl.DataFrame, rebuilt: pl.DataFrame) -> float:
    worst = 0.0
    for col in (c for c in published.columns if c != "date"):
        if col not in rebuilt.columns:
            continue
        j = (
            published.select("date", pl.col(col).alias("p"))
            .join(rebuilt.select("date", pl.col(col).alias("r")), on="date", how="inner")
            .filter(pl.col("p").is_not_null() & pl.col("r").is_not_null())
        )
        if j.height == 0:
            continue
        worst = max(worst, j.select((pl.col("p") - pl.col("r")).abs().max()).item())
    return worst


@pytest.mark.parametrize(
    "tier,dataset,fx_file",
    [(1, "async", "fx_async.csv"), (1, "sync", "fx_sync.csv"),
     (2, "async", "fx_async.csv"), (2, "sync", "fx_sync.csv")],
)
def test_usd_panel_reconstructs_exactly(tier, dataset, fx_file):
    local = _panel(f"tier{tier}/{dataset}/{dataset}_daily.csv")
    published = _panel(f"tier{tier}/{dataset}/{dataset}_daily_usd.csv")
    rebuilt = usd_panel(local, _ccy_map(), _fx_panel(dataset, fx_file))
    assert _max_deviation(published, rebuilt) <= TOLERANCE


@pytest.mark.parametrize(
    "tier,dataset,right_fx,wrong_fx",
    [(1, "async", "fx_async.csv", "fx_sync.csv"),
     (1, "sync", "fx_sync.csv", "fx_async.csv"),
     (2, "async", "fx_async.csv", "fx_sync.csv"),
     (2, "sync", "fx_sync.csv", "fx_async.csv")],
)
def test_usd_panel_uses_the_right_fx_source(tier, dataset, right_fx, wrong_fx):
    # The wiring test. An async panel built with sync FX would still correlate ~0.98 with
    # the truth, so only an EXACT reconstruction can tell the two sources apart.
    local = _panel(f"tier{tier}/{dataset}/{dataset}_daily.csv")
    published = _panel(f"tier{tier}/{dataset}/{dataset}_daily_usd.csv")
    ccy = _ccy_map()
    assert _max_deviation(published, usd_panel(local, ccy, _fx_panel(dataset, right_fx))) <= TOLERANCE
    wrong = _max_deviation(published, usd_panel(local, ccy, read_csv(FX_PATH / wrong_fx)))
    assert wrong > 1e-4, (
        f"tier{tier} {dataset}: the panel reconciles against BOTH FX sources "
        f"(max dev {wrong:.2e}) — the source-attribution test cannot detect a swap"
    )


def _synthetic_fx_agreement(frame: pl.DataFrame, source_file: str) -> tuple[float, int]:
    """(max |frame - file|, count of file observations the frame does not carry).

    The two synthetic FX files do not cover the same dates (Datastream starts in 1982,
    Compustat later; their last quotes differ too), so the missing-value count is only
    meaningful against the file the frame is supposed to have come from.
    """
    src = read_csv(FX_PATH / source_file)
    worst, missing = 0.0, 0
    for col in _FX_SYNTHETIC_COLS:
        j = (
            frame.select("date", pl.col(col).cast(pl.Float64).alias("got"))
            .join(
                src.select("date", pl.col(col).cast(pl.Float64).alias("want")),
                on="date", how="inner",
            )
            .filter(pl.col("want").is_not_null() & pl.col("want").is_finite())
        )
        assert j.height > 500, f"{col}: only {j.height} shared dates with {source_file}"
        missing += j.filter(pl.col("got").is_null()).height
        both = j.filter(pl.col("got").is_not_null())
        worst = max(worst, both.select((pl.col("got") - pl.col("want")).abs().max()).item())
    return worst, missing


@pytest.mark.parametrize(
    "which,right_file,wrong_file",
    [(0, "synthetic_fx_returns_async.csv", "synthetic_fx_returns_sync.csv"),
     (1, "synthetic_fx_returns_sync.csv", "synthetic_fx_returns_async.csv")],
)
def test_load_synthetic_returns_splices_the_right_fx_file(which, right_file, wrong_file):
    # The SPLICE-level wiring test. Exercise (a)'s source_diagonal reads the two synthetic
    # files by name, so swapping them INSIDE load_synthetic_returns leaves the diagonal
    # byte-identical and still reporting 8/8. A correlation test cannot see the swap either:
    # the two sources are the same rate at a different snapshot, so they correlate ~0.99
    # with each other. Only exact equality distinguishes them.
    t1f, _ = load_symbols(1)
    t2f, _ = load_symbols(2)
    equities = [f for f in (t1f + t2f) if f.dsindexcode is not None]
    frame = load_synthetic_returns(load_rf(), equities)[which]
    label = "async" if which == 0 else "sync"

    dev, missing = _synthetic_fx_agreement(frame, right_file)
    assert dev <= TOLERANCE, f"{label} synthetic FX does not match {right_file} (max dev {dev:.2e})"
    assert missing == 0, f"{label} synthetic FX drops {missing} observations {right_file} carries"

    wrong, _ = _synthetic_fx_agreement(frame, wrong_file)
    assert wrong > 1e-6, (
        f"the {label} synthetic reconciles against BOTH FX sources (max dev {wrong:.2e}) "
        f"-- a swap inside load_synthetic_returns would be undetectable"
    )


def test_usd_denominated_symbols_pass_through_untouched():
    local = _panel("tier1/async/async_daily.csv")
    published = _panel("tier1/async/async_daily_usd.csv")
    ccy = _ccy_map()
    usd_symbols = [c for c in local.columns if c != "date" and ccy.get(c) == "USD"]
    assert usd_symbols, "expected some USD-denominated symbols"
    for col in usd_symbols:
        j = (
            local.select("date", pl.col(col).alias("l"))
            .join(published.select("date", pl.col(col).alias("u")), on="date", how="inner")
            .filter(pl.col("l").is_not_null() & pl.col("u").is_not_null())
        )
        if j.height == 0:
            continue
        worst = j.select((pl.col("l") - pl.col("u")).abs().max()).item()
        assert worst == 0.0, f"{col} is USD-denominated but was converted (max diff {worst})"
