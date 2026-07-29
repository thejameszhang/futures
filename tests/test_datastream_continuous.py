import os

import polars as pl
import pytest

_WRDS = pytest.mark.skipif(
    not os.environ.get("GM_WRDS_TESTS"),
    reason="needs a live WRDS connection; set GM_WRDS_TESTS=1 (and a working ~/.pgpass)",
)


@_WRDS
def test_pulled_slice_reproduces_on_disk_within_window(tmp_path):
    """The freshly pulled slice matches the on-disk file's Settlement on shared
    (ClsCode, CalcSeriesName, Date_) keys within the file's date window, within tolerance,
    over the tier1 clscode universe datastream_comparison.py actually validates. Cross-vintage
    revisions beyond tolerance are reported, not hard-failed (spec I-2)."""
    from globalmacro.pipeline.download import pull_datastream_continuous
    from globalmacro.utils.config import load_config
    from globalmacro.utils.paths import FUTURES_PATH, PROJECT_ROOT
    on_disk_path = FUTURES_PATH / "datastream_continuous_series.csv"
    if not on_disk_path.exists():
        pytest.skip("no on-disk datastream_continuous_series.csv to compare against")

    # Read the fresh pull with PURE inference (no schema_overrides): this is the real M-2
    # check -- if ::integer were missing, ClsCode would be "102.00" and infer Float64 (and
    # ClsCode.is_in(<ints>) in the consumer would silently match nothing).
    fresh = pl.read_csv(pull_datastream_continuous(tmp_path))
    assert fresh["ClsCode"].dtype == pl.Int64, "ClsCode inferred non-Int64 -> ::integer cast missing (M-2)"

    # The clscodes datastream_comparison.py filters to (tier1.yaml), NOT arbitrary series.
    universe = [f.clscode for f in load_config(PROJECT_ROOT / "tier1.yaml") if f.clscode is not None]
    key = ["ClsCode", "CalcSeriesName", "Date_"]
    disk = (
        pl.scan_csv(on_disk_path)
        .filter((pl.col("RollMethodCode") == 0) & (pl.col("PositionFwdCode") == 0)
                & pl.col("Settlement").is_not_null() & pl.col("ClsCode").is_in(universe))
        .select([*key, "Settlement"]).collect()
    )
    window_hi = disk["Date_"].max()  # ISO date strings sort lexicographically
    f = fresh.filter(pl.col("ClsCode").is_in(universe) & (pl.col("Date_") <= window_hi)).select([*key, "Settlement"])
    j = disk.rename({"Settlement": "o"}).join(f.rename({"Settlement": "n"}), on=key, how="inner")
    assert j.height > 0, "no shared keys — join/column names drifted, or universe not in the file"
    # Coverage is a DIAGNOSTIC, not a hard gate: whole tier1 series legitimately drop out
    # across WRDS vintages -- WRDS periodically reassigns rollmethodcode, so a series the
    # on-disk file holds at RollMethodCode=0 can be RollMethodCode>=1 in a later pull and
    # thus fall outside this (consumer-mirroring) filter. That is upstream and benign for
    # this QA cross-check (empirically ~73/102 tier1 clscodes overlap at a given vintage).
    # The hard gates are non-empty (above) + settlement-within-tolerance (below).
    n_shared = len(set(fresh["ClsCode"].unique()) & set(disk["ClsCode"].unique()))
    print(f"coverage: {j.height}/{disk.height} on-disk rows matched ({j.height / disk.height:.1%}); "
          f"tier1 clscode overlap {n_shared}/{disk['ClsCode'].n_unique()} (rest = WRDS rollmethodcode drift)")
    tol = 1e-4 * pl.col("o").abs().clip(lower_bound=0.01)  # relative (1bp), floored at 0.01 abs
    within = j.filter((pl.col("n") - pl.col("o")).abs() <= tol).height
    frac = within / j.height
    print(f"datastream_continuous reproduction: {frac:.4%} within tol on {j.height}/{disk.height} keys / {len(universe)} tier1 clscodes")
    assert frac >= 0.98, f"settlement drift beyond tolerance on {(1-frac):.2%} of keys — inspect real regression vs benign revision"
