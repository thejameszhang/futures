from globalmacro.pipeline.download import PULL_SPECS


def test_pull_specs_are_consumed_only():
    assert PULL_SPECS["equities"] == ("tr_ds_equities", ["ds2indexdata"])
    assert PULL_SPECS["fx"] == ("tr_ds_equities", ["ds2fxcode", "ds2fxrate"])
    assert PULL_SPECS["futures"] == (
        "tr_ds_fut",
        ["dsfutclass", "dsfutcontr", "dsfuttrdcycle", "dsfutcontrinfo", "dsfutcontrval"],
    )
    assert PULL_SPECS["economics"] == ("tr_ds_econ", ["ecodata"])
    assert PULL_SPECS["comp"] == ("comp", ["exrt_dly"])
    # The big unconsumed fx tables must be gone.
    assert "ds2mktval" not in PULL_SPECS["fx"][1]
    assert "ds2equityindex" not in PULL_SPECS["equities"][1]


def test_datastream_continuous_pull_is_a_filtered_6col_join():
    from globalmacro.pipeline.download import DATASTREAM_CONTINUOUS_SQL as sql
    # the two joined tables + key
    assert "tr_ds_fut.dsfutcalcserval" in sql
    assert "tr_ds_fut.dsfutcalcserinfo" in sql
    assert "v.calcseriescode = i.calcseriescode" in sql
    # the consumed slice filter
    assert "rollmethodcode = 0" in sql and "positionfwdcode = 0" in sql
    assert "settlement is not null" in sql.lower()
    # ClsCode cast to integer (bare int -> scan_csv keeps Int, not Float64)
    assert 'i.clscode::integer AS "ClsCode"' in sql
    # exactly the 6 columns datastream_comparison.py reads, exact capitalized names
    for col in ("CalcSeriesName", "ClsCode", "Date_", "Settlement", "RollMethodCode", "PositionFwdCode"):
        assert f'"{col}"' in sql
