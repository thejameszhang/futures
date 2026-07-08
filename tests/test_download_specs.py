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
