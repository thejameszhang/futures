import inspect


def test_load_vix_is_removed():
    import globalmacro.build as build
    assert not hasattr(build, "load_vix"), "load_vix must be removed"


def test_build_synced_dataset_has_no_vix_open_param():
    from globalmacro.build import build_synced_dataset
    params = list(inspect.signature(build_synced_dataset).parameters)
    assert "vix_open" not in params, f"vix_open must stay removed, got {params}"


def test_splicing_map_has_no_vx_entry():
    from globalmacro.utils.splice import SPLICING_MAP
    assert "VX" not in SPLICING_MAP, "dead VX->vix_ret_rf_open splice must be removed"
