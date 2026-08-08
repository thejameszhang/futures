import inspect

from globalmacro import build


def test_build_exposes_both_halves_with_the_expected_shapes():
    assert callable(build.build_async) and callable(build.build_sync)
    assert list(inspect.signature(build.build_async).parameters) == ["synthetic_returns"]
    assert list(inspect.signature(build.build_sync).parameters) == ["synthetic_returns_synced"]
