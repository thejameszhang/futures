import inspect

import pytest


def test_build_parses_modes():
    from globalmacro.build import _parse_args
    assert _parse_args([]).mode is None
    assert _parse_args(["--async-only"]).mode == "async-only"
    assert _parse_args(["--full"]).mode == "full"


def test_build_rejects_unknown_flag():
    """Today `globalmacro build --nonsense` silently succeeds. It must not."""
    from globalmacro.build import _parse_args
    with pytest.raises(SystemExit) as e:
        _parse_args(["--nonsense"])
    assert e.value.code == 2


def test_main_accepts_mode_parameter():
    from globalmacro.build import main
    assert "mode" in inspect.signature(main).parameters
