import ast
import inspect
from pathlib import Path

import pytest

BUILD_PY = Path(__file__).resolve().parent.parent / "src" / "globalmacro" / "build.py"


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


def test_build_rejects_abbreviated_flag():
    """allow_abbrev must be off: `--f` / `--fu` / `--as` / `--async` must NOT silently
    resolve to --full / --async-only. Forward-fragile otherwise -- a later flag sharing
    a prefix would turn a working invocation into an ambiguity error."""
    from globalmacro.build import _parse_args
    for abbreviation in ("--f", "--fu", "--as", "--async"):
        with pytest.raises(SystemExit) as e:
            _parse_args([abbreviation])
        assert e.value.code == 2


def test_main_accepts_mode_parameter():
    from globalmacro.build import main
    params = inspect.signature(main).parameters
    assert "mode" in params
    assert params["mode"].default == "full"


def _find_main_block(tree: ast.Module) -> ast.If:
    """The top-level `if __name__ == "__main__":` block. Fails closed (raises rather
    than returning None) so a caller can't accidentally treat "not found" as success."""
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
        ):
            operands = [test.left, *test.comparators]
            names = [o for o in operands if isinstance(o, ast.Name)]
            consts = [o for o in operands if isinstance(o, ast.Constant)]
            if len(names) == 1 and len(consts) == 1 and names[0].id == "__name__" and consts[0].value == "__main__":
                return node
    raise AssertionError(
        f'{BUILD_PY}: no top-level `if __name__ == "__main__":` block found'
    )


def _find_parse_args_call_lineno(if_block: ast.If) -> int:
    """lineno of the `_parse_args(...)` call anywhere in the __main__ block."""
    for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[])):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_parse_args"
        ):
            return node.lineno
    raise AssertionError(
        f"{BUILD_PY}: no `_parse_args(...)` call found inside the "
        '`if __name__ == "__main__":` block'
    )


def _find_folders_to_create_assign_lineno(if_block: ast.If) -> int:
    """lineno of the `folders_to_create = [...]` assignment in the __main__ block."""
    for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[])):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "folders_to_create"
        ):
            return node.lineno
    raise AssertionError(
        f"{BUILD_PY}: no `folders_to_create = ...` assignment found inside the "
        '`if __name__ == "__main__":` block'
    )


def test_main_block_parses_args_before_creating_folders():
    """Static source-order guard, no filesystem access and no import of build.

    _parse_args()/resolve_mode() must run before folders_to_create (and therefore
    before the FileHandler/os.makedirs setup that precedes it), or every early-exit
    path (--help, an unknown flag, --full without sync inputs) truncates
    validation/validation_report.txt without regenerating it. Nothing else in the
    test suite touches the __main__ block, so a future edit that re-orders these
    statements would otherwise stay green while `globalmacro build` breaks on every
    invocation. Must fail closed: if the __main__ block or either statement is
    missing, this raises with a clear message rather than passing vacuously.
    """
    source = BUILD_PY.read_text()
    tree = ast.parse(source, filename=str(BUILD_PY))
    if_block = _find_main_block(tree)
    parse_args_lineno = _find_parse_args_call_lineno(if_block)
    folders_lineno = _find_folders_to_create_assign_lineno(if_block)
    assert parse_args_lineno < folders_lineno, (
        f"_parse_args() call (line {parse_args_lineno}) must precede the "
        f"folders_to_create assignment (line {folders_lineno}) in build.py's "
        '`if __name__ == "__main__":` block'
    )
