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


def test_build_rejects_both_flags_together():
    """--async-only and --full must be mutually exclusive. Without
    `p.add_mutually_exclusive_group()` in `_parse_args`, argparse would apply
    last-wins semantics -- `--async-only --full` would silently resolve to mode
    "full" instead of exiting 2."""
    from globalmacro.build import _parse_args
    with pytest.raises(SystemExit) as e:
        _parse_args(["--async-only", "--full"])
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
    """Smallest lineno of any `_parse_args(...)` call in the __main__ block."""
    linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_parse_args"
        )
    ]
    if not linenos:
        raise AssertionError(
            f"{BUILD_PY}: no `_parse_args(...)` call found inside the "
            '`if __name__ == "__main__":` block'
        )
    return min(linenos)


def _find_resolve_mode_call_lineno(if_block: ast.If) -> int:
    """Smallest lineno of any `resolve_mode(...)` call in the __main__ block."""
    linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_mode"
        )
    ]
    if not linenos:
        raise AssertionError(
            f"{BUILD_PY}: no `resolve_mode(...)` call found inside the "
            '`if __name__ == "__main__":` block'
        )
    return min(linenos)


def _find_filehandler_call_lineno(if_block: ast.If) -> int:
    """Smallest lineno of any `logging.FileHandler(...)` call in the __main__ block.

    This is the call that opens validation_report.txt with mode="w", truncating
    it. Located via ast the same way the other helpers in this file locate their
    targets: walk the block, match on the call's callee shape.
    """
    linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "FileHandler"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "logging"
        )
    ]
    if not linenos:
        raise AssertionError(
            f"{BUILD_PY}: no `logging.FileHandler(...)` call found inside the "
            '`if __name__ == "__main__":` block'
        )
    return min(linenos)


def _find_folders_to_create_assign_lineno(if_block: ast.If) -> int:
    """Smallest lineno of any `folders_to_create = [...]` assignment in the
    __main__ block."""
    linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "folders_to_create"
        )
    ]
    if not linenos:
        raise AssertionError(
            f"{BUILD_PY}: no `folders_to_create = ...` assignment found inside the "
            '`if __name__ == "__main__":` block'
        )
    return min(linenos)


def test_main_block_parses_args_before_creating_folders():
    """Static source-order guard, no filesystem access and no import of build.

    Checks three orderings inside build.py's `if __name__ == "__main__":` block:

    1. `_parse_args(...)` precedes `folders_to_create = [...]` -- the parsed mode
       must be available before the list a later task will make depend on it.
    2. `_parse_args(...)` precedes `logging.FileHandler(...)` -- an early exit
       during argument parsing (--help, an unknown flag) happens before the
       FileHandler call that opens validation_report.txt with mode="w", so it
       cannot truncate that report without regenerating it.
    3. `resolve_mode(...)` precedes `folders_to_create = [...]` -- same
       requirement as (1), checked directly on resolve_mode rather than only
       proxied through _parse_args.

    Nothing else in the test suite touches the __main__ block, so a future edit
    that re-orders these statements would otherwise stay green while
    `globalmacro build` breaks on some invocations. Must fail closed: if the
    __main__ block or any of the four statements this test locates is missing,
    this raises with a clear message rather than passing vacuously.
    """
    source = BUILD_PY.read_text()
    tree = ast.parse(source, filename=str(BUILD_PY))
    if_block = _find_main_block(tree)
    parse_args_lineno = _find_parse_args_call_lineno(if_block)
    resolve_mode_lineno = _find_resolve_mode_call_lineno(if_block)
    filehandler_lineno = _find_filehandler_call_lineno(if_block)
    folders_lineno = _find_folders_to_create_assign_lineno(if_block)

    assert parse_args_lineno < folders_lineno, (
        f"_parse_args() call (line {parse_args_lineno}) must precede the "
        f"folders_to_create assignment (line {folders_lineno}) in build.py's "
        '`if __name__ == "__main__":` block'
    )
    assert parse_args_lineno < filehandler_lineno, (
        f"_parse_args() call (line {parse_args_lineno}) must precede the "
        f"logging.FileHandler(...) call (line {filehandler_lineno}) in build.py's "
        '`if __name__ == "__main__":` block, or an early exit from argument '
        "parsing truncates validation_report.txt without regenerating it"
    )
    assert resolve_mode_lineno < folders_lineno, (
        f"resolve_mode() call (line {resolve_mode_lineno}) must precede the "
        f"folders_to_create assignment (line {folders_lineno}) in build.py's "
        '`if __name__ == "__main__":` block'
    )
