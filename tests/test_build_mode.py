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


def test_validate_mode_rejects_anything_but_the_two_valid_strings():
    """`main()` is directly callable with any string -- `main("Full")` or a typo'd
    `main("aysnc-only")` would otherwise silently take the async-only branch and
    report success on a truncated deliverable. `_validate_mode` is the runtime
    guard `main()` calls as its first statement.

    Tested directly, never via `main()` itself: this repo's tests must never
    invoke `main()` (it would build the real datasets against the real
    DATASETS_ROOT). Calling `_validate_mode` alone does no I/O.
    """
    from globalmacro.build import _validate_mode
    for valid in ("full", "async-only"):
        _validate_mode(valid)  # must not raise
    for invalid in ("Full", "FULL", "aysnc-only", "", "sync", "async"):
        with pytest.raises(ValueError):
            _validate_mode(invalid)


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


def _find_resolve_mode_target_name(if_block: ast.If) -> str:
    """The name bound by `<name> = resolve_mode(...)` in the __main__ block."""
    for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[])):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "resolve_mode"
        ):
            return node.targets[0].id
    raise AssertionError(
        f"{BUILD_PY}: no `<name> = resolve_mode(...)` assignment found inside the "
        '`if __name__ == "__main__":` block'
    )


def _find_folders_to_create_for_loop_lineno(if_block: ast.If) -> int:
    """Smallest lineno of any `for folder in folders_to_create:` loop in the
    __main__ block."""
    linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "folders_to_create"
        )
    ]
    if not linenos:
        raise AssertionError(
            f"{BUILD_PY}: no `for folder in folders_to_create:` loop found inside "
            'the `if __name__ == "__main__":` block'
        )
    return min(linenos)


def _find_main_function(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            return node
    raise AssertionError(f"{BUILD_PY}: no `def main(...):` function found")


def test_build_mode_is_logged_as_the_first_statement_in_main():
    """Deliberate, owner-approved -- the one exception to 'no economic logic
    changes': 'build mode: <mode>' must be main()'s first
    EFFECTIVE (output-producing) statement, so validation_report.txt -- written by
    the FileHandler attached in the __main__ block before main() is called --
    records its own provenance as its first line. Adds exactly one line; changes no
    number and no dataset.

    main()'s literal FIRST statement is now
    `_validate_mode(mode)`, not the log call -- the old order let `main("Bogus")`
    write "build mode: Bogus" and only THEN raise. _validate_mode produces no
    output of its own for a valid mode (proven directly, not assumed, in
    test_validate_mode_produces_no_output_for_a_valid_mode below), so the log call
    is still the first statement that actually WRITES anything -- this test checks
    both halves of that: body[0] is the validation guard, body[1] is the log call.

    Static AST check, no filesystem access and no call of build.main() itself --
    this repo's tests must never invoke main() (it would build the real datasets
    against the real DATASETS_ROOT; see
    test_validate_mode_rejects_anything_but_the_two_valid_strings above, which
    documents the same constraint for _validate_mode)."""
    source = BUILD_PY.read_text()
    tree = ast.parse(source, filename=str(BUILD_PY))
    main_fn = _find_main_function(tree)
    assert len(main_fn.body) >= 2, "main() must have at least a guard and a log statement"

    guard = main_fn.body[0]
    assert isinstance(guard, ast.Expr) and isinstance(guard.value, ast.Call), (
        f"expected main()'s first statement to be a call expression, got: "
        f"{ast.dump(guard)}"
    )
    guard_call = guard.value
    assert isinstance(guard_call.func, ast.Name) and guard_call.func.id == "_validate_mode", (
        f"expected main()'s first statement to be _validate_mode(mode) -- the guard "
        f"must run BEFORE anything is logged -- got: {ast.unparse(guard_call)}"
    )

    second = main_fn.body[1]
    assert isinstance(second, ast.Expr) and isinstance(second.value, ast.Call), (
        f"expected main()'s second statement to be a call expression, got: "
        f"{ast.dump(second)}"
    )
    call = second.value
    assert isinstance(call.func, ast.Attribute) and call.func.attr == "info", (
        f"expected main()'s second statement to be logger.info(...), got: "
        f"{ast.unparse(call)}"
    )
    unparsed = ast.unparse(call)
    assert "build mode:" in unparsed, (
        f"expected main()'s second statement to log 'build mode: <mode>', got: {unparsed}"
    )
    # The mode PARAMETER, not a hardcoded literal -- main(mode=...) 's own argument.
    mode_param = main_fn.args.args[0].arg if main_fn.args.args else None
    assert mode_param == "mode", f"main()'s first parameter must be named 'mode', got {mode_param!r}"
    assert mode_param in unparsed, (
        f"expected the logged line to reference main()'s `mode` parameter, got: {unparsed}"
    )
    assert guard_call.args and isinstance(guard_call.args[0], ast.Name), (
        f"expected _validate_mode to be called with main()'s `mode` argument, got: "
        f"{ast.unparse(guard_call)}"
    )
    assert guard_call.args[0].id == mode_param, (
        f"expected _validate_mode(mode) to pass main()'s own `mode` parameter through "
        f"unchanged, got: {ast.unparse(guard_call)}"
    )


def test_validate_mode_produces_no_output_for_a_valid_mode(caplog, capsys):
    """The reorder above (validate, then log) is only safe -- i.e.
    only preserves 'build mode: <mode>' as the first line written to
    validation_report.txt -- if _validate_mode itself writes NOTHING for a valid
    mode. Proven directly rather than assumed: zero log records at ANY level
    (caplog.at_level(logging.DEBUG) captures regardless of handler/level config,
    unlike relying on build.py's own logger configuration) and zero stdout/stderr."""
    import logging

    from globalmacro.build import _validate_mode
    with caplog.at_level(logging.DEBUG):
        _validate_mode("full")
        _validate_mode("async-only")
    assert caplog.records == []
    out, err = capsys.readouterr()
    assert out == ""
    assert err == ""


def test_folders_to_create_is_mode_dependent():
    """Static AST assertion, no filesystem access and no import of build.

    Nothing in the test suite otherwise checks that async-only mode skips
    creating datasets/tier{1,2}/sync -- a future edit could delete the
    mode-dependent filter and stay green while `globalmacro build --async-only`
    advertised a sync/ tree it never fills. This pins the shape of the fix rather
    than its exact filter expression: the resolved-mode variable bound by
    `<name> = resolve_mode(...)` must be referenced somewhere between the
    `folders_to_create = [...]` assignment and the
    `for folder in folders_to_create:` loop that consumes it.
    """
    source = BUILD_PY.read_text()
    tree = ast.parse(source, filename=str(BUILD_PY))
    if_block = _find_main_block(tree)
    mode_name = _find_resolve_mode_target_name(if_block)
    folders_lineno = _find_folders_to_create_assign_lineno(if_block)
    for_lineno = _find_folders_to_create_for_loop_lineno(if_block)
    assert folders_lineno < for_lineno, (
        f"folders_to_create assignment (line {folders_lineno}) must precede its "
        f"consuming for-loop (line {for_lineno})"
    )

    mode_ref_linenos = [
        node.lineno
        for node in ast.walk(ast.Module(body=if_block.body, type_ignores=[]))
        if (
            isinstance(node, ast.Name)
            and node.id == mode_name
            and isinstance(node.ctx, ast.Load)
            and folders_lineno <= node.lineno < for_lineno
        )
    ]
    assert mode_ref_linenos, (
        f"expected a reference to `{mode_name}` (the value bound by "
        f"`{mode_name} = resolve_mode(...)`) between the folders_to_create "
        f"assignment (line {folders_lineno}) and its for-loop (line {for_lineno}) "
        'in build.py\'s `if __name__ == "__main__":` block -- folder creation '
        "must be mode-dependent, or async-only mode would create sync/ "
        "directories it never fills"
    )


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
