import ast
from pathlib import Path

import pytest

FX = Path(__file__).resolve().parents[1] / "src" / "globalmacro" / "pipeline" / "fx.py"

# The two async OUTPUT FILENAMES, not the producing variable name. F8 (Reviewer B,
# proved by AST simulation): the old guard anchored on the variable name `fx_async`,
# so binding the synthetic result to an intermediate -- `synth_async =
# build_synthetic(fx_async)` above the Compustat call, then `synth_async.write_csv(...)`
# below it -- left only ONE statement matching "fx_async" AND ".write_csv", so the old
# `max(idxs) < compustat` held and the test passed while synthetic_fx_returns_async.csv
# was actually written AFTER Compustat. Anchoring on filenames instead survives that
# rename because the write target, not the local variable name, is what actually
# determines the file on disk.
ASYNC_OUTPUT_FILENAMES = ("fx_async.csv", "synthetic_fx_returns_async.csv")


def _main_block_statements_from_source(source: str) -> list[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            return [ast.unparse(s) for s in node.body]
    raise AssertionError("no __main__ block found")


def _main_block_statements() -> list[str]:
    return _main_block_statements_from_source(FX.read_text())


def _assert_async_written_before_compustat(stmts: list[str]) -> None:
    """The guard itself, factored out of the test function so it can be run against
    both the real fx.py __main__ block and a literal mutant source string (see
    test_guard_fires_on_the_intermediate_variable_mutant below) -- a guard that
    cannot fail is worse than no guard, and this branch has already shipped one."""
    # "fx_async.csv" / "synthetic_fx_returns_async.csv" vs "fx_sync.csv" /
    # "synthetic_fx_returns_sync.csv" differ by the same "a" that has caused eight
    # defects on this branch -- match on the exact filenames, not a bare "async" or
    # "fx_a" substring.
    async_write_idxs = [
        i for i, s in enumerate(stmts)
        if ".write_csv" in s and any(name in s for name in ASYNC_OUTPUT_FILENAMES)
    ]
    assert async_write_idxs, (
        f"no .write_csv statement found for any of {ASYNC_OUTPUT_FILENAMES}")
    # Both async outputs must actually appear, not just one of the two -- fx_async.csv
    # is one of two async outputs (the other is synthetic_fx_returns_async.csv, written
    # via build_synthetic(fx_async).write_csv(...)). Checking only one filename here
    # would pass a reorder that moves the OTHER one below the Compustat call.
    for name in ASYNC_OUTPUT_FILENAMES:
        assert any(name in stmts[i] for i in async_write_idxs), (
            f"no .write_csv statement found for {name}")
    last_async_write = max(async_write_idxs)

    # Robust Compustat-boundary detection: not just the literal call name
    # `save_compustat_fx_rates`, so a future rename of that function (that still
    # touches Compustat) doesn't silently blind this guard the same way the old
    # variable-name anchor did.
    compustat = next(
        (i for i, s in enumerate(stmts) if "compustat" in s.lower() or "COMPUSTAT_PATH" in s),
        None,
    )
    assert compustat is not None, "no Compustat-related statement found in __main__ block"

    assert last_async_write < compustat, (
        "a Compustat-touching statement must not run before every async output "
        f"({', '.join(ASYNC_OUTPUT_FILENAMES)}) .write_csv is complete -- a Compustat "
        "failure would otherwise destroy part of the already-computed async output too"
    )


def test_fx_async_is_written_before_compustat_is_touched():
    _assert_async_written_before_compustat(_main_block_statements())


def test_guard_fires_on_the_intermediate_variable_mutant():
    """F8 (Reviewer B, proved by AST simulation; the fix was written at Task 4 and
    never applied). Prove the FIXED guard actually rejects the exact regression it
    exists to prevent, rather than merely describing it: build the intermediate-
    variable mutant as a literal source string, run the guard's statement
    extraction over it, and assert the guard FAILS (raises AssertionError).

    The mutant binds the synthetic async result to an intermediate variable
    (`synth_async`) above the Compustat call, then writes it out via that
    intermediate AFTER Compustat -- under the OLD guard (anchored on the variable
    name `fx_async`), `synth_async.write_csv(...)` does not contain the substring
    "fx_async" (it is a different name), so only the FIRST async write was seen and
    the old guard passed while synthetic_fx_returns_async.csv was actually written
    after Compustat. The NEW guard (anchored on the output filename
    "synthetic_fx_returns_async.csv") must catch this regardless of which variable
    the file happens to be written from.
    """
    mutant_source = '''
if __name__ == "__main__":
    fx_async = build_datastream_direct_panel()
    fx_async.write_csv(FX_PATH / "fx_async.csv")
    synth_async = build_synthetic(fx_async)

    fx_sync = save_compustat_fx_rates()
    fx_sync.write_csv(FX_PATH / "fx_sync.csv")
    build_synthetic(fx_sync).write_csv(FX_PATH / "synthetic_fx_returns_sync.csv")

    synth_async.write_csv(FX_PATH / "synthetic_fx_returns_async.csv")
'''
    stmts = _main_block_statements_from_source(mutant_source)
    # Sanity check the mutant is shaped as intended: two "fx_async"-containing
    # statements exist (the assignment and the first write), but only ONE contains
    # ".write_csv" -- so the OLD (variable-name-anchored) guard's
    # `[i for i, s in enumerate(stmts) if "fx_async" in s and ".write_csv" in s]`
    # would find exactly one match and never see the late write.
    old_guard_matches = [i for i, s in enumerate(stmts) if "fx_async" in s and ".write_csv" in s]
    assert len(old_guard_matches) == 1, (
        "mutant fixture is not shaped as intended -- the old variable-name-anchored "
        f"guard should see exactly one match, saw {len(old_guard_matches)}"
    )
    with pytest.raises(AssertionError):
        _assert_async_written_before_compustat(stmts)


def test_guard_requires_both_async_filenames_present():
    """A source that only writes one of the two async outputs must fail loudly
    (missing filename), not silently pass on whichever one it found."""
    partial_source = '''
if __name__ == "__main__":
    fx_async = build_datastream_direct_panel()
    fx_async.write_csv(FX_PATH / "fx_async.csv")

    fx_sync = save_compustat_fx_rates()
    fx_sync.write_csv(FX_PATH / "fx_sync.csv")
    build_synthetic(fx_sync).write_csv(FX_PATH / "synthetic_fx_returns_sync.csv")
'''
    stmts = _main_block_statements_from_source(partial_source)
    with pytest.raises(AssertionError, match="synthetic_fx_returns_async.csv"):
        _assert_async_written_before_compustat(stmts)
