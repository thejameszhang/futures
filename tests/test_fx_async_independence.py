import ast
from pathlib import Path

FX = Path(__file__).resolve().parents[1] / "src" / "globalmacro" / "pipeline" / "fx.py"


def _main_block_statements() -> list[str]:
    tree = ast.parse(FX.read_text())
    for node in tree.body:
        if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'":
            return [ast.unparse(s) for s in node.body]
    raise AssertionError("no __main__ block found in fx.py")


def test_fx_async_is_written_before_compustat_is_touched():
    stmts = _main_block_statements()

    # "fx_async" is the variable name, not the bare substring "async" -- safe to match on
    # even though "sync" is a substring of "async": "fx_sync.write_csv(...)" and
    # "build_synthetic(fx_sync).write_csv(...)" both contain "fx_sync", never "fx_async".
    async_write_idxs = [i for i, s in enumerate(stmts) if "fx_async" in s and ".write_csv" in s]
    assert async_write_idxs, "no fx_async .write_csv statements found in __main__ block"
    # The LAST async-producing write, not just the first: fx_async.csv is one of two async
    # outputs (the other is synthetic_fx_returns_async.csv, written via
    # build_synthetic(fx_async).write_csv(...)). Checking only the first write here would
    # pass a reorder that moves fx_async.write_csv above save_compustat_fx_rates() while
    # leaving the synthetic async write below it -- still losing half the async output on a
    # Compustat failure, which is exactly what this test exists to prevent.
    last_async_write = max(async_write_idxs)

    compustat = next((i for i, s in enumerate(stmts) if "save_compustat_fx_rates" in s), None)
    assert compustat is not None, "no save_compustat_fx_rates() call found in __main__ block"

    assert last_async_write < compustat, (
        "save_compustat_fx_rates() must not run before every fx_async write (fx_async.csv "
        "and synthetic_fx_returns_async.csv) is complete -- a Compustat failure would "
        "otherwise destroy part of the already-computed async output too")
