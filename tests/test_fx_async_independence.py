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
    write_async = next(i for i, s in enumerate(stmts) if "fx_async.write_csv" in s)
    compustat = next(i for i, s in enumerate(stmts) if "save_compustat_fx_rates" in s)
    assert write_async < compustat, (
        "save_compustat_fx_rates() must not run before fx_async.csv is written -- a "
        "Compustat failure would otherwise destroy the async panel too")
