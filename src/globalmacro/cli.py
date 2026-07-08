"""globalmacro CLI: one discoverable entry point for the whole pipeline.

Pipeline stages keep their own `if __name__ == "__main__"` blocks; the CLI runs
each via `runpy` (identical to `python -m globalmacro.pipeline.<stage>`), so the
stage's module-level setup + argparse work unchanged and the heavily
module-global stage internals are never touched (byte-identical invariant safe).
`instrumentlists` and `validate` are plain functions, called directly. `runpy`
imports the target module on demand, so `globalmacro build` never imports `wrds`
(only `download` needs it).
"""
import runpy
import sys

# subcommand -> module executed as __main__ via runpy
_STAGE_MODULES = {
    "download": "globalmacro.pipeline.download",
    "equities": "globalmacro.pipeline.equities",
    "futures": "globalmacro.pipeline.futures",
    "fx": "globalmacro.pipeline.fx",
    "rates": "globalmacro.pipeline.rates",
    "tickhistory": "globalmacro.pipeline.tickhistory",
    "build": "globalmacro.build",
}
_FUNCTION_STAGES = ("instrumentlists", "validate", "run")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    choices = sorted([*_STAGE_MODULES, *_FUNCTION_STAGES])

    # Dispatch by hand (NOT argparse) so a stage's own `--help` is forwarded to it
    # rather than intercepted by a top-level parser's implicit -h/--help.
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: globalmacro <stage> [args...]\n\nstages:\n  " + "\n  ".join(choices))
        return
    stage, rest = argv[0], argv[1:]
    if stage not in choices:
        raise SystemExit(f"globalmacro: unknown stage {stage!r}; choose from {', '.join(choices)}")

    if stage == "instrumentlists":
        from globalmacro.utils.instrumentlists import generate_instrument_lists
        return generate_instrument_lists()
    if stage == "validate":
        from globalmacro.validation.run import main as validate_main
        return validate_main(rest)
    if stage == "run":
        import subprocess
        from globalmacro.utils.paths import PROJECT_ROOT
        script = PROJECT_ROOT / "slurm" / "run_all.sh"
        return subprocess.call(["bash", str(script), *rest])

    # Run the stage module as if invoked as a script: set argv so its argparse
    # (if any) sees the forwarded args, then execute its __main__ block.
    sys.argv = [f"globalmacro {stage}", *rest]
    runpy.run_module(_STAGE_MODULES[stage], run_name="__main__")


if __name__ == "__main__":
    raise SystemExit(main())
