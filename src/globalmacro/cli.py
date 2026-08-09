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
    "download-public": "globalmacro.pipeline.download_public",
    "equities": "globalmacro.pipeline.equities",
    "futures": "globalmacro.pipeline.futures",
    "fx": "globalmacro.pipeline.fx",
    "rates": "globalmacro.pipeline.rates",
    "tickhistory": "globalmacro.pipeline.tickhistory",
    "build": "globalmacro.build",
}
_FUNCTION_STAGES = ("instrumentlists", "validate", "run", "connect")


def _capability_report(shard_cap, creds: bool, checked: bool) -> str:
    """What can this machine build, and what should the researcher run next?

    `creds`/`checked` are booleans, not credential values -- this function never
    touches an actual username or password, only presence/validity flags computed
    by its caller.
    """
    from globalmacro.utils.capabilities import SHARD_STEMS
    if not creds:
        lseg = "no credentials found"
    elif not checked:
        lseg = "credentials present (not checked -- run `connect --check-lseg` to verify)"
    else:
        lseg = "credentials valid"

    n = len(SHARD_STEMS) * 2
    tick = f"ready ({n}/{n} shard sets)" if shard_cap.ready else f"not ready (0/{n} shard sets)"

    lines = [f"LSEG TickHistory:  {lseg}", f"Tick data:         {tick}"]
    if shard_cap.message:
        lines.append(f"                   {shard_cap.message}")
    if shard_cap.ready:
        lines.append("-> This machine can build the SYNC and ASYNC datasets.")
    else:
        lines.append("-> This machine can build the ASYNC datasets.")
        lines.append("   Sync datasets need LSEG tick data on disk.")
    lines.append("   Next:  globalmacro run --with-download")
    return "\n".join(lines)


def _lseg_identity_line(present: bool) -> str | None:
    """Deviation from the Task 8 brief (documented in task-8-report.md): the brief's
    `_capability_report` never surfaces `credential_username()`, but Task 7's review
    (watch-item W2) flags that function as something Task 8 should deliberately decide
    whether to print. Printed WHOLE, not masked -- consistent with the existing `connect`
    output for WRDS (`Connected as: {creds.username}`, a few lines above this function's
    call site) and with wrds_credentials.py's on-disk state file, which already persists
    a WRDS username unmasked. This function never reads DSS_PASSWORD, directly or
    indirectly: `credential_username()` only resolves DSS_USERNAME.
    """
    if not present:
        return None
    from globalmacro.tickhistory_credentials import credential_username
    username = credential_username()
    return f"LSEG username:     {username}" if username else None


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
    if stage == "connect":
        from globalmacro.wrds_credentials import get_wrds_credentials, reset_credentials
        reset = "--reset" in rest or "-r" in rest
        if reset:
            reset_credentials(full_reset=True)
            print("Credentials reset.")
            return 0
        try:
            creds = get_wrds_credentials()
            import os

            import wrds
            if creds.password is not None:
                os.environ["PGPASSWORD"] = creds.password
            # Actively test connection against WRDS
            db = wrds.Connection(wrds_username=creds.username)
            db.close()
            print(f"Connected as: {creds.username}")
        except Exception as exc:
            # RuntimeError = clean "no credentials" message from the module.
            # A wrong password raises sqlalchemy/psycopg2 OperationalError from
            # wrds.Connection.__init__ (autoconnect), NOT RuntimeError — catch
            # both so validation fails with a message instead of a raw traceback.
            print(f"WRDS connection failed: {exc}", file=sys.stderr)
            return 1
        # Below here the WRDS try/except has already succeeded and returned control
        # normally -- this is deliberately OUTSIDE that try/except (cli.py:101-118).
        # An exception from shards_ready() (a stale TICKHISTORY_PATH, an unreadable
        # scratch mount) would otherwise be caught by the WRDS handler above and
        # misreported as "WRDS connection failed", and would exit 1. LSEG/tick-shard
        # status must never affect connect's exit code: WRDS-only researchers are
        # exactly who this pre-flight (USAGE.md:53,55) must keep working for.
        try:
            from globalmacro.tickhistory_credentials import (
                credentials_present,
                validate_credentials,
            )
            from globalmacro.utils.capabilities import shards_ready
            checked = "--check-lseg" in rest
            present = credentials_present()
            # validate_credentials() makes a live network call; it is reachable
            # ONLY behind an explicit --check-lseg, and only when credentials are
            # present (no point probing a token endpoint with nothing to send).
            ok = validate_credentials() if (checked and present) else False
            identity = _lseg_identity_line(present)
            if identity:
                print(identity)
            print(_capability_report(shards_ready(),
                                     creds=(ok if checked else present),
                                     checked=checked))
        except Exception as exc:                       # never fatal to `connect`
            print(f"(capability report unavailable: {exc})", file=sys.stderr)
        return 0
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
