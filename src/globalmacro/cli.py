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


def _capability_report(
    shard_cap, creds: bool, checked: bool, present: bool | None = None
) -> str:
    """What can this machine build, and what should the researcher run next?

    `creds`/`checked`/`present` are booleans, not credential values -- this function
    never touches an actual username or password, only presence/validity flags
    computed by its caller.

    `present` is a 4th argument on top of the original 3-argument signature:
    collapsing "credentials present but rejected by LSEG" into the same `creds=False`
    state as "no credentials at all" told a researcher whose `--check-lseg` run failed
    that no credentials were found, when in fact some were found and rejected.
    `present` distinguishes those two states. It defaults to `None`, in which case
    this function falls back to `present = creds` -- the exact quantity a 3-argument
    caller always passed as `creds` before this parameter existed. That fallback is
    what keeps a 3-argument call bit-identical to the original 3-argument signature;
    a bare `present: bool = True` default would NOT do this (it makes every
    `creds=False` 3-argument call report "rejected" instead of "no credentials found",
    since `present=True` while `creds=False` is exactly the rejected state).
    """
    from globalmacro.utils.capabilities import shard_dirs
    if present is None:
        present = creds
    if not present:
        lseg = "no credentials found"
    elif not checked:
        lseg = "credentials present (not checked -- run `connect --check-lseg` to verify)"
    elif creds:
        lseg = "credentials valid"
    else:
        lseg = "credentials present but rejected -- check DSS_USERNAME/DSS_PASSWORD"

    n = len(shard_dirs())
    tick = f"ready ({n}/{n} shard sets)" if shard_cap.ready else "not ready"

    lines = [f"LSEG TickHistory:  {lseg}"]
    if not present:
        lines.append(
            "                   set DSS_USERNAME and DSS_PASSWORD (e.g. in .env) "
            "to enable LSEG tick-history checks.")
    lines.append(f"Tick data:         {tick}")
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
    """The LSEG identity line for `connect`'s capability report: the DSS username,
    printed whole and never masked, when credentials are present. Consistent with
    `connect`'s existing WRDS output a few lines above (`Connected as: {creds.username}`)
    and with wrds_credentials.py's on-disk state file, which already persists a WRDS
    username unmasked -- a login identifier is not a secret the way a password is.
    This function never reads DSS_PASSWORD, directly or indirectly:
    `credential_username()` only resolves DSS_USERNAME.

    `present` is expected to come from `credentials_present()`, which requires both
    DSS_USERNAME and DSS_PASSWORD to be set -- so when `present` is True,
    `credential_username()` is guaranteed non-empty at the sole call site in `main()`.
    """
    if not present:
        return None
    from globalmacro.tickhistory_credentials import credential_username
    username = credential_username()
    return f"LSEG username:     {username}"


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    choices = sorted([*_STAGE_MODULES, *_FUNCTION_STAGES])

    # Dispatch by hand (NOT argparse) so a stage's own `--help` is forwarded to it
    # rather than intercepted by a top-level parser's implicit -h/--help.
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: globalmacro <stage> [args...]\n\nstages:\n  " + "\n  ".join(choices) +
              "\n\nconnect --check-lseg  also validate LSEG credentials "
              "(makes a network call; WRDS-only check otherwise)")
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
                                     checked=checked,
                                     present=present))
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
