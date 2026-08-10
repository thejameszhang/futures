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
    from globalmacro.utils import paths as _paths
    from globalmacro.utils.capabilities import shard_dirs
    if present is None:
        present = creds

    # The JKP sector file is a genuine, manual, never-downloadable
    # prerequisite -- load_sectors_async() reads it unconditionally, in BOTH
    # modes (build.main() always runs build_async() first, even under --full).
    # Flagging it pre-run is correct and valuable: a missing copy otherwise
    # surfaces as a raw FileNotFoundError deep inside build. Verified against
    # paths.py rather than assumed: DATA_ROOT is the real constant
    # load_sectors_async() resolves its path from (build.py).
    #
    # Do not probe FX_PATH/synthetic_fx_returns_sync.csv here: that file is a
    # pipeline OUTPUT, not a prerequisite -- fx.py's __main__ writes it via
    # build_synthetic(fx_sync).write_csv(...) -- so it is absent on 100% of clean
    # clones, including a researcher with full Compustat entitlement who has done
    # everything right. No file-presence check can establish a Compustat
    # entitlement before the first download (exrt_dly.csv has the identical
    # problem, so it is not a substitute); the unconditional line appended to both
    # verdict branches below replaces the check with a plain statement instead.
    _jkp_sectors = _paths.DATA_ROOT / "jkp" / "updated_daily_ind_gics.csv"
    missing_manual_prereqs = [] if _jkp_sectors.exists() else [_jkp_sectors]
    # build_synced_dataset (build.py) reads DATA_ROOT/jkp/updated_daily_ind_gics_synced.csv
    # in the full/sync build path, and capabilities.py's
    # sync_stage_outputs_ready() docstring already flags that this file sits outside its
    # predicate -- a manual input, never a pipeline output, verified against USAGE.md's
    # Manual Prerequisite Data Files section, same as _jkp_sectors above. Checked only
    # when shard_cap.ready, matching build_synced_dataset's own reachability
    # (build.main() only calls it under `if mode == "full"`, and resolve_mode() never
    # picks full without shard_cap.ready): an async-only machine does not call it and,
    # per USAGE.md, does not need this file -- checking it unconditionally would falsely
    # block a report that is otherwise correctly async-only. Without this, a shard-ready
    # machine missing only this file was told it could build the SYNC datasets, then
    # failed later in `globalmacro build --full`.
    if shard_cap.ready:
        _jkp_sectors_synced = _paths.DATA_ROOT / "jkp" / "updated_daily_ind_gics_synced.csv"
        if not _jkp_sectors_synced.exists():
            missing_manual_prereqs.append(_jkp_sectors_synced)
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
    # The JKP check runs BEFORE the shard_cap.ready branch, not only in
    # its own elif -- otherwise a shard-ready machine
    # missing the JKP file could be told "can build the SYNC and ASYNC datasets" (JKP
    # gates build_async(), which build.main() calls unconditionally in every
    # mode, so shard readiness alone never implied JKP readiness). Applying the
    # SAME corrected check to both success branches, rather than adding a second,
    # differently-scoped check to shard_cap.ready, keeps exactly one place that
    # can go stale.
    if missing_manual_prereqs:
        # Named and pointed at USAGE.md rather than left to a later
        # FileNotFoundError or a DAG stalled behind a failed `comp` download.
        #
        # The missing JKP file blocks BOTH modes, not just async -- verified
        # against build.py's own main(), which calls build_async() (the function
        # that reads this file, via load_sectors_async()) UNCONDITIONALLY, before
        # the `if mode == "full"` branch that gates build_sync(). A full-mode run
        # never even reaches build_sync() while this file is missing.
        lines.append(
            "-> This machine needs one more file before it can build ANY dataset "
            "(build always runs the async half first):"
        )
        for p in missing_manual_prereqs:
            lines.append(f"   {p}")
        lines.append("   See USAGE.md's Detailed Data Prerequisites section.")
        # Both "can build" branches below carry this reminder; a researcher
        # missing the JKP file needs it just as much (Compustat gates the OTHER
        # manual prerequisite this same report can't verify from a file on disk --
        # see the note above about FX_PATH/synthetic_fx_returns_sync.csv), and
        # previously never saw it.
        lines.append("   Also requires a Compustat entitlement (comp.exrt_dly) -- see USAGE.md.")
    elif shard_cap.ready:
        lines.append("-> This machine can build the SYNC and ASYNC datasets.")
        lines.append("   Also requires a Compustat entitlement (comp.exrt_dly) -- see USAGE.md.")
    else:
        lines.append("-> This machine can build the ASYNC datasets.")
        lines.append("   Sync datasets need LSEG tick data on disk.")
        lines.append("   Also requires a Compustat entitlement (comp.exrt_dly) -- see USAGE.md.")
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
        # Below here the WRDS try/except above has already succeeded and returned
        # control normally -- this block is deliberately OUTSIDE it. An exception
        # from shards_ready() (a stale TICKHISTORY_PATH, an unreadable scratch
        # mount) would otherwise be caught by the WRDS handler and misreported as
        # "WRDS connection failed", exiting 1. LSEG/tick-shard status must never
        # affect connect's exit code: WRDS-only researchers are exactly who the
        # WRDS-credentials setup note and the HPC batch-execution warning in
        # USAGE.md's WRDS-downloadable-inputs section must keep working for.
        # Both references here are prose, not line numbers: the citation this
        # replaced was accurate when written but rotted silently eight commits
        # later when that doc section grew, and shifted again while being fixed.
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
