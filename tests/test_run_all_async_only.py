import difflib
import os
import shutil
import subprocess
from pathlib import Path

from globalmacro.utils.capabilities import SHARD_STEMS

REPO = Path(__file__).resolve().parents[1]
DAG_BASELINE = Path(__file__).resolve().parent / "data" / "dag-full-mode-baseline.txt"


def _dry_run(*args, env=None) -> str:
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", *args],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def _make_shards(root: Path) -> None:
    """A complete, verified shard tree: the 9 SHARD_STEMS x 2 sides, each with the
    _GATE1_OK marker shards_ready() checks for. Imported from capabilities rather
    than hardcoded so this fixture can't silently drift from the real stem list."""
    for stem in SHARD_STEMS:
        for kind in ("trades", "quotes"):
            d = root / kind / f"{stem}_{kind}"
            d.mkdir(parents=True)
            (d / "_GATE1_OK").write_text("gate1 ok\n")


def _fake_repo(tmp_path: Path, python_stub: str | None = None) -> Path:
    """A minimal repo skeleton for tests that need to control which
    .venv/bin/python run_all.sh finds. Under --dry-run, submit() only ever echoes
    the slurm/*.sh paths it's given -- it never checks they exist -- so copying
    run_all.sh itself is enough; the rest of slurm/ need not be mirrored."""
    slurm = tmp_path / "slurm"
    slurm.mkdir()
    (slurm / "run_all.sh").write_text((REPO / "slurm" / "run_all.sh").read_text())
    if python_stub is not None:
        bin_dir = tmp_path / ".venv" / "bin"
        bin_dir.mkdir(parents=True)
        py = bin_dir / "python"
        py.write_text(python_stub)
        os.chmod(py, 0o755)
    return tmp_path


def _dry_run_in(repo: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", *args],
                          cwd=repo, capture_output=True, text=True)


def _sbatch_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.startswith("  sbatch")]


def test_async_only_submits_no_tickhistory_jobs():
    assert "tickhistory.sh" not in _dry_run("--async-only")


def test_async_only_still_submits_all_four_futures_jobs():
    assert _dry_run("--async-only").count("futures.sh") == 4


def test_async_only_passes_the_mode_to_build_and_validate():
    out = _dry_run("--async-only")
    assert "build.sh --async-only" in out
    assert "validate.sh --async-only" in out


def test_async_only_build_depends_on_settlement_futures_only():
    """Guards the exact dependency-id list, not just its presence: a `settlement`
    -> `open` typo in the futures loop (M3) or reading $FUT instead of $FUT_SETTLE
    here (M4) would both silently swap the build dependency onto the wrong futures
    jobs -- the two 24h `open` jobs instead of (or in addition to) the settlement
    pair -- exactly the critical-path regression the settlement-only dependency
    exists to prevent."""
    out = _dry_run("--async-only")
    assert "--dependency=afterok:DRY2:DRY8:DRY4:DRY5 slurm/build.sh --async-only" in out


def test_full_mode_passes_an_explicit_flag_too(tmp_path):
    """Never nothing: build must not re-decide from disk hours later.

    Needs real shards on disk -- an explicit --full fails fast otherwise (see
    test_explicit_full_fails_fast_without_shards below) -- so this builds a fake
    shard tree rather than relying on this machine's production tick data: on a
    researcher machine with no shards this test must still pass.
    """
    _make_shards(tmp_path)
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    out = _dry_run("--full", env=env)
    assert "build.sh --full" in out and "validate.sh --full" in out


def test_explicit_full_fails_fast_without_shards(tmp_path):
    """The complementary shard state to the fixture above: no shards on disk means
    an explicit --full must fail fast (Step 4's whole point) rather than submit a
    DAG with nothing for the tickhistory jobs to read."""
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", "--full"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 1, f"stdout={r.stdout}\nstderr={r.stderr}"


def test_autodetects_async_only_with_no_shards(tmp_path):
    """The headline feature: no flag, no tick data, no sync artifacts either (a
    genuinely clean researcher tree) -> correct mode, no abort.

    FUTURES_DATASETS_ROOT is overridden alongside TICKHISTORY_PATH: this machine
    (like the owner's) has already built the sync half, so leaving DATASETS_ROOT at
    its default would make this test collide with the F1 guard below -- the "sync
    artifacts present" case, not the clean-tree case this test means to cover.
    """
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path / "tick"),
           "FUTURES_DATASETS_ROOT": str(tmp_path / "datasets")}
    assert "mode=async-only" in _dry_run(env=env)


def _stub_sbatch(bin_dir: Path, capture: Path) -> None:
    """A fake `sbatch` on PATH -- never touches the real scheduler (constraint: no
    SLURM submission). Whenever asked to submit a job whose argv mentions
    validate.sh, it dumps its OWN environment (the one it was forked+exec'd with) to
    `capture` before returning a fake job id -- exactly what a real sbatch would hand
    to the validate.sh job's environment under its default --export=ALL. This is the
    only way to observe env-var propagation through `submit()`'s real
    `sbatch --parsable ...` call (line 156): --dry-run's submit() branch never
    invokes sbatch at all, so it cannot exercise this."""
    stub = bin_dir / "sbatch"
    stub.write_text(
        "#!/bin/bash\n"
        "for a in \"$@\"; do\n"
        f'  case "$a" in *validate.sh*) env > "{capture}" ;; esac\n'
        "done\n"
        'echo "fake-job-$$"\n'
    )
    os.chmod(stub, 0o755)


def _intercepted_env(bin_dir: Path, capture: Path, **extra) -> dict:
    """Build the env for a NON-dry `run_all.sh` run, and refuse to hand it back unless
    the stub sbatch provably wins on PATH.

    The two callers below invoke run_all.sh WITHOUT --dry-run, so `submit()` executes a
    real `sbatch ...` for every stage -- the only way to observe env propagation, since
    the --dry-run branch never calls sbatch at all. The real scheduler IS on PATH here
    (/cm/shared/apps/slurm/current/bin/sbatch). If the prepend below ever stopped
    winning, these tests would submit the full 12-job DAG, including build.sh and
    validate.sh, which WRITE production datasets/ and validation/.

    So the interception is asserted here, BEFORE any subprocess starts, and separately
    from the mechanism it protects: this branch's own hard-won rule is that a test which
    exercises a dangerous path must carry its own protection, and the mechanism under
    test must not also be the protection. SLURM_CONF is pointed at a nonexistent file as
    a second, PATH-independent line of defense -- a real sbatch reached by any route
    cannot then find a controller to submit to.
    """
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
           "SLURM_CONF": "/nonexistent/slurm.conf", **extra}
    resolved = shutil.which("sbatch", path=env["PATH"])
    assert resolved == str(bin_dir / "sbatch"), (
        f"stub sbatch did not win on PATH (resolved to {resolved!r}); refusing to run "
        "run_all.sh non-dry, which would submit the real DAG and overwrite production "
        "datasets/ and validation/")
    return env


def test_autodetected_async_only_exports_marker_for_the_validate_job(tmp_path):
    """R2-1 (Opus review, both reviewers independently, PROVED). Task 9 forwards the
    RESOLVED mode as an explicit --async-only whether it was typed or auto-detected --
    `globalmacro validate`'s argv alone can never tell the two apart, which let an
    auto-detected downgrade (e.g. a researcher who reclaimed disk and deleted the tick
    shards but kept the 3 aggregate sync panels) delete 9 shipped validation
    artifacts nobody asked to lose. The fix: `export GM_MODE_AUTODETECTED=1` whenever
    MODE_EXPLICIT=0, before the first submit()/sbatch call -- see
    validation/run.py's explicit_async_only for the consuming side.

    This proves the PROPAGATION half, not just that the export line runs: sbatch's
    documented default (`man sbatch` on this cluster) is `--export=ALL` -- "the
    user's environment will be loaded ... from the caller's environment" -- i.e. the
    SUBMITTING shell's own environment, which is exactly run_all.sh's process. The
    stub sbatch (_stub_sbatch) captures what it actually received."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "validate_sbatch_env.txt"
    _stub_sbatch(bin_dir, capture)
    env = _intercepted_env(bin_dir, capture,
                           TICKHISTORY_PATH=str(tmp_path / "tick"),
                           FUTURES_DATASETS_ROOT=str(tmp_path / "datasets"))
    r = subprocess.run(["bash", "slurm/run_all.sh"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "mode=async-only" in out
    assert capture.exists(), f"stub sbatch was never asked to submit validate.sh:\n{out}"
    assert "GM_MODE_AUTODETECTED=1" in capture.read_text().splitlines()


def test_explicit_async_only_does_not_export_the_autodetect_marker(tmp_path):
    """Complement of the test above: a researcher who TYPES --async-only
    (MODE_EXPLICIT=1) must not have GM_MODE_AUTODETECTED set for the validate job --
    only an auto-detected mode exports it. Without this, the fix would over-correct
    into never deleting stale figures at all, even on a genuine explicit request."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "validate_sbatch_env.txt"
    _stub_sbatch(bin_dir, capture)
    env = _intercepted_env(bin_dir, capture)
    r = subprocess.run(["bash", "slurm/run_all.sh", "--async-only"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert capture.exists(), f"stub sbatch was never asked to submit validate.sh:\n{out}"
    assert "GM_MODE_AUTODETECTED=1" not in capture.read_text().splitlines()


def test_typed_async_only_clears_an_inherited_autodetect_marker(tmp_path):
    """R3-2 (Opus review B). run_all.sh must set the marker on BOTH branches, not just
    the auto-detect one: a GM_MODE_AUTODETECTED inherited from an outer environment
    would otherwise survive a TYPED --async-only and suppress the stale-figure cleanup
    the researcher explicitly requested. The flag has to describe this invocation."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "validate_sbatch_env.txt"
    _stub_sbatch(bin_dir, capture)
    env = _intercepted_env(bin_dir, capture, GM_MODE_AUTODETECTED="1")
    r = subprocess.run(["bash", "slurm/run_all.sh", "--async-only"], cwd=REPO,
                       capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert capture.exists(), f"stub sbatch was never asked to submit validate.sh:\n{out}"
    assert "GM_MODE_AUTODETECTED=1" not in capture.read_text().splitlines(), (
        "an inherited marker survived a typed --async-only")


def _make_sync_panels(root: Path) -> None:
    """sync_panels_ready()'s three files. Since F3a, sync_panels_ready() itself
    consults sync_stage_outputs_ready() first, so a realistic "healthy prior full
    build" fixture needs the ten stage outputs too, or sync_panels_ready() reports
    NOT ready for the wrong reason (missing stage outputs, not missing panels) and
    F1's guard (which checks .ready on both predicates) never fires. Call
    _make_sync_stage_outputs FIRST for exactly this reason -- tier2/currency's path
    is shared between the two fixtures (see test_capabilities.py's identical note),
    so this call also covers it."""
    _make_sync_stage_outputs(root)
    for rel in (
        "tier1/sync/sync_daily.csv",
        "tier2/sync/sync_daily.csv",
        "tier2/sync/currency_daily_returns.csv",
    ):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("date\n")


def _make_sync_stage_outputs(root: Path) -> None:
    """The OTHER of F1's two predicates: tickhistory-stage outputs, not the
    aggregated sync panels. Exercises sync_stage_outputs_ready() specifically, so
    the "either" in F1's guard is proved on both branches, not just one."""
    from globalmacro.utils.capabilities import SYNC_STAGE_OUTPUTS
    for tier, cls in SYNC_STAGE_OUTPUTS:
        p = root / tier / "sync" / f"{cls}_daily_returns.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("date\n")


def test_shards_absent_sync_panels_present_no_flag_aborts(tmp_path):
    """F1 (Reviewer A): auto-detect must not silently downgrade a machine that has
    already built the sync half. No tick shards + no explicit flag -> exit 1,
    message names the cause and both remedies.

    N7 (Reviewer B): since F3a, sync_panels_ready() consults sync_stage_outputs_
    ready() first and returns ITS result whenever it isn't ready -- so
    sync_panels_ready().ready now implies sync_stage_outputs_ready().ready, and
    _make_sync_panels (which calls _make_sync_stage_outputs before writing the 3
    aggregates, for exactly this reason) can no longer construct "panels ready,
    stage outputs not" in isolation. This test therefore exercises BOTH predicates
    ready together, not "sync panels" alone -- F1's "either" can no longer trigger
    on panels alone, only on stage outputs (which the companion test below,
    test_shards_absent_stage_outputs_present_no_flag_aborts, covers as the
    genuinely isolable case: stage outputs present, aggregate panels absent)."""
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path / "tick"),
           "FUTURES_DATASETS_ROOT": str(tmp_path / "datasets")}
    _make_sync_panels(tmp_path / "datasets")
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "already has sync" in out
    assert "sync panels" in out
    assert "repair the tick shards" in out
    assert "--async-only" in out


def test_shards_absent_stage_outputs_present_no_flag_aborts(tmp_path):
    """Same defect, other predicate: tickhistory stage outputs (not the aggregated
    sync panels) present with no tick shards and no explicit flag -> exit 1."""
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path / "tick"),
           "FUTURES_DATASETS_ROOT": str(tmp_path / "datasets")}
    _make_sync_stage_outputs(tmp_path / "datasets")
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    out = r.stdout + r.stderr
    assert r.returncode == 1, out
    assert "already has sync" in out
    assert "tickhistory stage outputs" in out


def test_shards_absent_sync_present_explicit_async_only_proceeds(tmp_path):
    """Same on-disk state as the abort tests above, but --async-only is explicit --
    the flag makes the downgrade deliberate, so it must proceed normally: exit 0,
    async-only DAG submitted, no abort message."""
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path / "tick"),
           "FUTURES_DATASETS_ROOT": str(tmp_path / "datasets")}
    _make_sync_panels(tmp_path / "datasets")
    out = _dry_run("--async-only", env=env)
    assert "build.sh --async-only" in out
    assert "validate.sh --async-only" in out
    assert "already has sync" not in out


def test_unknown_flag_still_rejected():
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", "--nope"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2


def test_conflicting_mode_flags_rejected():
    """--async-only --full (or the reverse) must not silently last-win: build.py's
    argparse mutually-exclusive group already rejects this combination for a direct
    `globalmacro build` invocation (tests/test_build_mode.py), so the launcher
    should match rather than picking whichever flag came last."""
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", "--async-only", "--full"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2
    r2 = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", "--full", "--async-only"],
                        cwd=REPO, capture_output=True, text=True)
    assert r2.returncode == 2


def test_capability_banner_is_not_mistaken_for_the_mode(tmp_path):
    """A library that logs a line on import must not make that stray line become
    $MODE: unguarded, `head -1` of the capability stdout takes it verbatim, which
    both silently degrades the DAG to async-only (garbage fails `[ "$MODE" = full
    ]`) and, unquoted, would word-split `--$MODE` into extra sbatch arguments so
    build and validate receive different (broken) mode flags."""
    repo = _fake_repo(tmp_path, python_stub=(
        "#!/bin/bash\n"
        "echo 'some library banner'\n"
        "echo 'full'\n"
        "echo ''\n"
    ))
    r = _dry_run_in(repo)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "mode=full" in out
    assert "validate.sh --full" in out
    # word-splitting would append " library" / " banner" as extra sbatch args
    assert "validate.sh --full library" not in out
    assert "validate.sh --some" not in out


def test_broken_venv_capability_check_falls_back_to_full(tmp_path):
    """If the capability check crashes outright (broken venv: import error, etc.),
    run_all.sh must not abort under `set -e` -- the `|| _CAP=""` at :42 exists so a
    crash degrades to the same "assume --full" fallback as a missing venv, and
    because MODE_EXPLICIT is correctly 0 here, the fail-fast re-check at :71 must
    stay unreached rather than re-running (and re-failing on) the same broken
    import."""
    repo = _fake_repo(tmp_path, python_stub=(
        "#!/bin/bash\n"
        "echo 'ModuleNotFoundError: no module named globalmacro' >&2\n"
        "exit 1\n"
    ))
    r = _dry_run_in(repo)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "mode=full" in out


def test_capability_check_error_is_surfaced_not_discarded(tmp_path):
    """F7: the capability probe's stderr must no longer be thrown away (was
    2>/dev/null). A broken/partial `uv sync`, a .venv built for another
    interpreter, or a misresolved root previously yielded MODE=full with only
    "capability check failed; assuming --full" -- the actual import error was
    discarded, leaving a tick-less researcher with no clue why the DAG stalled
    behind 12 tickhistory jobs it fails-toward-full submitted. The fail-toward-full
    direction is unchanged (still mode=full); this only adds visibility into why."""
    repo = _fake_repo(tmp_path, python_stub=(
        "#!/bin/bash\n"
        "echo 'ModuleNotFoundError: no module named globalmacro' >&2\n"
        "echo 'Traceback (most recent call last):' >&2\n"
        "exit 1\n"
    ))
    r = _dry_run_in(repo)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "mode=full" in out
    assert "ModuleNotFoundError: no module named globalmacro" in out


def test_full_mode_dag_matches_committed_baseline(tmp_path):
    """Regression guard for the full-mode DAG's dependency edges. Step 8b's
    comparison against a hand-captured DAG was a one-time manual check whose
    original capture lives at .superpowers/sdd/dag-pre-task9.txt -- gitignored, so
    this commits an equivalent copy under tests/data/dag-full-mode-baseline.txt.
    tests/test_capabilities.py's shard-stem guard only greps the two `for c in
    ...; do` class lists, so it can't catch dependency-edge regressions such as a
    dropped ${jLON2}, a tier-2 loop's ids never entering $TICK, or the london->ET
    serialization edge going missing.

    This is the POST-Task-9 full-mode DAG (F11): it includes --full on the
    build.sh/validate.sh lines, since passing an explicit mode through is current,
    correct Task-9 behaviour. The original .superpowers/sdd/dag-pre-task9.txt
    capture predates that and has no flag there -- this committed copy is not a
    byte-for-byte copy of it, just an equivalent baseline captured after Task 9.
    """
    _make_shards(tmp_path)
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    out = _dry_run("--full", env=env)
    actual = _sbatch_lines(out)
    expected = DAG_BASELINE.read_text().splitlines()
    if actual != expected:
        diff = "\n".join(difflib.unified_diff(
            expected, actual, fromfile="tests/data/dag-full-mode-baseline.txt",
            tofile="actual", lineterm=""))
        raise AssertionError(f"full-mode DAG drifted from the committed baseline:\n{diff}")
