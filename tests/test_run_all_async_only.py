import difflib
import os
import subprocess
from pathlib import Path

from globalmacro.utils.capabilities import SHARD_STEMS

REPO = Path(__file__).resolve().parents[1]
DAG_BASELINE = Path(__file__).resolve().parent / "data" / "dag-pre-task9.txt"


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
    """The headline feature: no flag, no tick data, correct mode."""
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    assert "mode=async-only" in _dry_run(env=env)


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


def test_full_mode_dag_matches_committed_baseline(tmp_path):
    """Regression guard for the full-mode DAG's dependency edges. Step 8b's
    comparison against the pre-task9 DAG was a one-time manual check whose baseline
    lives at .superpowers/sdd/dag-pre-task9.txt -- gitignored, so this commits an
    equivalent copy under tests/data/. tests/test_capabilities.py's shard-stem
    guard only greps the two `for c in ...; do` class lists, so it can't catch
    dependency-edge regressions such as a dropped ${jLON2}, a tier-2 loop's ids
    never entering $TICK, or the london->ET serialization edge going missing.

    The committed baseline includes --full on the build.sh/validate.sh lines,
    since passing an explicit mode through is current, correct, Task-9 behaviour
    (the pre-task9 capture predates that and has no flag there).
    """
    _make_shards(tmp_path)
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    out = _dry_run("--full", env=env)
    actual = _sbatch_lines(out)
    expected = DAG_BASELINE.read_text().splitlines()
    if actual != expected:
        diff = "\n".join(difflib.unified_diff(
            expected, actual, fromfile="tests/data/dag-pre-task9.txt", tofile="actual",
            lineterm=""))
        raise AssertionError(f"full-mode DAG drifted from the committed baseline:\n{diff}")
