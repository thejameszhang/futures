import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _dry_run(*args, env=None) -> str:
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", *args],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"exit={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def test_async_only_submits_no_tickhistory_jobs():
    assert "tickhistory.sh" not in _dry_run("--async-only")


def test_async_only_still_submits_all_four_futures_jobs():
    assert _dry_run("--async-only").count("futures.sh") == 4


def test_async_only_passes_the_mode_to_build_and_validate():
    out = _dry_run("--async-only")
    assert "build.sh --async-only" in out
    assert "validate.sh --async-only" in out


def test_full_mode_passes_an_explicit_flag_too():
    """Never nothing: build must not re-decide from disk hours later."""
    out = _dry_run("--full")
    assert "build.sh --full" in out and "validate.sh --full" in out


def test_autodetects_async_only_with_no_shards(tmp_path):
    """The headline feature: no flag, no tick data, correct mode."""
    import os
    env = {**os.environ, "TICKHISTORY_PATH": str(tmp_path)}
    assert "mode=async-only" in _dry_run(env=env)


def test_unknown_flag_still_rejected():
    r = subprocess.run(["bash", "slurm/run_all.sh", "--dry-run", "--nope"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 2
