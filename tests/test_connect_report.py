"""Tests for `globalmacro connect`'s capability report (Task 8).

Hermetic by construction: every test that reaches `cli.main(["connect", ...])`
fakes out `wrds_credentials.get_wrds_credentials` and the `wrds` module itself, so
none of them touch a real WRDS connection, a real ~/.pgpass, or the system keyring.
None of them read/write DATASETS_ROOT, VALIDATION_OUTPUT, or TICKHISTORY_PATH --
`capabilities.shards_ready` is always monkeypatched to a fixed `Capability`, never
called for real. `tc.validate_credentials` is replaced by the repo-wide autouse
`_no_live_lseg_network` fixture (tests/conftest.py) for every test in this file, so
no test can reach the real function -- and therefore
selectapi.datascope.refinitiv.com -- even if the `checked and present` gate in
`cli.py` regresses; the two tests that legitimately want to drive it install their
own stub, which simply overrides that fixture's for the duration of that test.
DSS_USERNAME/DSS_PASSWORD are always set via monkeypatch.setenv/delenv, never
inherited from a real .env.
"""
import sys
import types

import globalmacro.tickhistory_credentials as tc
import globalmacro.utils.capabilities as capmod
import globalmacro.utils.paths as pathsmod
import globalmacro.wrds_credentials as wc
from globalmacro import cli
from globalmacro.utils.capabilities import Capability

# ---------------------------------------------------------------------------
# `_capability_report`'s core state space: shard readiness drives the ASYNC-only vs
# SYNC-and-ASYNC verdict text, a shard warning surfaces verbatim, and the LSEG line
# distinguishes not-yet-checked from checked-and-valid.
# ---------------------------------------------------------------------------


def _with_jkp_sectors(monkeypatch, tmp_path) -> None:
    """Construct a DATA_ROOT with the JKP sector file present, so the "can build"
    verdict branches are reached without depending on this machine's real data/
    (F15) -- since F14, `_capability_report` checks this file BEFORE the
    shard_cap.ready branch too, so any test that wants a "can build" verdict
    (SYNC-and-ASYNC or ASYNC-only) needs it constructed, not just the
    JKP-missing tests."""
    root = tmp_path / "data"
    monkeypatch.setattr(pathsmod, "DATA_ROOT", root)
    (root / "jkp").mkdir(parents=True)
    (root / "jkp" / "updated_daily_ind_gics.csv").write_text("date\n")


def test_report_says_async_when_no_shards(tmp_path, monkeypatch):
    """Assert on whole clauses: "SYNC" is a substring of "ASYNC"."""
    _with_jkp_sectors(monkeypatch, tmp_path)
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "can build the ASYNC datasets" in out
    assert "Sync datasets need LSEG tick data" in out
    assert "globalmacro run --with-download" in out


def test_report_says_both_when_shards_ready(tmp_path, monkeypatch):
    _with_jkp_sectors(monkeypatch, tmp_path)
    out = cli._capability_report(Capability(True, None), creds=True, checked=False)
    assert "can build the SYNC and ASYNC datasets" in out
    assert "Sync datasets need LSEG tick data" not in out


def test_report_surfaces_the_shard_warning():
    cap = Capability(False, "missing tick shard directories: tier2_equity_trades")
    out = cli._capability_report(cap, creds=False, checked=False)
    assert "tier2_equity_trades" in out


def test_report_distinguishes_presence_from_validation():
    assert "not checked" in cli._capability_report(
        Capability(False, None), creds=True, checked=False)
    assert "credentials valid" in cli._capability_report(
        Capability(False, None), creds=True, checked=True)


def test_report_gives_a_remediation_hint_when_no_credentials():
    """The remediation hint is a note about the LSEG line, not the Tick data line --
    it must appear immediately after `LSEG TickHistory:` and before `Tick data:`, in
    the same 19-space continuation column used elsewhere for shard warnings."""
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    lines = out.splitlines()
    lseg_idx = next(i for i, line in enumerate(lines) if line.startswith("LSEG TickHistory:"))
    tick_idx = next(i for i, line in enumerate(lines) if line.startswith("Tick data:"))
    assert lines[lseg_idx + 1] == (
        "                   set DSS_USERNAME and DSS_PASSWORD (e.g. in .env) "
        "to enable LSEG tick-history checks.")
    assert lseg_idx + 1 < tick_idx


# ---------------------------------------------------------------------------
# F6/F14: `connect` must not claim it can build the ASYNC datasets when the JKP
# sector file build_async() needs unconditionally -- in BOTH modes -- is missing.
# F14 (Reviewer B, BLOCKER): an earlier version of this section also probed
# FX_PATH/synthetic_fx_returns_sync.csv, but that file is a pipeline OUTPUT
# (fx.py's __main__ writes it), not a prerequisite -- it is absent on 100% of
# clean clones, so that probe told every clean-clone researcher, including one
# with full Compustat entitlement, that they could not build the ASYNC datasets.
# The check was dropped entirely; only the genuinely-manual JKP file remains
# checked, and a plain (non-conditional) Compustat-entitlement reminder replaces
# it in the two "can build" verdicts (see test_missing_synthetic_fx_sync_file_
# no_longer_blocks_the_report below).
#
# F15: every test in this section constructs its own DATA_ROOT under tmp_path --
# never this machine's real data/ -- so it passes identically on a clean
# researcher clone with nothing in data/jkp/.
# ---------------------------------------------------------------------------


def test_report_names_missing_jkp_sectors_file(tmp_path, monkeypatch):
    monkeypatch.setattr(pathsmod, "DATA_ROOT", tmp_path / "no-data")
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "needs one more file" in out
    assert "updated_daily_ind_gics.csv" in out
    assert "USAGE.md" in out
    assert "can build the ASYNC datasets." not in out


def test_report_names_missing_jkp_sectors_file_says_it_blocks_any_dataset(tmp_path, monkeypatch):
    """R2-3: build.main() calls build_async() (the function that reads the JKP
    sector file, via load_sectors_async()) UNCONDITIONALLY, before the `if mode ==
    "full"` branch that gates build_sync() -- so the missing file blocks SYNC just
    as much as ASYNC, not only ASYNC as the old wording implied."""
    monkeypatch.setattr(pathsmod, "DATA_ROOT", tmp_path / "no-data")
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "before it can build ANY dataset" in out
    assert "before it can build the ASYNC datasets" not in out


def test_report_names_missing_jkp_sectors_file_still_mentions_compustat(tmp_path, monkeypatch):
    """R2-4: both "can build" verdict branches carry the Compustat-entitlement
    reminder; the missing-JKP branch omitted it, so a researcher missing JKP never
    saw it even though Compustat is required in every mode too."""
    monkeypatch.setattr(pathsmod, "DATA_ROOT", tmp_path / "no-data")
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "Compustat entitlement" in out


def test_missing_synthetic_fx_sync_file_no_longer_blocks_the_report(tmp_path, monkeypatch):
    """F14 regression proof: FX_PATH/synthetic_fx_returns_sync.csv is a pipeline
    OUTPUT, not a prerequisite -- a missing copy (the state of every clean clone)
    must no longer trigger the "needs one more file" verdict, only the
    genuinely-manual JKP file can. JKP is present here (constructed, F15) so this
    isolates the FX_PATH-is-now-ignored behaviour from the JKP check."""
    _with_jkp_sectors(monkeypatch, tmp_path)
    monkeypatch.setattr(pathsmod, "FX_PATH", tmp_path / "no-fx")   # never created
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "needs one more file" not in out
    assert "synthetic_fx_returns_sync.csv" not in out
    assert "can build the ASYNC datasets." in out
    assert "Compustat entitlement" in out


def test_exit_code_zero_when_jkp_missing(monkeypatch, tmp_path):
    _fake_wrds_success(monkeypatch)
    monkeypatch.delenv(tc.ENV_USERNAME, raising=False)
    monkeypatch.delenv(tc.ENV_PASSWORD, raising=False)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    monkeypatch.setattr(pathsmod, "DATA_ROOT", tmp_path / "no-data")
    assert cli.main(["connect"]) == 0


def test_exit_code_zero_when_both_present_no_shards(tmp_path, monkeypatch):
    """Both prerequisites present, no shards -- hermetic (F15): DATA_ROOT points
    at a constructed tmp_path with the JKP file actually on disk, never this
    machine's real files."""
    _with_jkp_sectors(monkeypatch, tmp_path)
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "can build the ASYNC datasets." in out
    assert "needs one more file" not in out


def test_report_states_the_shard_count_when_ready():
    """The `N/N shard sets` fraction is `len(shard_dirs())`, not a hardcoded number --
    pin the relationship, not a literal, so this doesn't drift if SHARD_STEMS grows."""
    n = len(capmod.shard_dirs())
    out = cli._capability_report(Capability(True, None), creds=True, checked=False)
    assert f"ready ({n}/{n} shard sets)" in out


# ---------------------------------------------------------------------------
# A failed --check-lseg must say credentials were found and rejected, not
# collapse into the same "no credentials found" state as never having any.
# ---------------------------------------------------------------------------

def test_report_distinguishes_rejected_from_absent():
    rejected = cli._capability_report(
        Capability(False, None), creds=False, checked=True, present=True)
    assert "credentials present but rejected -- check DSS_USERNAME/DSS_PASSWORD" in rejected
    assert "no credentials found" not in rejected

    absent = cli._capability_report(
        Capability(False, None), creds=False, checked=True, present=False)
    assert "no credentials found" in absent
    assert "rejected" not in absent


# ---------------------------------------------------------------------------
# The LSEG identity line: prints the DSS username whole (never masked, consistent
# with `connect`'s existing unmasked WRDS username output), only when credentials
# are present, and never touches DSS_PASSWORD.
# ---------------------------------------------------------------------------

def test_identity_line_absent_without_credentials(monkeypatch):
    # Genuinely clear, not just rely on an ambient .env being unset: paths._load_dotenv()
    # setdefaults DSS_USERNAME/DSS_PASSWORD from the real .env at import time regardless
    # of the shell environment, so only monkeypatch.delenv (post-import) proves this.
    monkeypatch.delenv(tc.ENV_USERNAME, raising=False)
    monkeypatch.delenv(tc.ENV_PASSWORD, raising=False)
    assert cli._lseg_identity_line(False) is None


def test_identity_line_prints_username_whole_when_present(monkeypatch):
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "hunter2-do-not-print-me")
    line = cli._lseg_identity_line(True)
    assert line is not None
    assert "researcher123" in line


def test_identity_line_never_contains_the_password(monkeypatch):
    secret = "s3cr3t-token-XYZ"
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, secret)
    line = cli._lseg_identity_line(True)
    assert secret not in (line or "")


# ---------------------------------------------------------------------------
# Fakes for the WRDS try/except so `connect` can be exercised end to end
# without ever touching a real network, ~/.pgpass, or the system keyring.
# ---------------------------------------------------------------------------

class _FakeCreds:
    username = "fake_wrds_user"
    password = None


class _FakeConnection:
    def __init__(self, *a, **kw):
        pass

    def close(self):
        pass


def _fake_wrds_success(monkeypatch):
    monkeypatch.setattr(wc, "get_wrds_credentials", lambda: _FakeCreds())
    fake_wrds = types.ModuleType("wrds")
    monkeypatch.setattr(fake_wrds, "Connection", _FakeConnection, raising=False)
    monkeypatch.setitem(sys.modules, "wrds", fake_wrds)


# ---------------------------------------------------------------------------
# Hard requirement 1: LSEG/tick-shard status must never change the exit code.
# ---------------------------------------------------------------------------

def test_exit_code_is_zero_on_wrds_success_regardless_of_lseg(monkeypatch, capsys):
    """This is also the flagship end-to-end state: a WRDS-only researcher with no
    LSEG credentials at all, `connect`'s own default `--check-lseg`-less path. The
    LSEG-line assertions below are what catch `present` defaulting the wrong way at
    the `_capability_report` call site (`cli.py`'s `present=present` regressing to a
    hardcoded `present=True`) -- without them this test only checks the exit code and
    passes regardless of which credential state the report claims."""
    _fake_wrds_success(monkeypatch)
    monkeypatch.delenv(tc.ENV_USERNAME, raising=False)
    monkeypatch.delenv(tc.ENV_PASSWORD, raising=False)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    assert cli.main(["connect"]) == 0
    out = capsys.readouterr()
    assert "no credentials found" in out.out
    assert "LSEG username:" not in out.out


def test_exit_code_is_zero_even_if_capability_report_raises(monkeypatch, capsys):
    """Hard requirement 2: a broken shards_ready() (stale TICKHISTORY_PATH, an
    unreadable scratch mount) must degrade to a warning, not a WRDS failure."""
    _fake_wrds_success(monkeypatch)
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")

    def _boom():
        raise OSError("stale TICKHISTORY_PATH: scratch mount unreadable")

    monkeypatch.setattr(capmod, "shards_ready", _boom)
    rc = cli.main(["connect"])
    out = capsys.readouterr()
    assert rc == 0
    assert "WRDS connection failed" not in out.err
    assert "capability report unavailable" in out.err


def test_exit_code_is_one_on_wrds_failure_regardless_of_lseg(monkeypatch, capsys):
    def _raise():
        raise RuntimeError("no WRDS credentials")

    monkeypatch.setattr(wc, "get_wrds_credentials", _raise)
    # Valid-looking LSEG credentials present -- must not rescue the exit code.
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")
    rc = cli.main(["connect"])
    out = capsys.readouterr()
    assert rc == 1
    assert "WRDS connection failed" in out.err


# ---------------------------------------------------------------------------
# --check-lseg gating: validate_credentials must be reachable ONLY behind it,
# and never actually invoked for real (it would make a live network call).
# ---------------------------------------------------------------------------

def test_check_lseg_absent_never_calls_validate_credentials(monkeypatch):
    # A raise-based tripwire can never work here -- cli.py's capability-report
    # block wraps everything in `except Exception`, so an AssertionError raised by a
    # fake validate_credentials() is caught and silently swallowed, and `main()`
    # still returns 0 either way. Recording whether the call happened is the only
    # pattern that actually distinguishes "called" from "not called".
    _fake_wrds_success(monkeypatch)
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")

    calls = []
    monkeypatch.setattr(tc, "validate_credentials", lambda: calls.append(1) or True)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    assert cli.main(["connect"]) == 0
    assert calls == []


def test_check_lseg_present_calls_validate_credentials(monkeypatch, capsys):
    _fake_wrds_success(monkeypatch)
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")

    calls = []
    monkeypatch.setattr(tc, "validate_credentials", lambda: calls.append(1) or True)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    rc = cli.main(["connect", "--check-lseg"])
    out = capsys.readouterr()
    assert rc == 0
    assert calls == [1]
    assert "credentials valid" in out.out
    # The identity-line wiring (cli.py's `identity = _lseg_identity_line(present);
    # if identity: print(identity)`) is only exercised end to end here -- the identity
    # tests above call the helper directly and never go through main().
    assert "LSEG username:     researcher123" in out.out


def test_check_lseg_present_and_rejected_reports_rejection_not_absence(monkeypatch, capsys):
    """A failed --check-lseg must be reported as found-and-rejected, not as
    'no credentials found' -- and the live check's actual answer (False) must be
    what drives that, not merely whether it ran."""
    _fake_wrds_success(monkeypatch)
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")

    calls = []
    monkeypatch.setattr(tc, "validate_credentials", lambda: calls.append(1) or False)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    rc = cli.main(["connect", "--check-lseg"])
    out = capsys.readouterr()
    assert rc == 0
    assert calls == [1]
    assert "credentials present but rejected -- check DSS_USERNAME/DSS_PASSWORD" in out.out
    assert "no credentials found" not in out.out
