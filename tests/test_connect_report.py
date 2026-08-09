"""Tests for `globalmacro connect`'s capability report (Task 8).

Hermetic by construction: every test that reaches `cli.main(["connect", ...])`
fakes out `wrds_credentials.get_wrds_credentials` and the `wrds` module itself, so
none of them touch a real WRDS connection, a real ~/.pgpass, or the system keyring.
None of them read/write DATASETS_ROOT, VALIDATION_OUTPUT, or TICKHISTORY_PATH --
`capabilities.shards_ready` is always monkeypatched to a fixed `Capability`, never
called for real. `tc.validate_credentials` is replaced by the autouse
`_no_live_lseg_network` fixture below for every test in this file, so no test can
reach the real function -- and therefore selectapi.datascope.refinitiv.com -- even
if the `checked and present` gate in `cli.py` regresses; the two tests that
legitimately want to drive it install their own stub, which simply overrides this
fixture's for the duration of that test. DSS_USERNAME/DSS_PASSWORD are always set
via monkeypatch.setenv/delenv, never inherited from a real .env.
"""
import sys
import types

import pytest

import globalmacro.tickhistory_credentials as tc
import globalmacro.utils.capabilities as capmod
import globalmacro.wrds_credentials as wc
from globalmacro import cli
from globalmacro.utils.capabilities import Capability

# ---------------------------------------------------------------------------
# A safety net against a live network call, applied automatically to every test
# in this module. `validate_credentials` makes a real HTTPS request; production only
# reaches it behind `checked and present` in `cli.py`, but three tests below
# (`test_exit_code_is_zero_on_wrds_success_regardless_of_lseg`,
# `test_exit_code_is_zero_even_if_capability_report_raises`,
# `test_exit_code_is_one_on_wrds_failure_regardless_of_lseg`) reach
# `cli.main(["connect"])` without mocking `validate_credentials` themselves, relying
# entirely on that one gate holding. This fixture removes that reliance instead of
# hoping the gate never regresses: it replaces `tc.validate_credentials` with a
# recording stub for every test, then -- after the test body has run -- asserts the
# stub was never invoked, unless the test opted out by name to install its own stub
# (the two tests that deliberately drive `--check-lseg` end to end).
#
# A raise-based tripwire cannot do this job: `cli.py`'s capability-report block
# wraps everything in `except Exception`, so a stub that raises from inside
# `cli.main()` is silently swallowed and proves nothing. Recording calls and
# asserting on the list after the test returns -- outside any `try/except` the
# production code owns -- is what actually distinguishes "called" from "not called".
#
# Function-scoped, not pytest's `scope="module"` (the built-in `monkeypatch` fixture
# it depends on is function-scoped, so a literal module-scoped fixture would need a
# raw `pytest.MonkeyPatch()` instead) -- but `autouse=True` at module level gives
# every test in this file the same guarantee, which is the property that matters.
# ---------------------------------------------------------------------------

_DRIVES_VALIDATE_CREDENTIALS_ITSELF = {
    "test_check_lseg_present_calls_validate_credentials",
    "test_check_lseg_present_and_rejected_reports_rejection_not_absence",
}


@pytest.fixture(autouse=True)
def _no_live_lseg_network(request, monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(tc, "validate_credentials", lambda: calls.append(1) or False)
    yield
    if request.node.name not in _DRIVES_VALIDATE_CREDENTIALS_ITSELF:
        assert calls == [], (
            f"{request.node.name} reached tc.validate_credentials without installing "
            "its own mock first -- in production this is a live HTTPS request to "
            "selectapi.datascope.refinitiv.com")

# ---------------------------------------------------------------------------
# `_capability_report`'s core state space: shard readiness drives the ASYNC-only vs
# SYNC-and-ASYNC verdict text, a shard warning surfaces verbatim, and the LSEG line
# distinguishes not-yet-checked from checked-and-valid.
# ---------------------------------------------------------------------------

def test_report_says_async_when_no_shards():
    """Assert on whole clauses: "SYNC" is a substring of "ASYNC"."""
    out = cli._capability_report(Capability(False, None), creds=False, checked=False)
    assert "can build the ASYNC datasets" in out
    assert "Sync datasets need LSEG tick data" in out
    assert "globalmacro run --with-download" in out


def test_report_says_both_when_shards_ready():
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
    fake_wrds.Connection = _FakeConnection
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
