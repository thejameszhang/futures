"""Tests for `globalmacro connect`'s capability report (Task 8).

Hermetic by construction: every test that reaches `cli.main(["connect", ...])`
fakes out `wrds_credentials.get_wrds_credentials` and the `wrds` module itself, so
none of them touch a real WRDS connection, a real ~/.pgpass, or the system keyring.
None of them read/write DATASETS_ROOT, VALIDATION_OUTPUT, or TICKHISTORY_PATH --
`capabilities.shards_ready` is always monkeypatched to a fixed `Capability`, never
called for real. `validate_credentials` is never allowed to run un-mocked, so no
test makes a live network call. DSS_USERNAME/DSS_PASSWORD are always set via
monkeypatch.setenv/delenv, never inherited from a real .env.
"""
import sys
import types

import globalmacro.tickhistory_credentials as tc
import globalmacro.utils.capabilities as capmod
import globalmacro.wrds_credentials as wc
from globalmacro import cli
from globalmacro.utils.capabilities import Capability

# ---------------------------------------------------------------------------
# The 4 tests specified verbatim by the Task 8 brief.
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
    assert "valid" in cli._capability_report(
        Capability(False, None), creds=True, checked=True)


# ---------------------------------------------------------------------------
# W2: the LSEG identity line (deviation beyond the brief -- see report).
# ---------------------------------------------------------------------------

def test_identity_line_absent_without_credentials():
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

def test_exit_code_is_zero_on_wrds_success_regardless_of_lseg(monkeypatch):
    _fake_wrds_success(monkeypatch)
    monkeypatch.delenv(tc.ENV_USERNAME, raising=False)
    monkeypatch.delenv(tc.ENV_PASSWORD, raising=False)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    assert cli.main(["connect"]) == 0


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
    _fake_wrds_success(monkeypatch)
    monkeypatch.setenv(tc.ENV_USERNAME, "researcher123")
    monkeypatch.setenv(tc.ENV_PASSWORD, "irrelevant")

    def _network_call():
        raise AssertionError("validate_credentials must not run without --check-lseg")

    monkeypatch.setattr(tc, "validate_credentials", _network_call)
    monkeypatch.setattr(capmod, "shards_ready", lambda: Capability(False, None))
    assert cli.main(["connect"]) == 0


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
