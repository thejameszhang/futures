import urllib.request

import pytest

from globalmacro import tickhistory_credentials as tc


def test_absent_when_env_unset(monkeypatch):
    monkeypatch.delenv("DSS_USERNAME", raising=False)
    monkeypatch.delenv("DSS_PASSWORD", raising=False)
    assert tc.credentials_present() is False
    assert tc.credential_username() is None


def test_present_when_both_set(monkeypatch):
    monkeypatch.setenv("DSS_USERNAME", "u")
    monkeypatch.setenv("DSS_PASSWORD", "p")
    assert tc.credentials_present() is True
    assert tc.credential_username() == "u"


def test_absent_when_only_username_set(monkeypatch):
    monkeypatch.setenv("DSS_USERNAME", "u")
    monkeypatch.delenv("DSS_PASSWORD", raising=False)
    assert tc.credentials_present() is False


@pytest.mark.calls_real_validate_credentials
def test_validate_credentials_never_leaks_on_request_construction_failure(monkeypatch, capsys):
    """W1 regression: validate_credentials()'s try must start before body/Request
    construction, not just around urlopen(). If it didn't, an exception raised while
    building the request -- which can carry the plaintext request body, i.e. the DSS
    password -- would propagate out of validate_credentials() uncaught, and a caller
    that prints str(exc) (cli.py's `except Exception as exc: print(f"...: {exc}")`)
    would leak it. Uses a synthetic sentinel, never a real credential value.

    Marked `calls_real_validate_credentials`: this is the one test in the suite that
    must call the REAL `tc.validate_credentials`, not the stub the repo-wide
    `_no_live_lseg_network` fixture (tests/conftest.py) installs everywhere else --
    the whole point is to exercise its actual try/except. It is still safe: patching
    `urllib.request.Request` to raise means `urlopen` is never reached, independent
    of that fixture.
    """
    monkeypatch.setenv("DSS_USERNAME", "u")
    monkeypatch.setenv("DSS_PASSWORD", "p")
    sentinel = "SENTINEL-PLAINTEXT-PAYLOAD-9f3a-not-a-real-secret"

    def _boom(*args, **kwargs):
        raise RuntimeError(f"request build failed, body was: {sentinel}")

    monkeypatch.setattr(urllib.request, "Request", _boom)

    assert tc.validate_credentials() is False

    out = capsys.readouterr()
    assert sentinel not in out.out
    assert sentinel not in out.err
