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
