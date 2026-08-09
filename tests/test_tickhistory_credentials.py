import pathlib
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


def test_network_guard_actually_replaces_validate_credentials():
    """F3 regression: prove the repo-wide autouse `_no_live_lseg_network` fixture
    (tests/conftest.py) has actually patched `tc.validate_credentials` for this test,
    not merely been defined and left unused. A conftest.py fixture guarding 349 tests
    can be reduced to a no-op (e.g. `calls = []; yield; assert calls == []` with the
    `monkeypatch.setattr` line removed) by a future refactor and CI stays green --
    nothing else in the suite drives a real network call, so nothing else notices.

    The stub is a lambda defined in tests/conftest.py, so its `__module__` is that
    module's name; the real `validate_credentials` lives in
    `globalmacro.tickhistory_credentials`. If the fixture stopped patching the
    attribute, `tc.validate_credentials` would be the real function again and this
    assertion is what would catch it (task-10-report.md records the gutted-fixture
    proof: with the `monkeypatch.setattr` call removed, this assertion fails).
    """
    assert tc.validate_credentials.__module__ != "globalmacro.tickhistory_credentials"


def test_network_guard_stub_records_keyword_argument_calls(pytester):
    """F1 regression: the autouse `_no_live_lseg_network` stub in tests/conftest.py
    installs `lambda *a, **k: calls.append(1) or False` against
    `validate_credentials(timeout: float = 10.0)`'s real signature. The pre-fix
    zero-arg stub (`lambda: calls.append(1) or False`) raises `TypeError` before
    `calls.append(1)` runs on any keyword-argument call; a caller that wraps the call
    in `except Exception` (cli.py:173 does) swallows that `TypeError`, and the
    guard's own teardown assertion then passes -- reporting "never called" for what
    would be a live HTTPS request in production.

    Exercising this in-process would mean deliberately tripping the guard's own
    teardown assertion for THIS test, which would show up as a failure/error in a
    suite this task requires to be green. Instead this runs the real, unmodified
    tests/conftest.py inside a `pytester`-managed subprocess: a throwaway test there
    calls `tc.validate_credentials(timeout=5.0)` directly. Under the fixed stub, that
    call does not raise, so the throwaway test's own body still shows PASSED -- and
    the SAME item's teardown then errors, because the call WAS recorded (`calls ==
    [1]`, not `[]`). One passed test plus one teardown error is the proof that a
    keyword-argument call is still caught, not silently lost -- and it is also a live
    demonstration of F2's corrected report: PASSED in the test body, error only at
    teardown, so only the exit code (not a skim for "PASSED") is honest.

    (Feeding the pre-fix zero-arg stub through this identical scratch run instead
    produces one outright FAILED test: the `TypeError` surfaces directly in the test
    body, because a bare direct call has no cli.py-style `except Exception` around it
    to swallow the exception. That is a different, more visible failure than the
    silent-pass this test guards against -- confirmed manually while building this
    test, not asserted here, since asserting on the pre-fix source string would just
    be testing a string literal rather than the guard's behaviour.)
    """
    real_conftest = (pathlib.Path(__file__).parent / "conftest.py").read_text()
    pytester.makepyfile(conftest=real_conftest)
    pytester.makepyfile(test_kwarg_call="""
        import globalmacro.tickhistory_credentials as tc

        def test_calls_validate_credentials_with_a_keyword_argument():
            tc.validate_credentials(timeout=5.0)
    """)
    result = pytester.runpytest_subprocess()
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*reached tc.validate_credentials*"])
