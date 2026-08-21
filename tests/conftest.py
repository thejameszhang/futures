"""Repo-wide test fixtures.

Two unrelated safety nets live here because both need to see every test
under tests/, not just the file that motivated them:

1. `needs_sync` marker: skips tests that need the sync datasets on disk, so
   the suite is runnable on a tick-less (async-only) machine. See
   `pytest_collection_modifyitems` below.
2. `_no_live_lseg_network`: a safety net against a live network call. It has
   no docstring of its own -- the module-level comment directly above its
   definition carries the prose instead.

A conftest.py fixture is global by construction -- a mistake here has the
largest possible blast radius. Both fixtures below default to the SAFE
behaviour (skip / stubbed-network) and require a test to opt in explicitly
(the `needs_sync` marker, or the `calls_real_validate_credentials` marker) to
get anything else, so a newly written test that does nothing special is
always on the guarded side.
"""
import pytest

import globalmacro.tickhistory_credentials as tc
from globalmacro.utils.capabilities import sync_panels_ready

# The `pytester` fixture is bundled with pytest but not enabled by default; it must be
# requested via `pytest_plugins` in an *initial* conftest.py -- one collected before any
# test file, which this one is (pyproject.toml's testpaths = ["tests"] makes this the
# root of collection). Declaring it in a conftest.py reached only during collection
# (e.g. one nested under a subpackage) is deprecated/rejected instead.
# tests/test_tickhistory_credentials.py uses it for a regression test proving the
# `_no_live_lseg_network` stub below matches validate_credentials' real signature:
# that test must trigger this fixture's own teardown assertion for real to prove a
# keyword-argument call is recorded, which would fail the CURRENT test run if done
# in-process -- `pytester` runs it as an isolated subprocess instead.
pytest_plugins = ["pytester"]

# ---------------------------------------------------------------------------
# needs_sync: skip sync-coupled tests on a machine with no LSEG tick data.
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_sync: requires the sync datasets to exist on disk")
    config.addinivalue_line(
        "markers",
        "calls_real_validate_credentials: exempt from the autouse "
        "_no_live_lseg_network stub -- the test exercises "
        "tc.validate_credentials' own implementation and supplies its own "
        "guarantee that doing so cannot reach the network.")


def pytest_collection_modifyitems(config, items):
    if sync_panels_ready().ready:
        return
    skip = pytest.mark.skip(reason="sync datasets absent (async-only machine)")
    for item in items:
        if item.get_closest_marker("needs_sync"):
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# A safety net against a live network call, applied automatically to every test
# under tests/. `tc.validate_credentials()` makes a real HTTPS request to
# selectapi.datascope.lseg.com, with real DSS credentials in the request body.
#
# Originally this lived only in tests/test_connect_report.py, guarding the
# three `cli.main(["connect", ...])` tests there that rely on `checked and present`
# gating in cli.py to keep `validate_credentials` unreached. That left
# tests/test_tickhistory_credentials.py -- the file whose function actually makes the
# call -- unguarded: a test added there that calls `tc.validate_credentials()` without
# its own stub records a live network attempt and passes silently. Moving the fixture
# here (autouse at the tests/ package level, not just one module) closes that gap for
# every file, present and future.
#
# Exactly one test under tests/ needs the REAL implementation to run --
# test_tickhistory_credentials.py's `test_validate_credentials_never_leaks_...`, which
# proves validate_credentials()'s own try/except starts before request-body
# construction. It is marked `calls_real_validate_credentials` and opts out of the
# stub below; it is still safe because IT patches `urllib.request.Request` to raise
# before `urlopen` is ever reachable, independently of this fixture. Every other test
# in the repo gets `tc.validate_credentials` replaced by a stub that records whether
# it was invoked; the assertion after `yield` then either passes silently (never
# called) or names the offending test.
#
# A raise-based tripwire cannot do this job: cli.py's capability-report block wraps
# everything in `except Exception`, so a stub that raises from inside `cli.main()` is
# silently swallowed and proves nothing. Recording calls and asserting on the list
# after the test returns -- outside any try/except the production code owns -- is
# what actually distinguishes "called" from "not called".
#
# Function-scoped, not pytest's `scope="module"` (the built-in `monkeypatch` fixture
# it depends on is function-scoped, so a literal module-scoped fixture would need a
# raw `pytest.MonkeyPatch()` instead) -- but `autouse=True` gives every test the same
# guarantee, which is the property that matters.
#
# The two `cli.main(["connect", "--check-lseg"])` tests that legitimately drive
# `validate_credentials` end to end do NOT need `calls_real_validate_credentials`:
# each installs its own `monkeypatch.setattr(tc, "validate_credentials", ...)` inside
# the test body, which -- being a later `setattr` on the same attribute in the same
# monkeypatch stack -- silently replaces this fixture's stub before it is ever
# called. This fixture's own recording list is therefore never touched by those two
# tests either way, so a name-based opt-out list for them would be dead code (an
# earlier version of this fixture had exactly that list; it never changed an
# outcome, and removing it left every test passing).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_live_lseg_network(request, monkeypatch):
    if request.node.get_closest_marker("calls_real_validate_credentials"):
        yield
        return
    calls: list[int] = []
    monkeypatch.setattr(tc, "validate_credentials", lambda *a, **k: calls.append(1) or False)
    yield
    assert calls == [], (
        f"{request.node.name} reached tc.validate_credentials without installing "
        "its own mock first -- in production this is a live HTTPS request to "
        "selectapi.datascope.lseg.com")
