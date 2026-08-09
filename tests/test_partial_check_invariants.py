from globalmacro.utils.capabilities import Capability
from globalmacro.validation import synthetic_equity, synthetic_fx


def _force_async(monkeypatch):
    absent = Capability(False, None)
    monkeypatch.setattr(synthetic_fx, "sync_panels_ready", lambda: absent)
    monkeypatch.setattr(synthetic_equity, "sync_panels_ready", lambda: absent)


def test_synthetic_equity_returns_no_invariants_in_async_only(monkeypatch):
    _force_async(monkeypatch)
    assert synthetic_equity._invariants() == []


def test_synthetic_fx_keeps_only_the_async_invariant(monkeypatch):
    """Names contain the dataset, and 'async' contains 'sync' -- assert on the source."""
    _force_async(monkeypatch)
    invs = synthetic_fx._invariants()
    assert len(invs) == 1
    assert invs[0].name.startswith("Datastream")


def test_skipping_never_manufactures_a_failing_invariant(monkeypatch):
    """The trap: an empty sync subset makes `total > 0 and ...` False, so a skipped
    invariant would report FAIL rather than being absent. Assert on the SHAPE of the
    list -- not on `.passed`, which would require built datasets to evaluate."""
    _force_async(monkeypatch)
    invs = synthetic_fx._invariants() + synthetic_equity._invariants()
    assert not any(i.value.startswith("0/0") for i in invs)
    # DEVIATION from the brief: its snippet checked `"sync futures" not in i.name`, which
    # is itself the "async contains sync" trap the whole plan warns about -- the surviving,
    # CORRECT invariant is named "...the ASYNC futures", and "async futures" contains the
    # substring "sync futures" (a-SYNC futures). That check would always fail, even on a
    # correct implementation. Match "the sync futures" (with the leading "the ") instead:
    # that phrase is present only in the sync-side name ("...on the sync futures") and
    # absent from the async-side one ("...on the async futures").
    assert all("the sync futures" not in i.name for i in invs)
