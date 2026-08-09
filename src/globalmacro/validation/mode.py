# src/globalmacro/validation/mode.py
"""The resolved validate mode ('full' or 'async-only'), readable by check callables.

run.py resolves the mode once (capabilities.resolve_mode, honouring an explicit
--async-only over whatever the sync panels' own readiness says) and passes it into
_available_checks(mode) and _symbol_count_sources(mode) as an ordinary argument. The
per-check Invariant/figure callables cannot take that argument -- base.Check declares
them as Callable[[], ...], and widening that protocol to thread a mode parameter through
every check in the registry (most of which don't need it) is more invasive than the
problem calls for. This module is the shim: run.py sets the mode for the duration of one
validate run, and a check's _invariants()/_figures() reads it back with current_mode().

Scoped, not sticky: validation_mode() is a context manager that restores the previous
value on the way out -- including on an exception -- so a run never leaves state for the
next one (in the same process) to inherit. See test_validate_mode.py's two-modes-in-one-
process test.

Default is "full": a caller that reaches a check's _invariants()/_figures() without going
through run.py's `with validation_mode(mode):` block -- a unit test invoking the function
directly, e.g. -- gets the pre-Task-6 behaviour, where the sync/async split was governed
by sync_panels_ready() alone.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_mode = "full"


def current_mode() -> str:
    return _mode


@contextmanager
def validation_mode(mode: str) -> Iterator[None]:
    global _mode
    previous = _mode
    _mode = mode
    try:
        yield
    finally:
        _mode = previous
