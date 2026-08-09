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

Validated, not just typed: validation_mode() rejects anything outside {"full",
"async-only"} -- the same defect class build._validate_mode was written to close (a
typo'd mode silently takes the `mode == "full"` == False branch and reports success on
a truncated deliverable). run.py's only caller passes resolve_mode()'s output, which is
already constrained to the two literals, so this is currently a belt-and-braces check on
today's one call site -- exactly the position build.main()'s guard was in when it was
added.

Process-wide, not context-local: `_mode` is a plain module global rather than a
`contextvars.ContextVar`, so a worker thread started inside `with validation_mode(...):`
observes that mode too, not just the thread that entered the block. Harmless today --
`main()` runs single-threaded end to end and nothing here spawns threads -- but a
deliberate choice, not an oversight, should this module ever grow concurrent callers.
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

_VALID_MODES = ("full", "async-only")

_mode = "full"


def current_mode() -> str:
    return _mode


@contextmanager
def validation_mode(mode: str) -> Iterator[None]:
    global _mode
    if mode not in _VALID_MODES:
        raise ValueError(
            f"globalmacro validate: unrecognized mode {mode!r}; "
            'expected "full" or "async-only"'
        )
    previous = _mode
    _mode = mode
    try:
        yield
    finally:
        _mode = previous
