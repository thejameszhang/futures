"""select_top_contracts_by_volume (futures.py) must be deterministic: the same input rows,
fed in a different order, must produce the same selected row for every (clscode, date_,
order) slot. Before the futcode secondary key, `.sort_by("volume").first()` had no
tie-break when two contracts in the same slot reported the same volume, so which row won
depended on the incoming DataFrame's row order.
"""
import itertools
from datetime import date

import polars as pl

from globalmacro.pipeline.futures import select_top_contracts_by_volume

# Three source rows for clscode 1, date_ 2020-03-02:
#   A and B share lasttrddate (2020-03-20, so both rank "order" 1) AND share volume (100) --
#   a genuine tie with no clue but futcode to break it. C is a distinct, unambiguous slot
#   (lasttrddate 2020-06-19 -> order 2) included so the fixture also proves the tie-break
#   doesn't disturb unrelated groups.
#
# `contrcode` is a real passthrough column (present on every production row via the
# dsfutcontr join) carried here specifically so a mutant that swaps the secondary sort key
# to some other real column can't hide behind a ColumnNotFoundError: A's contrcode (100) is
# LOWER than B's (900) -- the opposite ordering from their futcodes -- so a mutant that
# sorts by contrcode instead of futcode picks A, disagreeing with the futcode-correct
# answer (B), and fails on behaviour rather than a missing column.
_ROWS = [
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 3, 20), "volume": 100, "futcode": 555, "contrcode": 100, "label": "A"},
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 3, 20), "volume": 100, "futcode": 222, "contrcode": 900, "label": "B"},
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 6, 19), "volume": 50, "futcode": 999, "contrcode": 700, "label": "C"},
]


def _frame(order: list[int]) -> pl.DataFrame:
    rows = [_ROWS[i] for i in order]
    return pl.DataFrame({
        "clscode": [r["clscode"] for r in rows],
        "date_": [r["date_"] for r in rows],
        "lasttrddate": [r["lasttrddate"] for r in rows],
        "volume": [r["volume"] for r in rows],
        "futcode": [r["futcode"] for r in rows],
        "contrcode": [r["contrcode"] for r in rows],
        "label": [r["label"] for r in rows],
    })


def test_a_volume_tie_resolves_identically_regardless_of_input_row_order():
    """The property that matters: not "the code runs", but "row order cannot change the
    answer". A and B are tied on volume in every permutation; if the tie-break is missing
    or order-sensitive, different permutations pick different winners for the order=1 slot.
    Exhaustive over all 3! = 6 orderings of the 3-row fixture.
    """
    results = set()
    for perm in itertools.permutations(range(3)):
        got = (
            select_top_contracts_by_volume(_frame(list(perm)))
            .sort(["clscode", "date_", "order"])
            .get_column("label")
            .to_list()
        )
        results.add(tuple(got))
    assert len(results) == 1, f"selection depends on input row order: saw {results}"


def test_the_tie_break_prefers_the_lower_futcode():
    """Pins down WHICH row the tie-break picks (not just that it's consistent), so a mutant
    that flips the futcode sort direction is caught even though it would still pass the
    order-invariance test above (a consistently-wrong answer is still consistent)."""
    got = (
        select_top_contracts_by_volume(_frame([0, 1, 2]))
        .sort(["clscode", "date_", "order"])
        .get_column("label")
        .to_list()
    )
    assert got == ["B", "C"]  # B (futcode 222) beats A (futcode 555) on the volume=100 tie


def test_an_unambiguous_group_is_unaffected():
    """Sanity check: when volumes are NOT tied, the higher-volume row wins regardless of
    futcode -- the secondary key only ever breaks a tie, never overrides volume."""
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date_": [date(2020, 3, 2), date(2020, 3, 2)],
        "lasttrddate": [date(2020, 3, 20), date(2020, 3, 20)],
        "volume": [10, 100],
        "futcode": [999, 111],  # lowest futcode has the LOWER volume
        "contrcode": [200, 800],
        "label": ["low_vol_low_futcode", "high_vol_high_futcode"],
    })
    got = select_top_contracts_by_volume(df).get_column("label").to_list()
    assert got == ["high_vol_high_futcode"]


def test_only_the_top_five_expiries_survive():
    """The slot cap (`order <= 5`) moved inside the extracted function; nothing covered it.
    Six distinct, unambiguous expiries (no ties, so the futcode key never fires) -> orders
    1 through 6, and the sixth must be dropped."""
    n = 6
    df = pl.DataFrame({
        "clscode": [1] * n,
        "date_": [date(2020, 3, 2)] * n,
        "lasttrddate": [date(2020, m, 15) for m in range(1, n + 1)],
        "volume": [10 + i for i in range(n)],
        "futcode": [100 + i for i in range(n)],
        "contrcode": [200 + i for i in range(n)],
        "label": [f"L{i}" for i in range(1, n + 1)],
    })
    got = select_top_contracts_by_volume(df).sort("order")
    assert got.get_column("order").to_list() == [1, 2, 3, 4, 5]
    assert got.get_column("label").to_list() == ["L1", "L2", "L3", "L4", "L5"]


def test_a_null_volume_row_loses_to_any_non_null_volume_row_in_the_same_slot():
    """Latent path: today's production data has 0 slots mixing a null-volume row with a
    non-null-volume one (the 885,073 all-null slots are all single-row), so no existing
    fixture exercised `nulls_last` at all. One slot, two rows sharing lasttrddate (both
    "order" 1): a null-volume row and a non-null-volume row. `nulls_last=True` must place
    the null-volume row after the non-null one under the descending volume sort, so the
    non-null row wins regardless of futcode."""
    df = pl.DataFrame({
        "clscode": [1, 1],
        "date_": [date(2020, 4, 1), date(2020, 4, 1)],
        "lasttrddate": [date(2020, 4, 15), date(2020, 4, 15)],
        "volume": [None, 20],
        "futcode": [999, 111],  # null-volume row has the LOWER futcode
        "contrcode": [999, 111],
        "label": ["null_volume", "has_volume"],
    })
    got = select_top_contracts_by_volume(df).get_column("label").to_list()
    assert got == ["has_volume"]
