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
_ROWS = [
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 3, 20), "volume": 100, "futcode": 555, "label": "A"},
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 3, 20), "volume": 100, "futcode": 222, "label": "B"},
    {"clscode": 1, "date_": date(2020, 3, 2), "lasttrddate": date(2020, 6, 19), "volume": 50, "futcode": 999, "label": "C"},
]


def _frame(order: list[int]) -> pl.DataFrame:
    rows = [_ROWS[i] for i in order]
    return pl.DataFrame({
        "clscode": [r["clscode"] for r in rows],
        "date_": [r["date_"] for r in rows],
        "lasttrddate": [r["lasttrddate"] for r in rows],
        "volume": [r["volume"] for r in rows],
        "futcode": [r["futcode"] for r in rows],
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
        "label": ["low_vol_low_futcode", "high_vol_high_futcode"],
    })
    got = select_top_contracts_by_volume(df).get_column("label").to_list()
    assert got == ["high_vol_high_futcode"]
