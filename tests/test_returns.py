import polars as pl
from datetime import date
from globalmacro.utils.characteristics import calc_returns_until_expiry


def test_returns_until_expiry_no_roll_is_price_ratio():
    df = pl.DataFrame({
        "clscode": [1, 1, 1],
        "daystomaturity_1": [5, 4, 3],              # shift -> [null, 5, 4], never 0
        "lasttrddate_1": [date(2020, 3, 20)] * 3,
        "lasttrddate_2": [date(2020, 6, 19)] * 3,   # shift != lasttrddate_1 -> no roll
        "settlement_1": [100.0, 110.0, 121.0],
        "settlement_2": [200.0, 200.0, 200.0],      # unused in no-roll branch, must exist
    })
    # Post-currency-switch (Task 0.1), calc_returns_until_expiry returns a single
    # local expr reading settlement_{i} (no `_local` sibling), aliased ret_temp_{i}.
    out = df.with_columns(calc_returns_until_expiry(1, "settlement"))
    ret = out["ret_temp_1"].to_list()
    assert ret[0] is None
    assert abs(ret[1] - 0.10) < 1e-12   # 110/100 - 1
    assert abs(ret[2] - 0.10) < 1e-12   # 121/110 - 1
