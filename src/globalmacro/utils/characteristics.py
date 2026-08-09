import polars as pl


def calc_price_with_roll(i: int, price_type: str) -> list[pl.Expr]:
    """
    Calculate price of the futures contract
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the price of the i-th month contract
    """
    return ([
        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'{price_type}_{i + 1}'))
        .otherwise(pl.col(f'{price_type}_{i}'))
        .over(pl.col('clscode'))
        .alias(f'adj_{price_type}_{i}'),
    ])

def calc_returns_until_expiry(i: int, price_type: str) -> list[pl.Expr]:
    """
    Helper function for calculating returns of futures contracts with and without price adjustment
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the returns of the i-th month contract until expiry date of each contract
    """
    # True when the front dropped out and everything shifted up a slot, so today's slot i is
    # yesterday's slot i+1 -- the SAME contract, and the ratio below is correct.
    roll = (pl.col('daystomaturity_1').shift(1) == 0) | (pl.col('lasttrddate_1') == pl.col('lasttrddate_2').shift(1))
    # Which contract the denominator price actually belongs to, given that choice.
    denominator = pl.when(roll).then(pl.col(f'lasttrddate_{i + 1}').shift(1)).otherwise(pl.col(f'lasttrddate_{i}').shift(1))
    # futures.py ranks contracts over the rows that EXIST for a date, and a contract with no
    # price has no row -- so on an exchange holiday a far-dated contract can occupy slot 1
    # (NG 2012-09-03 +62%, C 2016-12-28 +5038%). `roll` already rescues the common one-slot
    # shift; this guard covers the residual. A ratio between two different contracts is not a
    # return, so emit null rather than a number. 1,074 cells across 105 classes.
    same_contract = pl.col(f'lasttrddate_{i}') == denominator
    return ([
        pl.when(same_contract)
        .then(
            pl.when(roll)
            .then((pl.col(f'{price_type}_{i}') / pl.col(f'{price_type}_{i + 1}').shift(1)) - 1)
            .otherwise((pl.col(f'{price_type}_{i}') / pl.col(f'{price_type}_{i}').shift(1)) - 1)
        )
        .otherwise(None)
        .over('clscode').alias(f'ret_temp_{i}'),
    ])

def contract_identity_exprs(i: int) -> tuple[pl.Expr, pl.Expr]:
    """(numerator contract, denominator contract) for the finalised ret_i.

    Single source of truth: futures.py's post-coalesce guard and cross_contract_cells both
    use this, so the fix and its verification cannot measure different things.

    ret_i is ret_temp_{i+1} when exp_1 == 1, else ret_temp_i
    (calc_returns_with_price_adj_and_roll), so the numerator slot shifts with exp_1.
    """
    roll = (pl.col('daystomaturity_1').shift(1).over('clscode') == 0) | (
        pl.col('lasttrddate_1') == pl.col('lasttrddate_2').shift(1).over('clscode'))
    num = pl.when(pl.col('exp_1') == 1).then(pl.col(f'lasttrddate_{i + 1}')).otherwise(pl.col(f'lasttrddate_{i}'))
    den = (pl.when(pl.col('exp_1') == 1)
           .then(pl.when(roll).then(pl.col(f'lasttrddate_{i + 2}').shift(1).over('clscode'))
                 .otherwise(pl.col(f'lasttrddate_{i + 1}').shift(1).over('clscode')))
           .otherwise(pl.when(roll).then(pl.col(f'lasttrddate_{i + 1}').shift(1).over('clscode'))
                      .otherwise(pl.col(f'lasttrddate_{i}').shift(1).over('clscode'))))
    return num, den

def calc_returns_with_price_adj_and_roll(i: int) -> list[pl.Expr]:
    """
    Calculate the returns of the futures contracts with price adjustment and roll
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the returns of the i-th month contract with price adjustment and roll
    """
    return ([
        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'ret_temp_{i + 1}'))
        .otherwise(pl.col(f'ret_temp_{i}'))
        .over(pl.col('clscode'))
        .alias(f'ret_{i}'),
    ])

# Edge case: front month contract rolls and expires on the day before
def calc_total_returns_with_roll(i: int, price_type: str) -> list[pl.Expr]:
    """
    Calculate the returns of the futures contracts without price adjustment and roll
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the returns of the i-th month contract without price adjustment and roll
    """
    return ([
        pl.when(((pl.col('exp_1') == 1) & (pl.col('switch_1') == 1)) & (pl.col('daystomaturity_1').shift(1) != 0))
        .then((pl.col(f'{price_type}_{i + 1}').shift(1) / pl.col(f'{price_type}_{i}').shift(1)) * (pl.col(f'ret_temp_{i + 1}') + 1) - 1)
        .otherwise(pl.col(f'ret_{i}'))
        .over(pl.col('clscode'))
        .alias(f'ret_total_{i}'),
    ])

def calc_spot_price_until_expiry(i: int, price_type: str) -> list[pl.Expr]:
    """
    Helper function for calculating the futures-implied spot prices. Mostly relevant for commodities.
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied spot price until expiry of the i-th month contract
    """
    return ([
        (pl.col(f'{price_type}_{i}') * (1 + (pl.col(f'basis_{i}') / 365) * pl.col(f'daystomaturity_{i}')))
        .over(pl.col('clscode'))
        .alias(f'spot_price_{i}'),

        (pl.col(f'{price_type}_local_{i}') * (1 + (pl.col(f'basis_local_{i}') / 365) * pl.col(f'daystomaturity_{i}')))
        .over(pl.col('clscode'))
        .alias(f'spot_price_local_{i}'),
    ])

def calc_spot_price_with_roll(i: int) -> list[pl.Expr]:
    """
    Calculate the the futures-implied spot price with roll
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied spot price with roll of the i-th month contract
    """
    return ([
        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'spot_price_{i + 1}'))
        .otherwise(pl.col(f'spot_price_{i}'))
        .over(pl.col('clscode'))
        .alias(f'adj_spot_price_{i}'),

        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'spot_price_local_{i + 1}'))
        .otherwise(pl.col(f'spot_price_local_{i}'))
        .over(pl.col('clscode'))
        .alias(f'adj_spot_price_local_{i}'),
    ])

# Calculate the the futures-implied spot return with price adjustment and roll
def calc_spot_returns_until_expiry(i: int) -> list[pl.Expr]:
    """
    Helper function for calculating the futures-implied spot returns. Mostly relevant for commodities.
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied spot return until expiry of the i-th month contract
    """
    return ([
        pl.when(pl.col('daystomaturity_1').shift(1) == 0)\
        .then((pl.col(f'spot_price_{i}') / pl.col(f'spot_price_{i + 1}').shift(1)) - 1)
        .otherwise((pl.col(f'spot_price_{i}') / pl.col(f'spot_price_{i}').shift(1)) - 1)
        .over('clscode')
        .alias(f'spot_ret_temp_{i}'),

        pl.when(pl.col('daystomaturity_1').shift(1) == 0)
        .then((pl.col(f'spot_price_local_{i}') / pl.col(f'spot_price_local_{i + 1}').shift(1)) - 1)
        .otherwise((pl.col(f'spot_price_local_{i}') / pl.col(f'spot_price_local_{i}').shift(1)) - 1)
        .over('clscode')
        .alias(f'spot_ret_temp_local_{i}'),
    ])

def calc_spot_return_with_price_adj_and_roll(i: int) -> list[pl.Expr]:
    """
    Calculate the the futures-implied spot return with price adjustment and roll
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied spot return with price adjustment and roll of the i-th month contract
    """
    return ([
        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'spot_ret_temp_{i + 1}'))
        .otherwise(pl.col(f'spot_ret_temp_{i}'))
        .over(pl.col('clscode'))
        .alias(f'spot_ret_{i}'),

        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'spot_ret_temp_local_{i + 1}'))
        .otherwise(pl.col(f'spot_ret_temp_local_{i}'))
        .over(pl.col('clscode'))
        .alias(f'spot_ret_local_{i}'),
    ])

# Calculate the the futures-implied basis until expiry
def calc_basis_until_expiry(i: int, price_type: str) -> list[pl.Expr]:
    """
    Helper function for calculating the futures-implied basis. Mostly relevant for commodities.
    See Gorton and Rouwenhorst (2012) Page 83.
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied basis until expiry of the i-th month contract
    """
    return ([
        (365 * (pl.col(f'{price_type}_{i}') / pl.col(f'{price_type}_{i + 1}') - 1) / (pl.col(f'daystomaturity_{i + 1}') - pl.col(f'daystomaturity_{i}')))
        .over(pl.col('clscode'))
        .alias(f'basis_{i}'),

        (365 * (pl.col(f'{price_type}_local_{i}') / pl.col(f'{price_type}_local_{i + 1}') - 1) / (pl.col(f'daystomaturity_{i + 1}') - pl.col(f'daystomaturity_{i}')))
        .over(pl.col('clscode'))
        .alias(f'basis_local_{i}'),
    ])

def calc_basis_with_roll(i: int) -> list[pl.Expr]:
    """
    Calculate the the futures-implied basis with roll
    Args:
        i: int - the index of the month contract
    Returns:
        pl.Expr - the futures-implied basis with roll of the i-th month contract
    """
    return ([
        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'basis_{i + 1}'))
        .otherwise(pl.col(f'basis_{i}'))
        .over(pl.col('clscode'))
        .alias(f'adj_basis_{i}'),

        pl.when(pl.col('exp_1') == 1)
        .then(pl.col(f'basis_local_{i + 1}'))
        .otherwise(pl.col(f'basis_local_{i}'))
        .over(pl.col('clscode'))
        .alias(f'adj_basis_local_{i}'),
    ])
