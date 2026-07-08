import pandas as pd
import os
from .config import load_config
from .models import AssetClass, Future
from .paths import DATA_ROOT, PROJECT_ROOT

def get_asset_class_rics(futures: list[Future], assetclass: AssetClass, tier: int) -> tuple[list[str], int]:
    rics = []
    for future in futures:
        if future.ric and future.asset_class[0] == assetclass:
            for ric in future.ric:
                for i in range(1, 7):
                    rics.append(f"{ric}c{i}")
        elif assetclass == AssetClass.TRADITIONAL and future.ric and AssetClass.TRADITIONAL in future.asset_class:
            for ric in future.ric:
                for i in range(1, 13):
                    rics.append(f"{ric}c{i}")
    rics = sorted(rics)
    out_dir = DATA_ROOT / "tickhistory" / "instrumentlists"
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / f"tier{tier}_{assetclass.value}.csv"
    if not rics:
        # No instruments in this (asset_class, tier) group (e.g. tier-2 crypto/housing
        # are declared in the loop but absent from the config) -> empty pull-list.
        df = pd.DataFrame(columns=["RIC"])
        df.to_csv(out_path, index=False)
        return df
    df = pd.DataFrame([["RIC"] * len(rics), rics]).T
    df = df.rename(columns={0: "RIC", 1: df.iloc[0][1]})
    df = df.drop(index=0).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df

def generate_instrument_lists() -> None:
    """Write per-(tier, asset-class) RIC pull-lists to data/tickhistory/instrumentlists/.
    These specify which RICs to pull from LSEG TickHistory (a prerequisite acquisition step)."""
    tier1_futures = load_config(PROJECT_ROOT / "tier1.yaml")
    tier2_futures = load_config(PROJECT_ROOT / "tier2.yaml")

    assetclasses_tier_tuples = [
        # Tier 1
        (AssetClass.COMMODITY, 1),
        (AssetClass.CURRENCY, 1),
        (AssetClass.BOND, 1),
        (AssetClass.EQUITY, 1),
        (AssetClass.VOLATILITY, 1),
        (AssetClass.STIR, 1),
        (AssetClass.TRADITIONAL, 1),
        # Tier 2
        (AssetClass.CURRENCY, 2),
        (AssetClass.EQUITY, 2),
        (AssetClass.CRYPTOCURRENCY, 2),
        (AssetClass.HOUSING, 2),
    ]

    for assetclass, tier in assetclasses_tier_tuples:
        f = tier1_futures if tier == 1 else tier2_futures
        get_asset_class_rics(f, assetclass, tier)