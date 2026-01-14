import pandas as pd
import os
from utils.config import load_config
from utils.models import AssetClass, Future
from utils.paths import DATA_ROOT, PROJECT_ROOT

futures = load_config(PROJECT_ROOT / "tier1.yaml")

def get_asset_class_rics(futures: list[Future], assetclass: AssetClass) -> tuple[list[str], int]:
    rics = []
    for future in futures:
        if future.ric and future.asset_class[0] == assetclass:
            for ric in future.ric:
                for i in range(1, 7):
                    rics.append(f"{ric}c{i}")
        elif future.ric and AssetClass.TRADITIONAL in future.asset_class:
            for ric in future.ric:
                for i in range(1, 13):
                    rics.append(f"{ric}c{i}")
    rics = sorted(rics)
    df = pd.DataFrame([["RIC"] * len(rics), rics]).T
    df = df.rename(columns={0: "RIC", 1: df.iloc[0][1]})
    df = df.drop(index=0).reset_index(drop=True)
    os.makedirs(DATA_ROOT / "tickhistory" / "instrumentlists", exist_ok=True)
    df.to_csv(DATA_ROOT / "tickhistory" / "instrumentlists" / f"{assetclass.value}.csv", index=False)
    return df

assetclasses = [
    AssetClass.COMMODITY, 
    AssetClass.CURRENCY, 
    AssetClass.BOND, 
    AssetClass.EQUITY, 
    AssetClass.VOLATILITY,
    AssetClass.STIR,
    AssetClass.TRADITIONAL,
]

for assetclass in assetclasses:
    get_asset_class_rics(futures, assetclass)