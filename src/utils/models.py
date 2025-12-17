from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import time
from enum import Enum


class AssetClass(Enum):
    """Enumeration of asset classes for futures contracts."""
    COMMODITY = "commodity"
    CURRENCY = "currency"
    BOND = "bond"
    EQUITY = "equity"
    VOLATILITY = "volatility"
    SHORT_TERM_INTEREST_RATE = "stir"
    SECTOR = "sector"
    HOUSING = "housing"
    CRYPTOCURRENCY = "cryptocurrency"
    TRADITIONAL = "traditional"
    US_EQUITY = "us_equity"
    NONUS_EQUITY = "nonus_equity"
    HISTORICAL = "historical"

@dataclass
class Future:
    """Data class representing a futures contract configuration."""
    # Datastream data
    symbol: str
    contrcode: int
    exchange: int
    exchange_name: str
    clscode: int
    calcseriesname: str
    name: str
    asset_class: List[AssetClass]
    ct: Optional[List[int]] = None  # Allowed contract expiry months
    historical: Optional[bool] = None # T if the contract is no longer traded but useful for longer time series
    # CFTC data
    cftc_code: Optional[str] = None
    cftc_names: Optional[List[str]] = None
    # Time-synced data
    ric: Optional[List[str]] = None
    settlement_start: Optional[time] = None
    settlement_end: Optional[time] = None
    round: Optional[float] = None
    adjustments: Optional[List[Dict[str, Any]]] = None  # Historical price adjustments
    
    @classmethod
    def from_dict(cls, symbol: str, data: dict) -> 'Future':
        """Create a Future instance from a symbol and data dictionary."""
        asset_classes = data.get('asset_class', 'unknown')
        if isinstance(asset_classes, str):
            asset_classes = [asset_classes]
        asset_classes = [AssetClass(asset_class) for asset_class in asset_classes]

        ric = data.get('ric')
        if isinstance(ric, str):
            ric = [ric]

        return cls(
            # Required fields for non-time-synced data
            symbol=symbol,
            contrcode=data['contrcode'],
            exchange=data['exchange'],
            exchange_name=data['exchange_name'],
            clscode=data['clscode'],
            calcseriesname=data['calcseriesname'],
            name=data['name'],
            asset_class=asset_classes,
            ct=data.get('ct'),
            # CFTC data
            cftc_code=data.get('cftc_code'),
            cftc_names=data.get('cftc_names'),
            # Time-synced data
            ric=ric,
            settlement_start=parse_time(data.get('settlement_start')),
            settlement_end=parse_time(data.get('settlement_end')),
            round=data.get('round'),
            adjustments=data.get('adjustments'),
        )

def parse_time(time_str: Optional[str]) -> Optional[time]:
    """Parse time string in format 'HH:MM' or 'HH:MM:SS' to time object."""
    if time_str is None:
        return None
    parts = time_str.split(':')
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) > 2 else 0
    return time(hour, minute, second)
