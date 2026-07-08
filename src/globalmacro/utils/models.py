from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import time
from enum import Enum
import polars as pl

FuturesPanel = pl.DataFrame
CharacteristicPanel = pl.DataFrame


class AssetClass(Enum):
    """Enumeration of asset classes for futures contracts."""
    COMMODITY = "commodity"
    CURRENCY = "currency"
    BOND = "bond"
    EQUITY = "equity"
    VOLATILITY = "volatility"
    STIR = "stir"
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
    # Datastream Futures data
    symbol: str
    contrcode: int
    exchange: int
    exchange_name: str
    clscode: int
    calcseriesname: str
    name: str
    asset_class: List[AssetClass]
    curcdd: str
    # Datastream Commodities data
    comcode: Optional[List[int]] = None
    ct: Optional[List[int]] = None  # Allowed contract expiry months
    historical: Optional[bool] = None # T if the contract is no longer traded but useful for longer time series
    # Datastream Equities data
    dsindexcode: Optional[List[int]] = None
    dsindexmnem: Optional[List[str]] = None
    # Datastream FX data
    exrateintcode: Optional[int] = None
    fwd_exrateintcode: Optional[int] = None
    inverted_pair: Optional[bool] = False
    # Datastream Economics data
    ecoseriesid: Optional[int] = None
    dsnumber: Optional[str] = None
    dsmnemonic: Optional[str] = None
    libor: Optional[str] = None
    # CFTC data
    cftc_contract_market_codes: Optional[List[str]] = None
    # LSEG TickHistory data
    ric: Optional[List[str]] = None
    settlement_start: Optional[time] = None
    settlement_end: Optional[time] = None
    round: Optional[float] = None
    adjustments: Optional[List[Dict[str, Any]]] = None  # Historical price adjustments
    is_us_asset: Optional[bool] = False 
    exchange_pmc_name: Optional[str] = None
    
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

        dsindexcode = data.get('dsindexcode')
        if isinstance(dsindexcode, int):
            dsindexcode = [dsindexcode]

        dsindexmnem = data.get('dsindexmnem')
        if isinstance(dsindexmnem, str):
            dsindexmnem = [dsindexmnem]

        comcode = data.get('comcode')
        if isinstance(comcode, (int, float)):
            comcode = [int(comcode)]
        elif isinstance(comcode, list):
            comcode = [int(code) for code in comcode if code is not None]

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
            curcdd=data.get('curcdd'),
            comcode=comcode,
            ct=data.get('ct'),
            historical=AssetClass.HISTORICAL in asset_classes,
            # Datastream Equities data
            dsindexcode=dsindexcode,
            dsindexmnem=dsindexmnem,
            libor=data.get('libor'),
            # Datastream Economics data
            ecoseriesid=data.get('ecoseriesid'),
            dsnumber=data.get('dsnumber'),
            dsmnemonic=data.get('dsmnemonic'),
            # Datastream FX Rates data
            exrateintcode=data.get('exrateintcode'),
            fwd_exrateintcode=data.get('fwd_exrateintcode'),
            inverted_pair=data.get('inverted_pair', False),
            # CFTC data
            cftc_contract_market_codes=_normalize_string_list(
                data.get('cftc_contract_market_codes')
            ),
            # Time-synced data
            ric=ric,
            settlement_start=parse_time(data.get('settlement_start')),
            settlement_end=parse_time(data.get('settlement_end')),
            round=data.get('round'),
            adjustments=data.get('adjustments'),
            exchange_pmc_name=data.get('exchange_pmc_name'),
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


def _normalize_string_list(values: Optional[Any]) -> Optional[List[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [str(value) for value in values if value is not None]
    return [str(values)]
