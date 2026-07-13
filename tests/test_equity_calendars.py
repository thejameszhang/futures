"""Every equity index must be filtered by the exchange where its CONSTITUENTS trade.

equities.py uses `exchange_pmc_name` to decide which days the CASH index traded. Get it
wrong and days the market was shut survive the filter; Datastream's padded level makes the
price look unchanged and the return computes to a fabricated 0.0.

TWO ways to get it wrong, and the test must catch BOTH:
  * a FUTURES venue instead of a cash exchange -- ES on CME_Equity, a 23h session.
  * the wrong COUNTRY -- CN (FTSE China A50) on Singapore's calendar. That is an 8h session,
    so a session-length check PASSES it, while 93 of CN's 94 zeros fall on days SHANGHAI was
    closed.

Hence an explicit venue map covering ALL 37 indices. Each entry is one lookup to verify.
"""
import pandas_market_calendars as pmc
import pytest

from globalmacro.utils.config import load_config
from globalmacro.utils.paths import PROJECT_ROOT

EXPECTED_VENUE = {
    "160120006": "JPX",      # Mini TOPIX
    "164120019": "JPX",      # Mini Nikkei 225
    "A01":       "XKRX",     # Kospi 200
    "AP":        "ASX",      # SPI 200
    "BXF":       "XBRU",     # BEL20
    "CN":        "XSHG",     # FTSE China A50   -- REPOINTED: Shanghai, NOT Singapore
    "DTOP":      "XJSE",     # South African All-Share
    "EMD":       "NYSE",     # S&P 400 Midcap   -- REPOINTED: was CME_Equity (23h futures venue)
    "ES":        "NYSE",     # S&P 500          -- REPOINTED: was CME_Equity
    "FATX":      "XWBO",     # ATX
    "FCE":       "XPAR",     # CAC 40
    "FDAX":      "XETR",     # DAX
    "FESX":      "EUREX",    # Euro Stoxx 50 -- pan-Eurozone; EUREX is the right proxy
    "FFOX":      "XHEL",     # OMXH25 Finland   -- REPOINTED: Helsinki, NOT Eurex
    "FIB":       "XMIL",     # FTSE MIB
    "FIE":       "XMAD",     # IBEX 35
    "FKLI":      "XKLS",     # FTSE Bursa Malaysia KLCI
    "FSMI":      "XSWX",     # Swiss Market Index
    "FTI":       "XAMS",     # AEX
    "FW20":      "XWAR",     # WIG20
    "FXC25":     "XCSE",     # OMXC20 Denmark
    "FXS30":     "XSTO",     # OMXS30 Sweden
    "HHI":       "XHKG",     # FTSE China H-Shares -- H-shares LIST in Hong Kong
    "HSI":       "XHKG",     # Hang Seng
    "IND":       "BVMF",     # Bovespa
    "IPC":       "XMEX",     # IPC Mexico
    "MXF":       "XTAI",     # TAIEX
    "NIFTY":     "XNSE",     # NIFTY 50
    "NQ":        "NYSE",     # Nasdaq 100 -- REPOINTED. pmc's "NASDAQ" IS the NYSE calendar
                             #   (19,208 sessions each, symmetric difference 0), so name it
                             #   NYSE rather than imply a distinction that does not exist.
    "OBF":       "XOSL",     # OBX Norway
    "PSI":       "XLIS",     # PSI 20 Portugal
    "RTY":       "NYSE",     # Russell 2000     -- REPOINTED: was CME_Equity
    "SET50":     "XBKK",     # SET 50 Thailand
    "SGP":       "XSES",     # MSCI Singapore   -- correctly Singapore
    "SXF":       "TSX",      # S&P/TSX 60
    "YM":        "NYSE",     # Dow Jones        -- REPOINTED: was CBOT_Equity
    "Z":         "LSE",      # FTSE 100
}

MAX_CASH_SESSION_HOURS = 12.0   # cash sessions run 4.5-8.5h; futures venues ~23h


def _equity_indices():
    futures = load_config(PROJECT_ROOT / "tier1.yaml") + load_config(PROJECT_ROOT / "tier2.yaml")
    return sorted({
        (str(f.symbol), f.exchange_pmc_name)
        for f in futures
        if f.dsindexcode is not None and f.exchange_pmc_name
    })


def test_every_equity_index_has_a_pinned_venue():
    # Without this, an index absent from the map is silently unchecked -- and the session-length
    # test CANNOT backstop it: EUREX models an 8.5h session, so repointing FDAX from XETR to
    # EUREX (a derivatives venue) would pass both tests and go undetected.
    missing = sorted({s for s, _ in _equity_indices()} - set(EXPECTED_VENUE))
    assert not missing, f"no expected venue pinned for: {missing}"


def test_venue_map_has_no_orphans():
    orphans = sorted(set(EXPECTED_VENUE) - {s for s, _ in _equity_indices()})
    assert not orphans, f"EXPECTED_VENUE names symbols that no longer exist: {orphans}"


@pytest.mark.parametrize("symbol,calendar", _equity_indices())
def test_index_uses_its_constituents_exchange(symbol, calendar):
    expected = EXPECTED_VENUE[symbol]
    assert calendar == expected, (
        f"{symbol} is filtered by {calendar}, but its constituents trade on {expected}. "
        f"Days its market was shut survive the filter and become fabricated 0.0 returns."
    )


@pytest.mark.parametrize("symbol,calendar", _equity_indices())
def test_index_is_not_filtered_by_a_futures_venue(symbol, calendar):
    cal = pmc.get_calendar(calendar)
    open_h = cal.open_time.hour + cal.open_time.minute / 60
    close_h = cal.close_time.hour + cal.close_time.minute / 60
    session_hours = (close_h - open_h) % 24
    assert session_hours <= MAX_CASH_SESSION_HOURS, (
        f"{symbol} is filtered by {calendar}, a {session_hours:.1f}h session -- a FUTURES "
        f"venue, not a cash exchange."
    )
