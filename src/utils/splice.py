# Splice historical futures to actively-traded futures
SPLICING_MAP = {
    # S&P 500
    "ES": "SP",
    # S&P 400 Midcap
    "MD": "EMD_open",
    "EMD": "MD",
    # Nasdaq
    "NQ": "ND",
    # VSTOXX
    "FVS": "FVSX",
    # Dow Jones
    "DJ": "YM_open",
    "YM": "DJ",
    # Russell 2000 is an exception, 4-way stitch; see docs
    "RL": "ER2",
    "TF": "RL",
    "RTY_open": "TF",
    "RTY": "RTY_open",
    # Austrian index,
    "FATX": "ATX",
    # Nikkei 225
    "164120019": "167120018",
    # TOPIX Index
    "160120006": "161060005",
    # MSCI EAFE,
    "EFE": "MFS_open",
    "MFS": "EFE",
    # Unleaded gasoline replaced by RBOB Gasoline in 2006; stitch together
    "RB": "HU",
    # Short term interest rates
    # Eurodollar -> 3-Month SOFR with open prices
    "GE": "SR3_open",
    "SR3": "GE",
    # 3-Month TONA replaced the Euroyen in 2024
    "91": "TIFEY",
    # 3-Month SONIA replaced the 3-Month Short Sterling in 2018
    "SO3": "L",
    # 3-Month SARON replaced the 3-Month Euroswiss
    "SA3": "FES",
    # Euro Schatz replaced the Schatz in 1998
    "FGBS": "SH2Z",
    # Euro Bund replaced the Bund
    "FGBL": "BDL",
    # Euro Bobl replaced the Bobl
    "FGBM": "BDM",
    # CAC 40 Index
    "FCE": "FCH",
    # Swiss CONF
    "CONF": "CON",
    # Swiss Market Index
    "FSMI": "SMI",
    # AEX Index
    "FTI": "EOE",
    # OMXS30 (Sweden) Index
    "FXS30": "OMX",
    # German Mark to the Euro
    "6E": "DM",
    # FIB
    "FIB": "IFX",
    # Gas Oil (IPE) to Gas Oil (ICE)
    "G": "GG",
    # Brent Crude Oil (IPE) to Brent Crude Oil (ICE)
    "BRN": "BR",
    # VIX with open prices
    "VX_open": "VIX_open_cboe",
    "VX": "VX_open",
    # Norwegian Krona with open prices because of missing data
    "NOK": "NOK_open",
    # Swedish Krona with open prices because of missing data
    "SEK": "SEK_open",
    # New Zealand Dollar with open prices because of missing data
    "6N": "6N_open",
    # Gold with open prices because of missing data
    "GC": "GC_open",
    # Lean hogs with open prices because of missing data
    "LE": "LE_open",
    # Feeder Cattle with open prices because of missing data
    "GF": "GF_open",
    # Sugar with open prices because of missing data
    "ZS": "ZS_open",
    # Coffee with open prices because of missing data
    "KC": "KC_open",
    # Soybeans with open prices because of missing data
    "SB": "SB_open",
    # NG with open prices because of missing data
    "NG": "NG_open",
    # Cocoa with open prices because of missing data
    "CC": "CC_open",
    # Cotton with open prices because of missing data
    "CT": "CT_open",
    # 2 year US Treasury with open prices because of missing data
    "ZT": "ZT_open",
    # Light Crude Oil with open prices because of missing data
    "CL": "CL_open",
    # LME Aluminium with settlement prices (1-2 hour before 9:30am EST)
    "AHD": "AHD_lme",
    # LME Copper with settlement prices (1-2 hour before 9:30am EST)
    "CAD": "CAD_lme",
    # LME Nickel with settlement prices (1-2 hour before 9:30am EST)
    "NID": "NID_lme",
    # LME Zinc with settlement prices (1-2 hour before 9:30am EST)
    "ZSD": "ZSD_lme",
    # LME Lead with settlement prices (1-2 hour before 9:30am EST)
    "PBD": "PBD_lme",
}