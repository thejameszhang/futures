# Splice historical futures to actively-traded futures
SPLICING_MAP = {
    # S&P 500
    "ES": "SP",
    # S&P 400 Midcap
    "EMD": "MD",
    # Nasdaq
    "NQ": "ND",
    # VSTOXX
    "FVS": "FVSX",
    # Dow Jones
    "YM": "DJ",
    # Russell 2000 is an exception, 4-way stitch; see docs
    "RL": "ER2",
    "TF": "RL",
    "RTY": "TF",
    # Austrian index,
    "FATX": "ATX",
    # Nikkei 225
    "164120019": "167120018",
    # TOPIX Index
    "160120006": "161060005",
    # MSCI EAFE,
    "MFS": "EFE",
    # Unleaded gasoline replaced by RBOB Gasoline in 2006; stitch together
    "RB": "HU",
    # Gas Oil (IPE) to Gas Oil (ICE)
    "G": "GG",
    # Brent Crude Oil (IPE) to Brent Crude Oil (ICE)
    "BRN": "BR",
    # Short term interest rates
    # Eurodollar -> 3-Month SOFR
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

    # Tier 2 now
    # OMXC20 Denmark Index
    "C20CAP": "OMXC20",
    "FXC25": "C20CAP",
}
