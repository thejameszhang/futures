import polars as pl
from globalmacro.utils.config import load_config
from globalmacro.utils.models import AssetClass
from globalmacro.utils.paths import ECONOMICS_PATH, PROJECT_ROOT


def main():
    wide_ecodata = (
        ecodata
        .with_columns(series_value=pl.col("series_value").cast(pl.Float64))
        .pivot(index="perioddate", on="symbol", values="series_value")
        .sort("perioddate")
    )
    
    # Join daily 3M Eurodollar data from WRDS
    ded3 = (
        pl.read_csv(ECONOMICS_PATH / "ded3_wrds.csv", schema_overrides={"date": pl.Date, "ded3": pl.Float64})
        .select(["date", "ded3"])
        .drop_nulls()
        .sort("date")
    )

    wide_ecodata = (
        wide_ecodata
        .join(ded3, left_on="perioddate", right_on="date", how="right")
        .sort("date")
        .fill_null(strategy="forward")
    )

    # Supplement 3 month LIBOR rates with OECD data
    oecd = (
        pl.read_csv(ECONOMICS_PATH / "oecd.csv")
        .with_columns(
            pl.col("TIME_PERIOD").str.to_date("%Y-%m").alias("date")
        )
        .select(["date", "Reference area", "OBS_VALUE"])
        .sort(["Reference area", "date"])
        .pivot(index="date", on="Reference area", values="OBS_VALUE")
        .sort("date")
        .fill_null(strategy="forward")
    )

    wide_ecodata = wide_ecodata.join(oecd, on="date", how="left").sort("date")
    wide_ecodata.write_csv(ECONOMICS_PATH / "3month_libor_rates.csv")

if __name__ == "__main__":
    futures = load_config(PROJECT_ROOT / "tier1.yaml")
    stirs = list(filter(lambda f: f.asset_class[0] == AssetClass.STIR and f.ecoseriesid, futures))
    
    # Universe's short-term interest rate + additional interbank rate series.
    extra_rates = [
        {
            "ecoseriesid": 134262,
            "symbol": "NOK_rate_monthly",
            "name": "3-Month Norway Interbank Rate (monthly)",
            "dsnumber": 348600511,
            "dsmnemonic": "NWINTER3",
        },
        {
            "ecoseriesid": 136162,
            "symbol": "SEK_rate",
            "name": "3-Month Sweden Interbank Rate",
            "dsnumber": 365411473,
            "dsmnemonic": "SDINTER3",
        },
        {
            "ecoseriesid": 200512,
            "symbol": "NOK_rate",
            "name": "3-Month Norway Interbank Rate (daily)",
            "dsnumber": None,
            "dsmnemonic": "NWIBK3M",
        },
        {
            "ecoseriesid": 200259,
            "symbol": "RMB_rate",
            "name": "China Interbank Rate 3 Month (daily)",
            "dsnumber": None,
            "dsmnemonic": "CHIB3MO",
        },
        {
            "ecoseriesid": 200569,
            "symbol": "6Z_rate",
            "name": "South African JIBAR 3 Month (daily)",
            "dsnumber": None,
            "dsmnemonic": "SAJIB3M",
        },
        {
            "ecoseriesid": 200250,
            "symbol": "6L_rate",
            "name": "Brazil Overnight SELIC (daily)",
            "dsnumber": None,
            "dsmnemonic": "BROVERN",
        },
        {
            "ecoseriesid": 137116,
            "symbol": "CZK_rate",
            "name": "Prague Interbank Offer Rate 3 Month",
            "dsnumber": 640015053,
            "dsmnemonic": "CZINTER3",
        },
    ]

    ecoseriesids = [f.ecoseriesid for f in stirs] + [rate["ecoseriesid"] for rate in extra_rates]

    key = pl.DataFrame({
        "ecoseriesid": [f.ecoseriesid for f in stirs] + [rate["ecoseriesid"] for rate in extra_rates],
        "symbol": [f.symbol for f in stirs] + [rate["symbol"] for rate in extra_rates],
        "name": [f.name for f in stirs] + [rate["name"] for rate in extra_rates],
        "dsnumber": [f.dsnumber for f in stirs] + [rate["dsnumber"] for rate in extra_rates],
        "dsmnemonic": [f.dsmnemonic for f in stirs] + [rate["dsmnemonic"] for rate in extra_rates],
    })

    # Data for the 3 month LIBOR rates series
    ecodata = (
        pl.scan_csv(ECONOMICS_PATH / "ecodata.csv", schema_overrides={"perioddate": pl.Date, "announceddate": pl.Date})
        .filter(pl.col("ecoseriesid").is_in(ecoseriesids))
        .sort(["ecoseriesid", "perioddate"])
        .unique(subset=["ecoseriesid", "perioddate"], keep="last")
        .sort(["ecoseriesid", "perioddate"])
        .collect()
    )
    ecodata = ecodata.join(key, on="ecoseriesid", how="left")

    main()
