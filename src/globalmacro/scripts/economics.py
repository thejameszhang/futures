import polars as pl
from utils.config import load_config
from utils.models import AssetClass
from utils.paths import ECONOMICS_PATH, PROJECT_ROOT


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
    
    # Universe's short-term interest rate + the 3 month Norway and Sweden interbanking rates
    ecoseriesids = [float(f.ecoseriesid) for f in stirs] + [134262.0, 136162.0]

    key = pl.DataFrame({
        "ecoseriesid": ecoseriesids,
        "symbol": [f.symbol for f in stirs] + ["NOK_rate", "SEK_rate"],
        "name": [f.name for f in stirs] + ["3-Month Norway Interbank Rate", "3-Month Sweden Interbank Rate"],
        "dsnumber": [f.dsnumber for f in stirs] + [348600511, 365411473],
        "dsmnemonic": [f.dsmnemonic for f in stirs] + ["NWINTER3", "SDINTER3"],
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