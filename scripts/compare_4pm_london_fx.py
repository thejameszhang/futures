import argparse

import polars as pl

from globalmacro.pipeline.fx import SYMBOL_TO_CURCDD_MAPPING, save_compustat_fx_rates
from globalmacro.utils.paths import COMPUSTAT_PATH, DATASETS_ROOT


def compute_gap_aware_spot_return(futures_df: pl.DataFrame, spot_df: pl.DataFrame, symbol: str, ccy: str) -> pl.DataFrame:
    spot = spot_df.select(["date", ccy]).rename({ccy: "__fx_level"})

    fut = futures_df.select(["date", symbol])
    panel_start = fut.filter(pl.col(symbol).is_not_null()).select(pl.col("date").min()).item()

    if panel_start is None:
        return pl.DataFrame(schema={"date": pl.Date, "r_fut": pl.Float64, "r_spot": pl.Float64})

    seed = spot.filter((pl.col("date") < panel_start) & pl.col("__fx_level").is_not_null()).tail(1)
    fut_filtered = fut.filter(pl.col("date") >= panel_start)

    if seed.height > 0:
        seed_date = seed["date"][0]
        seed_fut = pl.DataFrame({"date": [seed_date], symbol: [None]}, schema={"date": pl.Date, symbol: pl.Float64})
        fut_concat = pl.concat([seed_fut, fut_filtered])
    else:
        fut_concat = fut_filtered

    df = fut_concat.join(spot, on="date", how="left").sort("date")
    df = df.with_columns(pl.col("__fx_level").forward_fill())

    observed = pl.when(pl.col(symbol).is_not_null()).then(pl.col("__fx_level")).otherwise(None)
    df = df.with_columns(observed.alias("__observed_level"))

    prev_observation = pl.col("__observed_level").forward_fill().shift(1)
    denom = pl.coalesce([prev_observation, pl.col("__fx_level").shift(1)])

    df = df.with_columns((pl.col("__fx_level") / denom - 1.0).alias("r_spot"))

    df = df.filter(pl.col("date") >= panel_start)
    df = df.rename({symbol: "r_fut"})
    return df.select(["date", "r_fut", "r_spot"])


def evaluate_dataset(sync_target: str, spot_df: pl.DataFrame, start_date: str | None = None) -> pl.DataFrame:
    results = []

    for tier in [1, 2]:
        sync_dir = "sync_london" if sync_target == "london" else "sync"
        futures_path = DATASETS_ROOT / f"tier{tier}" / sync_dir / "currency_daily_returns.csv"
        if not futures_path.exists():
            continue
        futures_df = pl.read_csv(futures_path, schema_overrides={"date": pl.Date})
        symbol_cols = [c for c in futures_df.columns if c not in ("date", "date_")]
        futures_df = futures_df.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in symbol_cols])

        for symbol in futures_df.columns:
            if symbol in ("date", "date_") or symbol not in SYMBOL_TO_CURCDD_MAPPING:
                continue

            ccy = SYMBOL_TO_CURCDD_MAPPING[symbol]
            if ccy not in spot_df.columns:
                continue

            df = compute_gap_aware_spot_return(futures_df, spot_df, symbol, ccy)

            valid = df.drop_nulls(["r_fut", "r_spot"])
            if start_date:
                start_dt = pl.lit(start_date).str.to_date("%Y-%m-%d")
                valid = valid.filter(pl.col("date") >= start_dt)
            if valid.height < 10:
                continue

            pearson = valid.select(pl.corr("r_fut", "r_spot")).item()
            spearman = valid.select(pl.corr("r_fut", "r_spot", method="spearman")).item()

            results.append({
                "tier": tier,
                "symbol": symbol,
                "ccy": ccy,
                "n_obs": valid.height,
                "pearson": round(pearson, 4),
                "spearman": round(spearman, 4),
            })

    if results:
        return pl.DataFrame(results).sort(["tier", "symbol"])
    return pl.DataFrame()


def run_comparison():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sync_target", choices=["london", "et", "both"], default="both")
    parser.add_argument("--start_date", type=str, default=None, help="Filter start date YYYY-MM-DD")
    args = parser.parse_args()

    spot_df = save_compustat_fx_rates()
    pl.Config.set_tbl_rows(100)

    if args.sync_target in ("london", "et"):
        df = evaluate_dataset(args.sync_target, spot_df, start_date=args.start_date)
        if df.height > 0:
            print(df)
            if args.start_date:
                filename = f"{args.sync_target}_fx_futures_correlation_from_{args.start_date}.csv"
            else:
                filename = f"{args.sync_target}_fx_futures_correlation.csv"
            out_path = COMPUSTAT_PATH / filename
            df.write_csv(out_path)
            print(f"Saved to {out_path}")
        else:
            print("No results to display.")
    else:
        df_london = evaluate_dataset("london", spot_df, start_date=args.start_date)
        df_et = evaluate_dataset("et", spot_df, start_date=args.start_date)
        if df_london.height > 0 and df_et.height > 0:
            df = df_london.join(df_et, on=["tier", "symbol", "ccy"], suffix="_et")
            df = df.rename({
                "n_obs": "n_obs_london",
                "pearson": "pearson_london",
                "spearman": "spearman_london",
            })
            df = df.with_columns(
                (pl.col("pearson_london") - pl.col("pearson_et")).alias("pearson_diff"),
                (pl.col("spearman_london") - pl.col("spearman_et")).alias("spearman_diff"),
            )
            print(df)
            if args.start_date:
                filename = f"fx_futures_correlation_comparison_from_{args.start_date}.csv"
            else:
                filename = "fx_futures_correlation_comparison.csv"
            out_path = COMPUSTAT_PATH / filename
            df.write_csv(out_path)
            print(f"Saved to {out_path}")
        else:
            print("No results to display.")


if __name__ == "__main__":
    run_comparison()
