import argparse
import os
import sys
import warnings
import pandas as pd
import wrds
from utils.paths import DATA_ROOT

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to WRDS
    conn = wrds.Connection()
    use_dbapi = False
    dbapi_conn = None

    # List all tables in the library
    tables = conn.list_tables(library=LIBRARY)
    if TABLES is not None:
        tables = list(filter(lambda x: x in TABLES, tables))
    elif TABLE_PREFIX is not None:
        tables = list(filter(lambda x: x.startswith(TABLE_PREFIX), tables))
    else:
        raise ValueError("Either TABLES or TABLE_PREFIX must be provided")
    
    downloaded_tables = set(os.listdir(OUTPUT_DIR))
    downloaded_tables = set(map(lambda x: x.split(".")[0], downloaded_tables))
    tables = list(set(tables) - downloaded_tables)
    
    if not tables:
        print(f"No tables found in library '{LIBRARY}'.")
        sys.exit(1)

    print(f"Found {len(tables)} tables in '{LIBRARY}'. Downloading {tables} to CSV...")

    try:
        for t in tables:
            fq = f"{LIBRARY}.{t}"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            out_csv = os.path.join(OUTPUT_DIR, t + ".csv")

            try:
                print(f"Querying {fq} ...")
                sql = f"SELECT * FROM {fq};"
                if use_dbapi:
                    df = pd.read_sql_query(sql, dbapi_conn)
                else:
                    df = conn.raw_sql(sql)
            except Exception as e:
                if (not use_dbapi) and ("cursor" in str(e)):
                    use_dbapi = True
                    warnings.filterwarnings(
                        "ignore",
                        message="pandas only supports SQLAlchemy connectable*",
                        category=UserWarning,
                    )
                    dbapi_conn = conn.engine.raw_connection()
                    try:
                        df = pd.read_sql_query(sql, dbapi_conn)
                    except Exception as e2:
                        print(f"Failed {fq}: {e2}")
                        continue
                else:
                    print(f"Failed {fq}: {e}")
                    continue

            if df.empty:
                print(f"{fq} is empty; skipping.")
                continue

            df.to_csv(out_csv)
            print(f"Wrote {out_csv} (rows={len(df)})")
    finally:
        if dbapi_conn is not None:
            dbapi_conn.close()
        conn.close()

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='equities', help='Datastream library to download')
    args = parser.parse_args()
    DATABASE = args.database
    OUTPUT_DIR = DATA_ROOT / "datastream" / DATABASE
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    TABLES = None
    TABLE_PREFIX = None

    if DATABASE == "equities":
        LIBRARY = "tr_ds_equities"
        TABLES = ["ds2datatype", "ds2equityindex", "ds2indexaddldata", "ds2indexdata", "ds2indexdatatype"]
    elif DATABASE == "fx":
        LIBRARY = "tr_ds_equities"
        TABLES = ["ds2fxcode", "ds2fxrate", "ds2mktval", "ds2primqtprc", "ds2primqtri", "ds2scdqtprc", "ds2scdqtri"]
    elif DATABASE == "futures":
        LIBRARY = "tr_ds_fut"
        TABLE_PREFIX = "dsf"
    elif DATABASE == "economics":
        LIBRARY = "tr_ds_econ"
        TABLE_PREFIX = "eco"
    elif DATABASE == "commodities":
        LIBRARY = "tr_ds_comds"
        TABLE_PREFIX = "dsc"
    elif DATABASE == "comp":
        LIBRARY = "comp"
        TABLES = ["exrt_dly"]
    else:
        raise ValueError(f"Invalid database: {DATABASE}")
    
    main()
